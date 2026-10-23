#!/usr/bin/env bash
# Check rollout PPUs every 60 seconds.
# Once all PPUs are idle, wait 5 minutes and confirm they are still idle.
# Then collect DAPO rollouts.
# A failed collection is cleaned up and retried at most twice.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET_SCRIPT="${TARGET_SCRIPT:-/mnt/cpfs/users/wxh/github-source/GRPO/scripts/train/run-grpo-code.sh}"
PROJECT_ROOT="${PROJECT_ROOT:-$(cd -- "${SCRIPT_DIR}/../../.." && pwd)}"
export SANDBOX_DIR="${SANDBOX_DIR:-${PROJECT_ROOT}/SandboxFusion}"
export SANDBOX_PYTHON="${SANDBOX_PYTHON:-/mnt/cpfs/users/wxh/miniconda3/envs/sandboxfusion-server/bin/python}"

# The target launcher sets CUDA_VISIBLE_DEVICES=0,...,7 and uses DP=8.
PPU_IDS_CSV="${PPU_IDS_CSV:-0,1,2,3,4,5,6,7}"
MEM_THRESHOLD_MIB="${MEM_THRESHOLD_MIB:-1024}"

# Normal polling interval: check every 1 minute.
CHECK_INTERVAL_SECONDS="${CHECK_INTERVAL_SECONDS:-120}"

# After the first idle check succeeds, wait 5 minutes before confirming again.
IDLE_CONFIRM_INTERVAL_SECONDS="${IDLE_CONFIRM_INTERVAL_SECONDS:-120}"

REQUIRED_CONSECUTIVE_CHECKS=2
MAX_ATTEMPTS="${MAX_ATTEMPTS:-3}"

PPU_SMI_BIN="${PPU_SMI_BIN:-$(command -v ppu-smi || true)}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/auto_collect_logs}"
LOCK_FILE="${LOCK_FILE:-/tmp/zhy_auto_collect_dapo_2correct_2incorrect.lock}"

IFS=',' read -r -a PPU_IDS <<< "${PPU_IDS_CSV}"
CURRENT_LEADER_PID=""
CURRENT_PGID=""

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

require_valid_configuration() {
    if [[ ! -f "${TARGET_SCRIPT}" ]]; then
        echo "Target script does not exist: ${TARGET_SCRIPT}" >&2
        exit 1
    fi
    if [[ -z "${PPU_SMI_BIN}" || ! -x "${PPU_SMI_BIN}" ]]; then
        echo "Cannot find executable ppu-smi. Set PPU_SMI_BIN explicitly." >&2
        exit 1
    fi
    if ! command -v setsid >/dev/null 2>&1; then
        echo "setsid is required to isolate and clean up one attempt." >&2
        exit 1
    fi
    if (( ${#PPU_IDS[@]} == 0 )); then
        echo "PPU_IDS_CSV must name at least one PPU." >&2
        exit 1
    fi
    if [[ ! "${MEM_THRESHOLD_MIB}" =~ ^[0-9]+$ ]] || (( MEM_THRESHOLD_MIB == 0 )); then
        echo "MEM_THRESHOLD_MIB must be a positive integer." >&2
        exit 1
    fi
    if [[ ! "${CHECK_INTERVAL_SECONDS}" =~ ^[0-9]+$ ]] || (( CHECK_INTERVAL_SECONDS == 0 )); then
        echo "CHECK_INTERVAL_SECONDS must be a positive integer." >&2
        exit 1
    fi
    if [[ ! "${IDLE_CONFIRM_INTERVAL_SECONDS}" =~ ^[0-9]+$ ]] || (( IDLE_CONFIRM_INTERVAL_SECONDS == 0 )); then
        echo "IDLE_CONFIRM_INTERVAL_SECONDS must be a positive integer." >&2
        exit 1
    fi
    if [[ ! "${MAX_ATTEMPTS}" =~ ^[0-9]+$ ]] || (( MAX_ATTEMPTS == 0 )); then
        echo "MAX_ATTEMPTS must be a positive integer." >&2
        exit 1
    fi
}

all_ppus_below_threshold() {
    local ppu_id raw_value value
    local -a memory_report=()

    for ppu_id in "${PPU_IDS[@]}"; do
        if ! raw_value="$("${PPU_SMI_BIN}" \
            -i "${ppu_id}" \
            --query-ppu=memory.used \
            --format=csv,noheader,nounits 2>&1)"; then
            log "Cannot query PPU${ppu_id}: ${raw_value}"
            return 1
        fi

        value="${raw_value//[[:space:]]/}"
        if [[ ! "${value}" =~ ^[0-9]+$ ]]; then
            log "Invalid memory value for PPU${ppu_id}: ${raw_value}"
            return 1
        fi
        memory_report+=("PPU${ppu_id}=${value}MiB")

        if (( value >= MEM_THRESHOLD_MIB )); then
            log "PPU memory: ${memory_report[*]} (not idle: threshold ${MEM_THRESHOLD_MIB}MiB)"
            return 1
        fi
    done

    log "PPU memory: ${memory_report[*]} (all below ${MEM_THRESHOLD_MIB}MiB)"
    return 0
}

wait_for_two_idle_checks() {
    local consecutive=0

    while (( consecutive < REQUIRED_CONSECUTIVE_CHECKS )); do
        if all_ppus_below_threshold; then
            ((consecutive += 1))
            log "Idle checks: ${consecutive}/${REQUIRED_CONSECUTIVE_CHECKS}"

            if (( consecutive == 1 )); then
                log "First idle check passed; confirming again in ${IDLE_CONFIRM_INTERVAL_SECONDS}s."
                sleep "${IDLE_CONFIRM_INTERVAL_SECONDS}"
            fi
        else
            consecutive=0
            log "Idle checks reset."
            log "Next PPU check in ${CHECK_INTERVAL_SECONDS}s."
            sleep "${CHECK_INTERVAL_SECONDS}"
        fi
    done
}

terminate_current_attempt() {
    local pgid="${CURRENT_PGID}"

    if [[ -z "${pgid}" && -z "${CURRENT_LEADER_PID}" ]]; then
        return
    fi

    if [[ ! "${pgid}" =~ ^[0-9]+$ ]]; then
        log "Refusing unsafe process-group cleanup (pgid=${pgid:-unset})."
        if [[ "${CURRENT_LEADER_PID}" =~ ^[0-9]+$ ]] && kill -0 "${CURRENT_LEADER_PID}" 2>/dev/null; then
            kill -TERM "${CURRENT_LEADER_PID}" 2>/dev/null || true
        fi
        CURRENT_LEADER_PID=""
        CURRENT_PGID=""
        return
    fi

    if ! kill -0 -- "-${pgid}" 2>/dev/null; then
        CURRENT_LEADER_PID=""
        CURRENT_PGID=""
        return
    fi

    log "Stopping failed attempt process group ${pgid}."
    kill -TERM -- "-${pgid}" 2>/dev/null || true

    local _
    for _ in $(seq 1 15); do
        if ! kill -0 -- "-${pgid}" 2>/dev/null; then
            CURRENT_LEADER_PID=""
            CURRENT_PGID=""
            return
        fi
        sleep 1
    done

    log "Process group ${pgid} survived SIGTERM; sending SIGKILL."
    kill -KILL -- "-${pgid}" 2>/dev/null || true
    CURRENT_LEADER_PID=""
    CURRENT_PGID=""
}

on_exit() {
    local exit_code=$?
    trap - EXIT INT TERM
    terminate_current_attempt
    exit "${exit_code}"
}
trap on_exit EXIT INT TERM

run_one_attempt() {
    local attempt=$1
    local attempt_log="${LOG_DIR}/attempt_${attempt}_$(date '+%Y%m%d_%H%M%S').log"
    local exit_code

    log "Starting attempt ${attempt}/${MAX_ATTEMPTS}; target log: ${attempt_log}"

    # Stop existing Ray processes before launching the target script.
    log "Running ray stop --force before attempt ${attempt}."
    ray stop --force

    # A separate session/process group lets cleanup affect only this attempt,
    # including its SGLang server and worker children, never unrelated jobs.
    setsid bash "${TARGET_SCRIPT}" >"${attempt_log}" 2>&1 &
    CURRENT_LEADER_PID=$!

    # In a non-interactive watcher, setsid execs bash in a new session whose
    # leader and process-group ID are the just-created child PID.
    CURRENT_PGID="${CURRENT_LEADER_PID}"
    log "Attempt ${attempt} leader PID=${CURRENT_LEADER_PID}, PGID=${CURRENT_PGID}."

    set +e
    wait "${CURRENT_LEADER_PID}"
    exit_code=$?
    set -e

    if (( exit_code == 0 )); then
        log "Attempt ${attempt} finished successfully."
        CURRENT_LEADER_PID=""
        CURRENT_PGID=""
        return 0
    fi

    log "Attempt ${attempt} failed with exit code ${exit_code}."
    tail -n 80 "${attempt_log}" >&2 || true
    terminate_current_attempt
    return 1
}

mkdir -p "${LOG_DIR}"
require_valid_configuration

exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
    echo "Another auto-collection watcher is already running: ${LOCK_FILE}" >&2
    exit 1
fi

log "Watcher started."
log "Target: ${TARGET_SCRIPT}"
log "PPUs: ${PPU_IDS_CSV}; condition: all memory < ${MEM_THRESHOLD_MIB}MiB"
log "Normal check interval: ${CHECK_INTERVAL_SECONDS}s."
log "After first idle check: confirm again after ${IDLE_CONFIRM_INTERVAL_SECONDS}s."
log "Max attempts: ${MAX_ATTEMPTS}."

for attempt in $(seq 1 "${MAX_ATTEMPTS}"); do
    wait_for_two_idle_checks
    if run_one_attempt "${attempt}"; then
        log "Collection succeeded; watcher exiting."
        exit 0
    fi
done

log "Collection failed after ${MAX_ATTEMPTS} attempts."
exit 1
