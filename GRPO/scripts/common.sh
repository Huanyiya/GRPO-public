#!/usr/bin/env bash

# Shared launcher paths. PROJECT_ROOT is the directory containing this repo.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "${REPO_ROOT}/.." && pwd)}

MODEL_ROOT=${MODEL_ROOT:-${PROJECT_ROOT}/models}
MODEL_NAME="Qwen3.5-${MODEL_SIZE}"
MODEL_PATH="${MODEL_ROOT}/${MODEL_NAME}"
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints}
LOG_ROOT=${LOG_ROOT:-${PROJECT_ROOT}/logs}
# VENV_ROOT=${VENV_ROOT:-${REPO_ROOT}/.venv}
VENV_ROOT="/mnt/cpfs/users/zhy/opd/slime-OPD/.venv"
PYTHON_BIN=${PYTHON_BIN:-${VENV_ROOT}/bin/python}
RAY_BIN=${RAY_BIN:-${VENV_ROOT}/bin/ray}
MEGATRON_PATH=${MEGATRON_PATH:-${PROJECT_ROOT}/Megatron-LM}

NUM_GPUS=${NUM_GPUS:-8}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-6379}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-8265}
SOCKET_IFNAME=${SOCKET_IFNAME:-eth0}
# The first compiled SGLang forward may exceed the default 20-second health check.
SGLANG_HEALTH_CHECK_TIMEOUT=${SGLANG_HEALTH_CHECK_TIMEOUT:-120}
SGLANG_WARMUP_TIMEOUT=${SGLANG_WARMUP_TIMEOUT:-1800}
WEIGHT_SYNC_TRANSPORT=${WEIGHT_SYNC_TRANSPORT:-nccl}

GRPO_DRY_RUN=false
GRPO_CLI_ARGS=()
declare -A GRPO_CLI_KEYS=()
declare -A GRPO_CLI_LAST_VALUE=()

grpo_option_key() {
  local option=${1%%=*}
  case "${option}" in
    --no-offload-train) printf '%s' --offload-train ;;
    --no-offload-rollout) printf '%s' --offload-rollout ;;
    *) printf '%s' "${option}" ;;
  esac
}

# Keep complete option groups, including multi-value options such as --spec
# and --eval-prompt-data. A value beginning with -- must use --option=value.
grpo_parse_cli() {
  local token option key value
  while (($#)); do
    token=$1
    shift
    if [[ "${token}" == --dry-run ]]; then
      GRPO_DRY_RUN=true
      continue
    fi
    if [[ "${token}" != --* ]]; then
      echo "Expected a train.py option, got: ${token}" >&2
      return 2
    fi
    option=${token%%=*}
    key=$(grpo_option_key "${option}")
    GRPO_CLI_KEYS["${key}"]=1
    GRPO_CLI_ARGS+=("${token}")
    if [[ "${token}" == *=* ]]; then
      GRPO_CLI_LAST_VALUE["${option}"]=${token#*=}
    else
      value=
      while (($#)) && [[ "$1" != --* ]]; do
        value=$1
        GRPO_CLI_ARGS+=("$1")
        shift
      done
      GRPO_CLI_LAST_VALUE["${option}"]=${value}
    fi
  done
}

# Build one final argv. Command-line groups replace matching defaults before
# train.py sees them, including both sides of BooleanOptionalAction options.
grpo_merge_train_args() {
  local -a defaults=("$@") group=()
  local -A last_cli_index=()
  local i=0 start token key
  GRPO_TRAIN_ARGS=()
  while ((i < ${#defaults[@]})); do
    token=${defaults[i]}
    if [[ "${token}" != --* ]]; then
      echo "Invalid default argument: ${token}" >&2
      return 2
    fi
    key=$(grpo_option_key "${token}")
    group=("${token}")
    i=$((i + 1))
    while ((i < ${#defaults[@]})) && [[ "${defaults[i]}" != --* ]]; do
      group+=("${defaults[i]}")
      i=$((i + 1))
    done
    if [[ ! -v GRPO_CLI_KEYS["${key}"] ]]; then
      GRPO_TRAIN_ARGS+=("${group[@]}")
    fi
  done

  i=0
  while ((i < ${#GRPO_CLI_ARGS[@]})); do
    start=$i
    key=$(grpo_option_key "${GRPO_CLI_ARGS[i]}")
    last_cli_index["${key}"]=$start
    i=$((i + 1))
    while ((i < ${#GRPO_CLI_ARGS[@]})) && [[ "${GRPO_CLI_ARGS[i]}" != --* ]]; do
      i=$((i + 1))
    done
  done

  i=0
  while ((i < ${#GRPO_CLI_ARGS[@]})); do
    start=$i
    key=$(grpo_option_key "${GRPO_CLI_ARGS[i]}")
    group=("${GRPO_CLI_ARGS[i]}")
    i=$((i + 1))
    while ((i < ${#GRPO_CLI_ARGS[@]})) && [[ "${GRPO_CLI_ARGS[i]}" != --* ]]; do
      group+=("${GRPO_CLI_ARGS[i]}")
      i=$((i + 1))
    done
    if [[ ${last_cli_index["${key}"]} == "${start}" ]]; then
      GRPO_TRAIN_ARGS+=("${group[@]}")
    fi
  done
}

grpo_print_dry_run() {
  printf 'REPO_ROOT=%s\nMODEL_DIR=%s\nINIT_CKPT=%s\nDATA_PATH=%s\nEVAL_DATA_PATH=%s\nLOAD_DIR=%s\nSAVE_DIR=%s\nPYTHON_BIN=%s\nRAY_BIN=%s\nMEGATRON_PATH=%s\n' \
    "${REPO_ROOT}" "${MODEL_DIR}" "${INIT_CKPT}" "${DATA_PATH}" "${EVAL_DATA_PATH}" \
    "${LOAD_DIR}" "${SAVE_DIR}" "${PYTHON_BIN}" "${RAY_BIN}" "${MEGATRON_PATH}"
  printf 'train.py argv:\n'
  local i=0 token
  while ((i < ${#GRPO_TRAIN_ARGS[@]})); do
    token=${GRPO_TRAIN_ARGS[i]}
    if [[ "${token}" == --wandb-key ]]; then
      printf '  --wandb-key <redacted>\n'
      i=$((i + 2))
      continue
    fi
    if [[ "${token}" == --wandb-key=* ]]; then
      printf '  --wandb-key=<redacted>\n'
      i=$((i + 1))
      continue
    fi
    printf '  %q' "${token}"
    i=$((i + 1))
    while ((i < ${#GRPO_TRAIN_ARGS[@]})) && [[ "${GRPO_TRAIN_ARGS[i]}" != --* ]]; do
      printf ' %q' "${GRPO_TRAIN_ARGS[i]}"
      i=$((i + 1))
    done
    printf '\n'
  done
}
