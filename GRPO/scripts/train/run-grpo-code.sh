#!/usr/bin/env bash
set -euo pipefail

# Start SandboxFusion for code rewards, then launch the GRPO training script.
# By default, GRPO and SandboxFusion are sibling directories. Override
# SANDBOX_DIR and SANDBOX_PYTHON when they are installed elsewhere.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT=${PROJECT_ROOT:-$(cd -- "${REPO_ROOT}/.." && pwd)}
TRAIN_SCRIPT=${TRAIN_SCRIPT:-${SCRIPT_DIR}/grpo-code.sh}
SANDBOX_DIR=${SANDBOX_DIR:-${PROJECT_ROOT}/SandboxFusion}
SANDBOX_PYTHON=${SANDBOX_PYTHON:-${SANDBOX_DIR}/.venv/bin/python}
SANDBOX_HOST=${SANDBOX_HOST:-127.0.0.1}
SANDBOX_PORT=${SANDBOX_PORT:-8080}
SANDBOX_LOG_DIR=${SANDBOX_LOG_DIR:-${PROJECT_ROOT}/logs/sandboxfusion}

if [[ ! -f "${TRAIN_SCRIPT}" ]]; then
  echo "Training script does not exist: ${TRAIN_SCRIPT}" >&2
  exit 1
fi

# A training dry-run needs neither a sandbox server nor a GPU job.
for arg in "$@"; do
  if [[ "${arg}" == --dry-run ]]; then
    exec bash "${TRAIN_SCRIPT}" "$@"
  fi
done

# An explicit URL may point to a remote SandboxFusion deployment.
if [[ -n "${SANDBOX_FUSION_URL:-}" ]]; then
  exec bash "${TRAIN_SCRIPT}" "$@"
fi

SANDBOX_BASE_URL="http://${SANDBOX_HOST}:${SANDBOX_PORT}"
export SANDBOX_FUSION_URL="${SANDBOX_BASE_URL}/run_code"
SANDBOX_LOG="${SANDBOX_LOG_DIR}/sandboxfusion.log"
SANDBOX_PID_FILE="${SANDBOX_LOG_DIR}/sandboxfusion.pid"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to check SandboxFusion readiness." >&2
  exit 1
fi

check_sandbox() {
  [[ "$(curl --noproxy '*' --fail --silent --show-error --max-time 3 \
    "${SANDBOX_BASE_URL}/v1/ping" 2>/dev/null)" == '"pong"' ]]
}

if check_sandbox; then
  echo "SandboxFusion is already running at ${SANDBOX_BASE_URL}."
else
  if [[ ! -d "${SANDBOX_DIR}" ]]; then
    echo "SandboxFusion directory does not exist: ${SANDBOX_DIR}" >&2
    exit 1
  fi
  if [[ ! -x "${SANDBOX_PYTHON}" ]]; then
    echo "SandboxFusion Python is not executable: ${SANDBOX_PYTHON}" >&2
    echo "Set SANDBOX_PYTHON to the Python executable with SandboxFusion dependencies." >&2
    exit 1
  fi

  mkdir -p "${SANDBOX_LOG_DIR}"
  (
    cd "${SANDBOX_DIR}"
    nohup "${SANDBOX_PYTHON}" -m uvicorn sandbox.server.server:app \
      --host "${SANDBOX_HOST}" --port "${SANDBOX_PORT}" --log-level info \
      >> "${SANDBOX_LOG}" 2>&1 < /dev/null &
    echo $! > "${SANDBOX_PID_FILE}"
  )
  sandbox_pid=$(<"${SANDBOX_PID_FILE}")
  echo "Started SandboxFusion (PID ${sandbox_pid}); waiting for readiness."

  ready=false
  for ((attempt=1; attempt<=60; attempt++)); do
    if check_sandbox; then
      ready=true
      break
    fi
    if ! kill -0 "${sandbox_pid}" 2>/dev/null; then
      echo "SandboxFusion exited before it became ready." >&2
      tail -n 50 "${SANDBOX_LOG}" >&2
      exit 1
    fi
    sleep 2
  done
  if [[ "${ready}" != true ]]; then
    echo "SandboxFusion did not become ready within 120 seconds." >&2
    tail -n 50 "${SANDBOX_LOG}" >&2
    exit 1
  fi
fi

echo "Starting GRPO code training with ${SANDBOX_FUSION_URL}."
exec bash "${TRAIN_SCRIPT}" "$@"
