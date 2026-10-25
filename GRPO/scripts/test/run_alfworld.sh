#!/usr/bin/env bash
# Standalone SDA-style ALFWorld evaluation for a converted Hugging Face model.

set -euo pipefail

TEST_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${TEST_DIR}/../.." && pwd)"
PROJECT_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"

MODEL_DIR="${MODEL_DIR:/mnt/cpfs/users/wxh/github-source/checkpoints_hf/alfworld/Qwen3.5-2B-GRPO/iter_0000099}"
ALFWORLD_DATA="${ALFWORLD_DATA:-${DATA_ROOT:-${PROJECT_ROOT}/data}/alfworld}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PORT="${PORT:-30000}"
TP_SIZE="${TP_SIZE:-1}"
EVAL_SEED="${EVAL_SEED:-0}"
TASK_SEED="${TASK_SEED:-0}"
TEMPERATURE="${TEMPERATURE:-1.0}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/logs/alfworld_eval/$(basename -- "${MODEL_DIR%/}")_seed${EVAL_SEED}_$(date +%Y%m%d_%H%M%S)}"
SERVER_URL="${SERVER_URL:-http://127.0.0.1:${PORT}}"

if [[ ! -f "${MODEL_DIR}/config.json" ]]; then
  echo "Expected converted Hugging Face model at ${MODEL_DIR}" >&2
  exit 1
fi
if [[ ! -d "${ALFWORLD_DATA}/json_2.1.1/valid_seen" ]]; then
  echo "Missing ALFWorld valid_seen under ${ALFWORLD_DATA}" >&2
  exit 1
fi

export ALFWORLD_DATA
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export SGLANG_HEALTH_CHECK_TIMEOUT="${SGLANG_HEALTH_CHECK_TIMEOUT:-120}"
export SGLANG_WARMUP_TIMEOUT="${SGLANG_WARMUP_TIMEOUT:-1800}"

"${PYTHON_BIN}" "${TEST_DIR}/eval_alfworld.py" \
  --model-path "${MODEL_DIR}" \
  --alfworld-data "${ALFWORLD_DATA}" \
  --output-dir "${OUTPUT_DIR}" \
  --seed "${EVAL_SEED}" \
  --task-seed "${TASK_SEED}" \
  --temperature "${TEMPERATURE}" \
  --check-only \
  "$@"

# Set START_SERVER=0 and SERVER_URL to use an already running SGLang server.
if [[ "${START_SERVER:-1}" == "1" ]]; then
  SERVER_LOG="${OUTPUT_DIR}/sglang.log"
  mkdir -p "${OUTPUT_DIR}"
  "${PYTHON_BIN}" -m sglang.launch_server \
    --model-path "${MODEL_DIR}" \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --tensor-parallel-size "${TP_SIZE}" \
    --context-length 2560 \
    --mem-fraction-static "${SGLANG_MEM_FRACTION:-0.7}" \
    --mamba-scheduler-strategy extra_buffer \
    --random-seed "${EVAL_SEED}" \
    >"${SERVER_LOG}" 2>&1 &
  SERVER_PID=$!
  cleanup() {
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  }
  trap cleanup EXIT

  for ((attempt = 0; attempt < 180; attempt++)); do
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "SGLang exited during startup; see ${SERVER_LOG}" >&2
      exit 1
    fi
    if "${PYTHON_BIN}" -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1] + "/health_generate", timeout=2)' "${SERVER_URL%/}" >/dev/null 2>&1; then
      break
    fi
    sleep 5
  done
  if ! "${PYTHON_BIN}" -c 'import sys, urllib.request; urllib.request.urlopen(sys.argv[1] + "/health_generate", timeout=2)' "${SERVER_URL%/}" >/dev/null 2>&1; then
    echo "Timed out waiting for SGLang; see ${SERVER_LOG}" >&2
    exit 1
  fi
fi

cd "${REPO_ROOT}"
"${PYTHON_BIN}" "${TEST_DIR}/eval_alfworld.py" \
  --model-path "${MODEL_DIR}" \
  --alfworld-data "${ALFWORLD_DATA}" \
  --output-dir "${OUTPUT_DIR}" \
  --server-url "${SERVER_URL}" \
  --seed "${EVAL_SEED}" \
  --task-seed "${TASK_SEED}" \
  --temperature "${TEMPERATURE}" \
  "$@"
