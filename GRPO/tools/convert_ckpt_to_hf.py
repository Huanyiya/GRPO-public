#!/usr/bin/env bash
# Convert every ALFWorld Qwen3.5-2B GRPO torch_dist checkpoint to Hugging Face.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"

CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-${PROJECT_ROOT}/checkpoints/webshop/Qwen3.5-2B-GRPO}"
BASE_MODEL_DIR="${BASE_MODEL_DIR:-${PROJECT_ROOT}/models/Qwen3.5-2B}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/checkpoints_hf/webshop/Qwen3.5-2B-GRPO}"
MEGATRON_PATH="${MEGATRON_PATH:-${PROJECT_ROOT}/Megatron-LM}"
PYTHON_BIN="/mnt/cpfs/users/wxh/slime-OPD/.venv/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python is not executable: ${PYTHON_BIN}" >&2
    exit 1
fi

if [[ ! -d "${CHECKPOINT_ROOT}" ]]; then
    echo "Missing checkpoint root: ${CHECKPOINT_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${BASE_MODEL_DIR}/config.json" ]]; then
    echo "Missing Hugging Face base model in ${BASE_MODEL_DIR}" >&2
    exit 1
fi
if [[ ! -d "${MEGATRON_PATH}/megatron" ]]; then
    echo "Missing Megatron-LM in ${MEGATRON_PATH}" >&2
    exit 1
fi
export PYTHONPATH="${REPO_ROOT}:${MEGATRON_PATH}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-${PROJECT_ROOT}}"

VOCAB_SIZE="$("${PYTHON_BIN}" - "${BASE_MODEL_DIR}/config.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as reader:
    config = json.load(reader)
if config.get("model_type") != "qwen3_5":
    raise SystemExit("Expected a Qwen3.5 base model")
vocab_size = config.get("text_config", {}).get("vocab_size")
if not isinstance(vocab_size, int) or vocab_size <= 0:
    raise SystemExit("Could not determine Qwen3.5 text vocabulary size")
print(vocab_size)
PY
)"

mkdir -p "${PROJECT_ROOT}/checkpoints_hf/alfworld" "${OUTPUT_ROOT}"

validate_output() {
    local output_dir=$1
    "${PYTHON_BIN}" - "${output_dir}" "${BASE_MODEL_DIR}" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
base_dir = Path(sys.argv[2])
index_path = output_dir / "model.safetensors.index.json"
if not (output_dir / "config.json").is_file() or not index_path.is_file():
    raise SystemExit("Conversion did not produce a complete Hugging Face model")
with index_path.open(encoding="utf-8") as reader:
    index = json.load(reader)
weight_map = index.get("weight_map", {})
if not weight_map or any(not (output_dir / name).is_file() for name in set(weight_map.values())):
    raise SystemExit("Converted model index refers to missing weight files")
base_index_path = base_dir / "model.safetensors.index.json"
if base_index_path.is_file():
    with base_index_path.open(encoding="utf-8") as reader:
        base_keys = set(json.load(reader)["weight_map"])
    missing = base_keys - set(weight_map)
    if missing:
        raise SystemExit(f"Converted checkpoint is missing {len(missing)} base model tensors")
print(f"Validated {len(weight_map)} model tensors in {len(set(weight_map.values()))} files")
PY
}

mapfile -d '' CHECKPOINT_DIRS < <(
    find "${CHECKPOINT_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'iter_*' -print0 | sort -z
)
if [[ ${#CHECKPOINT_DIRS[@]} -eq 0 ]]; then
    echo "No iter_* checkpoint directories found in ${CHECKPOINT_ROOT}" >&2
    exit 1
fi

echo "Base model: ${BASE_MODEL_DIR}"
echo "Checkpoint root: ${CHECKPOINT_ROOT}"
echo "Output root: ${OUTPUT_ROOT}"

for CHECKPOINT_DIR in "${CHECKPOINT_DIRS[@]}"; do
    if [[ ! -f "${CHECKPOINT_DIR}/.metadata" || ! -f "${CHECKPOINT_DIR}/common.pt" ]]; then
        echo "Skipping incomplete torch_dist checkpoint: ${CHECKPOINT_DIR}" >&2
        continue
    fi

    CHECKPOINT_NAME=$(basename -- "${CHECKPOINT_DIR}")
    OUTPUT_DIR="${OUTPUT_ROOT}/${CHECKPOINT_NAME}"
    if [[ -e "${OUTPUT_DIR}" ]]; then
        echo "Output already exists; validating and skipping: ${OUTPUT_DIR}"
        validate_output "${OUTPUT_DIR}"
        continue
    fi

    PARTIAL_DIR="${OUTPUT_DIR}.partial.$$"
    if [[ -e "${PARTIAL_DIR}" ]]; then
        echo "Temporary output already exists: ${PARTIAL_DIR}" >&2
        exit 1
    fi
    cleanup() {
        if [[ -d "${PARTIAL_DIR}" ]]; then
            rm -rf -- "${PARTIAL_DIR}"
        fi
    }
    trap cleanup EXIT

    echo "Converting ${CHECKPOINT_DIR}"
    echo "Output: ${OUTPUT_DIR}"
    "${PYTHON_BIN}" "${REPO_ROOT}/tools/convert_torch_dist_to_hf.py" \
        --model-name qwen3_5 \
        --input-dir "${CHECKPOINT_DIR}" \
        --output-dir "${PARTIAL_DIR}" \
        --origin-hf-dir "${BASE_MODEL_DIR}" \
        --add-missing-from-origin-hf \
        --vocab-size "${VOCAB_SIZE}"

    validate_output "${PARTIAL_DIR}"
    mv -- "${PARTIAL_DIR}" "${OUTPUT_DIR}"
    trap - EXIT
    echo "Hugging Face checkpoint ready: ${OUTPUT_DIR}"
done
