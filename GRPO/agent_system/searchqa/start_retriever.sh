#!/usr/bin/env bash

# Run this repository's existing Search-R1 retriever on port 8000.
# The encoder requires a GPU; FAISS_GPU=1 additionally places the index on GPUs.
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
PROJECT_ROOT="$(cd -- "${REPO_ROOT}/.." && pwd)"
DATA_ROOT=${DATA_ROOT:-${PROJECT_ROOT}/data}
SEARCH_ASSET_ROOT=${SEARCH_ASSET_ROOT:-${DATA_ROOT}/searchR1}
RETRIEVER_MODEL=${RETRIEVER_MODEL:-${PROJECT_ROOT}/models/e5-base-v2}
PYTHON_BIN=${PYTHON_BIN:-python}

for required_path in \
  "${SEARCH_ASSET_ROOT}/e5_Flat.index" \
  "${SEARCH_ASSET_ROOT}/wiki-18.jsonl" \
  "${RETRIEVER_MODEL}/config.json"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "Missing Search-R1 retrieval asset: ${required_path}" >&2
    exit 1
  fi
done

EXTRA_ARGS=()
if [[ "${FAISS_GPU:-0}" == 1 ]]; then EXTRA_ARGS+=(--faiss_gpu); fi

exec "${PYTHON_BIN}" "${REPO_ROOT}/examples/search-r1/local_dense_retriever/retrieval_server.py" \
  --index_path "${SEARCH_ASSET_ROOT}/e5_Flat.index" \
  --corpus_path "${SEARCH_ASSET_ROOT}/wiki-18.jsonl" \
  --retriever_name e5 \
  --retriever_model "${RETRIEVER_MODEL}" \
  --topk 3 \
  "${EXTRA_ARGS[@]}"
