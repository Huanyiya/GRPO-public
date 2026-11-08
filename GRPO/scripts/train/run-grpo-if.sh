#!/usr/bin/env bash

# GRPO on the local instruction-following Parquet dataset.
# Uses open-instruct's IFEvalG constraints for the IF reward.

set -euo pipefail
unset PYTORCH_CUDA_ALLOC_CONF

MODEL_SIZE=${MODEL_SIZE:-4B}
TASK_NAME=instruction_following
NUM_GPUS=${NUM_GPUS:-8}

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
cd "${REPO_ROOT}"
grpo_parse_cli "$@"

GPU_IDS=${GPU_IDS:-$(seq -s, 0 "$((NUM_GPUS - 1))")}
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
MODEL_DIR=${GRPO_CLI_LAST_VALUE[--hf-checkpoint]:-${MODEL_DIR:-${MODEL_PATH}}}
INIT_CKPT=${GRPO_CLI_LAST_VALUE[--ref-load]:-${INIT_CKPT:-${MODEL_ROOT}/${MODEL_NAME}_torch_dist}}
SAVE_DIR=${GRPO_CLI_LAST_VALUE[--save]:-${SAVE_DIR:-${CHECKPOINT_ROOT}/${TASK_NAME}/${MODEL_NAME}-GRPO}}
LOAD_DIR=${GRPO_CLI_LAST_VALUE[--load]:-${LOAD_DIR:-${SAVE_DIR}}}
DATA_PATH=${GRPO_CLI_LAST_VALUE[--prompt-data]:-${IF_TRAIN_DATA:-${DATA_ROOT}/instruction_following/raw/data/train-00000-of-00001.parquet}}
OPEN_INSTRUCT_ROOT=${OPEN_INSTRUCT_ROOT:-${PROJECT_ROOT}/open-instruct}
IF_NLTK_DATA=${IF_NLTK_DATA:-${DATA_ROOT}/instruction_following/nltk_data}
IF_REWARD_PATH=${IF_REWARD_PATH:-slime_plugins.if_eval.reward.reward_func}
EVAL_DATA_PATH=
SWANLAB_DIR=${SWANLAB_DIR:-${LOG_ROOT}/${TASK_NAME}/${MODEL_NAME}/swanlab}
DEFAULT_RAY_TMP_DIR=/tmp/ray-if-$$
RAY_TMP_DIR=${RAY_TMP_DIR:-${DEFAULT_RAY_TMP_DIR}}
DEFAULT_WORKER_TMP_DIR=/tmp/if-worker-$$
WORKER_TMP_DIR=${WORKER_TMP_DIR:-${DEFAULT_WORKER_TMP_DIR}}
RAY_LOG_ARCHIVE_DIR=${RAY_LOG_ARCHIVE_DIR:-${LOG_ROOT}/ray}
SGLANG_CONFIG_PATH=${WORKER_TMP_DIR}/sglang-text-only.yaml

MODEL_CONFIG="${SCRIPT_DIR}/models/qwen3.5-${MODEL_SIZE}.sh"
if [[ ! -f "${MODEL_CONFIG}" ]]; then
  echo "Model configuration not found: ${MODEL_CONFIG}" >&2
  exit 1
fi
source "${MODEL_CONFIG}"

export PYTHONUNBUFFERED=1 MASTER_ADDR MASTER_PORT
export SGLANG_HEALTH_CHECK_TIMEOUT SGLANG_WARMUP_TIMEOUT
export RAY_DISABLE_DASHBOARD_GPU_METRICS=1 RAY_TMP_DIR
export TMPDIR="${WORKER_TMP_DIR}" TMP="${WORKER_TMP_DIR}" TEMP="${WORKER_TMP_DIR}"
export PYTHONPATH="${REPO_ROOT}:${OPEN_INSTRUCT_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export NLTK_DATA="${IF_NLTK_DATA}${NLTK_DATA:+:${NLTK_DATA}}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ftp_proxy FTP_PROXY NO_PROXY RAY_ADDRESS
export no_proxy="localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}"
export NO_PROXY="${no_proxy}"

if [[ "${GRPO_DRY_RUN}" != true ]]; then
  mkdir -p "${WORKER_TMP_DIR}"
  for required_path in \
    "${DATA_PATH}" "${MODEL_DIR}" "${INIT_CKPT}" \
    "${PYTHON_BIN}" "${RAY_BIN}" "${MEGATRON_PATH}" "${OPEN_INSTRUCT_ROOT}"; do
    if [[ ! -e "${required_path}" ]]; then
      echo "Required IF or training path does not exist: ${required_path}" >&2
      exit 1
    fi
  done
  if [[ ! -d "${IF_NLTK_DATA}/tokenizers/punkt_tab/english" ]]; then
    echo "Missing NLTK punkt_tab data under ${IF_NLTK_DATA}. Download it with: ${PYTHON_BIN} -m nltk.downloader -d ${IF_NLTK_DATA} punkt_tab" >&2
    exit 1
  fi
  "${PYTHON_BIN}" - "${DATA_PATH}" <<'PY'
import sys
import pyarrow.parquet as pq

parquet = pq.ParquetFile(sys.argv[1])
required = {"messages", "ground_truth"}
missing = required - set(parquet.schema_arrow.names)
if missing or parquet.metadata.num_rows == 0:
    raise ValueError(f"Invalid IF data: missing={sorted(missing)}, rows={parquet.metadata.num_rows}")
row = parquet.read_row_group(0, columns=["messages", "ground_truth"]).slice(0, 1).to_pylist()[0]
if not row["messages"] or not row["ground_truth"]:
    raise ValueError("IF data needs nonempty messages and ground_truth")
PY
  "${PYTHON_BIN}" -c 'import nltk; nltk.data.find("tokenizers/punkt_tab/english/"); from open_instruct.IFEvalG import instructions_registry; import slime_plugins.if_eval.reward, slime_plugins.if_eval.metrics'
  mkdir -p "${SWANLAB_DIR}" "${RAY_TMP_DIR}" "${RAY_LOG_ARCHIVE_DIR}" "${WORKER_TMP_DIR}"
  cat > "${SGLANG_CONFIG_PATH}" <<EOF
sglang:
  - name: default
    server_groups:
      - worker_type: regular
        num_gpus: ${NUM_GPUS}
        num_gpus_per_engine: 1
        overrides:
          enable_multimodal: false
EOF
fi

NVLINK_COUNT=0
if [[ "${GRPO_DRY_RUN}" != true ]] && command -v nvidia-smi >/dev/null 2>&1; then
  NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
fi
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then HAS_NVLINK=1; else HAS_NVLINK=0; fi
export REPO_ROOT MEGATRON_PATH HAS_NVLINK SOCKET_IFNAME

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_DIR}"
  --ref-load "${INIT_CKPT}"
  --load "${LOAD_DIR}"
  --save "${SAVE_DIR}"
  --save-interval 10
)

ROLLOUT_ARGS=(
  --prompt-data "${DATA_PATH}"
  --input-key messages
  --label-key ground_truth
  --apply-chat-template
  --rollout-shuffle
  --apply-chat-template-kwargs '{"enable_thinking": false}'
  --num-rollout 100
  --rollout-batch-size 64
  --n-samples-per-prompt 8
  --num-steps-per-rollout 1
  --rollout-max-prompt-len 2048
  --rollout-max-response-len 2048
  --rollout-max-context-len 4096
  --rollout-temperature 1.0
  --rollout-top-p 1.0
  --balance-data
  --save-rollout-text-data "${SAVE_DIR}/rollout-text/rollout_{rollout_id}.txt"
)

REWARD_ARGS=(
  --custom-rm-path "${IF_REWARD_PATH}"
  --custom-rollout-log-function-path slime_plugins.if_eval.metrics.log_train_batch_accuracy
)

GRPO_ARGS=(
  --advantage-estimator grpo
  # Retain slime's default group standard-deviation normalization.
  --kl-coef 0.0
  --use-kl-loss
  --kl-loss-type low_var_kl
  --kl-loss-coef 0.01
  --entropy-coef 0.001
  --eps-clip 0.2
  --eps-clip-high 0.2
  --eps-clip-c 3.0
)

WEIGHT_SYNC_ARGS=()
case "${WEIGHT_SYNC_TRANSPORT}" in
  nccl) ;;
  disk)
    UPDATE_WEIGHT_DISK_DIR=${UPDATE_WEIGHT_DISK_DIR:-${SAVE_DIR}/weight-sync}
    if [[ "${GRPO_DRY_RUN}" != true ]]; then mkdir -p "${UPDATE_WEIGHT_DISK_DIR}"; fi
    WEIGHT_SYNC_ARGS=(
      --update-weight-mode full
      --update-weight-transport disk
      --update-weight-disk-dir "${UPDATE_WEIGHT_DISK_DIR}"
    )
    ;;
  *) echo "Unknown WEIGHT_SYNC_TRANSPORT=${WEIGHT_SYNC_TRANSPORT}; choose nccl or disk." >&2; exit 1 ;;
esac

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.01
  --adam-beta1 0.9
  --adam-beta2 0.999
  --override-opt-param-scheduler
)

SWANLAB_ARGS=(
  --use-swanlab
  --swanlab-mode "${SWANLAB_MODE:-online}"
  --swanlab-dir "${SWANLAB_DIR}"
  --swanlab-project "GRPO-${TASK_NAME}-${MODEL_NAME}"
  --swanlab-group "${TASK_NAME}-${MODEL_NAME}-nonthinking"
)
if [[ -n "${SWANLAB_KEY:-}" ]]; then SWANLAB_ARGS+=(--swanlab-key "${SWANLAB_KEY}"); fi
if [[ -n "${SWANLAB_WORKSPACE:-}" ]]; then SWANLAB_ARGS+=(--swanlab-workspace "${SWANLAB_WORKSPACE}"); fi

PERF_ARGS=(
  --log-grpo-step-timing
  --grpo-step-timing-log-file "${STEP_TIMING_LOG_FILE:-${LOG_ROOT}/${TASK_NAME}/${MODEL_NAME}/step-timing.log}"
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --context-parallel-size 2
  --sequence-parallel
  --use-distributed-optimizer
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 4096
  --per-turn-loss
  --log-probs-chunk-size 1024
)

SGLANG_ARGS=(
  --sglang-config "${SGLANG_CONFIG_PATH}"
  --rollout-num-gpus-per-engine 1
  # Leave one token of headroom above 2048 prompt + 2048 response tokens.
  --sglang-context-length 4097
  --sglang-server-concurrency 64
  --sglang-mem-fraction-static 0.5
  --sglang-mamba-scheduler-strategy extra_buffer
)

MISC_ARGS=(
  --offload-train
  --offload-rollout
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --attention-backend flash
)

DEFAULT_TRAIN_ARGS=(
  --actor-num-nodes 1
  --actor-num-gpus-per-node "${NUM_GPUS}"
  --rollout-num-gpus "${NUM_GPUS}"
  --num-gpus-per-node "${NUM_GPUS}"
  --colocate
  "${MODEL_ARGS[@]}"
  "${CKPT_ARGS[@]}"
  "${ROLLOUT_ARGS[@]}"
  "${REWARD_ARGS[@]}"
  "${GRPO_ARGS[@]}"
  "${WEIGHT_SYNC_ARGS[@]}"
  "${OPTIMIZER_ARGS[@]}"
  "${SWANLAB_ARGS[@]}"
  "${PERF_ARGS[@]}"
  "${SGLANG_ARGS[@]}"
  "${MISC_ARGS[@]}"
)
grpo_merge_train_args "${DEFAULT_TRAIN_ARGS[@]}"
if [[ "${GRPO_DRY_RUN}" == true ]]; then
  printf 'IF reward: %s\n' "${GRPO_CLI_LAST_VALUE[--custom-rm-path]:-${IF_REWARD_PATH:-not configured}}"
  grpo_print_dry_run
  exit 0
fi

"${RAY_BIN}" stop --force >/dev/null 2>&1 || true
"${RAY_BIN}" start \
  --head \
  --node-ip-address "${MASTER_ADDR}" \
  --port "${MASTER_PORT}" \
  --num-gpus "${NUM_GPUS}" \
  --temp-dir "${RAY_TMP_DIR}" \
  --disable-usage-stats \
  --dashboard-host 0.0.0.0 \
  --dashboard-port "${RAY_DASHBOARD_PORT}"

archive_ray_logs() {
  local exit_status=$?
  trap - EXIT
  local session_dir
  session_dir=$(readlink -f -- "${RAY_TMP_DIR}/session_latest") || session_dir=
  if [[ -n "${session_dir}" && -d "${session_dir}/logs" ]]; then
    mkdir -p "${RAY_LOG_ARCHIVE_DIR}/$(basename -- "${session_dir}")"
    cp -a -- "${session_dir}/logs" "${RAY_LOG_ARCHIVE_DIR}/$(basename -- "${session_dir}")/" || true
  fi
  "${RAY_BIN}" stop --force >/dev/null 2>&1 || true
  if [[ "${RAY_TMP_DIR}" == "${DEFAULT_RAY_TMP_DIR}" ]]; then rm -rf -- "${RAY_TMP_DIR}"; fi
  if [[ "${WORKER_TMP_DIR}" == "${DEFAULT_WORKER_TMP_DIR}" ]]; then rm -rf -- "${WORKER_TMP_DIR}"; fi
  exit "${exit_status}"
}
trap archive_ray_logs EXIT

RUNTIME_ENV_JSON=$("${PYTHON_BIN}" - <<'PY'
import json
import os

keys = ("TMPDIR", "TMP", "TEMP", "MASTER_ADDR", "MASTER_PORT", "NLTK_DATA",
        "SGLANG_HEALTH_CHECK_TIMEOUT", "SGLANG_WARMUP_TIMEOUT", "no_proxy", "NO_PROXY")
values = {key: os.environ[key] for key in keys}
values["PYTHONPATH"] = os.pathsep.join((os.environ["REPO_ROOT"],
                                        os.environ["MEGATRON_PATH"],
                                        os.environ.get("PYTHONPATH", "")))
values.update(CUDA_DEVICE_MAX_CONNECTIONS="1", NCCL_NVLS_ENABLE=os.environ["HAS_NVLINK"],
              GLOO_SOCKET_IFNAME=os.environ["SOCKET_IFNAME"],
              TP_SOCKET_IFNAME=os.environ["SOCKET_IFNAME"],
              RAY_DISABLE_DASHBOARD_GPU_METRICS="1", SLIME_RELOAD_PROCESS_GROUPS="0")
print(json.dumps({"env_vars": values}))
PY
)

"${RAY_BIN}" job submit \
  --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${PYTHON_BIN}" train.py "${GRPO_TRAIN_ARGS[@]}"
