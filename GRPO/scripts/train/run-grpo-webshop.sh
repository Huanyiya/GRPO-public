#!/usr/bin/env bash

# slime GRPO on the small WebShop simulator, following SDA's training rollout.
# Required: WEBSHOP_ROOT (source tree with web_agent_site and search_engine),
#           WEBSHOP_DATA (items_shuffle_1000.json and items_ins_v2_1000.json).
# Example: WEBSHOP_ROOT=/path/to/webshop WEBSHOP_DATA=/path/to/data/webshop \
#          bash scripts/train/run-grpo-webshop.sh --dry-run

set -euo pipefail
unset PYTORCH_CUDA_ALLOC_CONF

export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
export PATH="$JAVA_HOME/bin:$PATH"
MODEL_SIZE=${MODEL_SIZE:-2B}
TASK_NAME=webshop
NUM_GPUS=${NUM_GPUS:-8}

source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
cd "${REPO_ROOT}"
grpo_parse_cli "$@"

GPU_IDS=${GPU_IDS:-$(seq -s, 0 "$((NUM_GPUS - 1))")}
export CUDA_VISIBLE_DEVICES="${GPU_IDS}"
WEBSHOP_ROOT=$(realpath -m -- "${WEBSHOP_ROOT:-${REPO_ROOT}/webshop}")
WEBSHOP_DATA=$(realpath -m -- "${WEBSHOP_DATA:-${DATA_ROOT}/webshop}")
export WEBSHOP_ROOT WEBSHOP_DATA

MODEL_DIR=${GRPO_CLI_LAST_VALUE[--hf-checkpoint]:-${MODEL_DIR:-${MODEL_PATH}}}
INIT_CKPT=${GRPO_CLI_LAST_VALUE[--ref-load]:-${INIT_CKPT:-${MODEL_ROOT}/${MODEL_NAME}_torch_dist}}
SAVE_DIR=${GRPO_CLI_LAST_VALUE[--save]:-${SAVE_DIR:-${CHECKPOINT_ROOT}/${TASK_NAME}/${MODEL_NAME}-GRPO}}
LOAD_DIR=${GRPO_CLI_LAST_VALUE[--load]:-${LOAD_DIR:-${SAVE_DIR}}}
DATA_PATH=
EVAL_DATA_PATH=
WANDB_DIR=${GRPO_CLI_LAST_VALUE[--wandb-dir]:-${WANDB_DIR:-${LOG_ROOT}/${TASK_NAME}/${MODEL_NAME}/wandb}}
DEFAULT_RAY_TMP_DIR=/tmp/ray-webshop-$$
RAY_TMP_DIR=${RAY_TMP_DIR:-${DEFAULT_RAY_TMP_DIR}}
DEFAULT_WORKER_TMP_DIR=/tmp/webshop-worker-$$
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
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ftp_proxy FTP_PROXY NO_PROXY RAY_ADDRESS
export no_proxy="localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}"
export NO_PROXY="${no_proxy}"

if [[ "${GRPO_DRY_RUN}" != true ]]; then
  for required_path in \
    "${WEBSHOP_ROOT}/web_agent_site/envs/web_agent_text_env.py" \
    "${WEBSHOP_ROOT}/search_engine/indexes" \
    "${WEBSHOP_DATA}/items_shuffle_1000.json" \
    "${WEBSHOP_DATA}/items_ins_v2_1000.json" \
    "${MODEL_DIR}" "${INIT_CKPT}" "${PYTHON_BIN}" "${RAY_BIN}" "${MEGATRON_PATH}"; do
    if [[ ! -e "${required_path}" ]]; then
      echo "Required WebShop or training path does not exist: ${required_path}" >&2
      exit 1
    fi
  done
  PYTHONPATH="${REPO_ROOT}:${WEBSHOP_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" -c \
    'import gym, pyserini, spacy, slime; spacy.load("en_core_web_sm")'
  mkdir -p "${WANDB_DIR}" "${RAY_TMP_DIR}" "${RAY_LOG_ARCHIVE_DIR}" "${WORKER_TMP_DIR}"
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
  --data-source-path slime_plugins.webshop.rollout.WebshopDataSource
  --custom-generate-function-path slime_plugins.webshop.rollout.generate
  --custom-reward-post-process-path slime_plugins.alfworld.rewards.normalize_step_rewards
  --apply-chat-template-kwargs '{"enable_thinking": false}'
  --num-rollout 100
  --rollout-batch-size 16
  --n-samples-per-prompt 8
  --num-steps-per-rollout 1
  --rollout-max-prompt-len 4096
  --rollout-max-response-len 512
  --rollout-max-context-len 4608
  --rollout-temperature 1.0
  --rollout-top-p 1.0
  --balance-data

)

GRPO_ARGS=(
  --advantage-estimator grpo
  --kl-coef 0.0
  --kl-loss-coef 0.0
  --entropy-coef 0.0
  --eps-clip 0.2
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
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --override-opt-param-scheduler
)

WANDB_ARGS=(
    --use-wandb
    --wandb-mode online
    --wandb-dir "${WANDB_DIR}"
    --wandb-project "GRPO-${TASK_NAME}-${MODEL_NAME}"
    --wandb-group "${TASK_NAME}-${MODEL_NAME}-nonthinking"
  )


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
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --use-dynamic-batch-size
  --max-tokens-per-gpu 8192
  --per-turn-loss
  --log-probs-chunk-size 1024
)

SGLANG_ARGS=(
  --sglang-config "${SGLANG_CONFIG_PATH}"
  --rollout-num-gpus-per-engine 1
  --sglang-context-length 4608
  --sglang-server-concurrency 64
  --sglang-mem-fraction-static 0.7
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
  "${GRPO_ARGS[@]}"
  "${WEIGHT_SYNC_ARGS[@]}"
  "${OPTIMIZER_ARGS[@]}"
  "${WANDB_ARGS[@]}"
  "${PERF_ARGS[@]}"
  "${SGLANG_ARGS[@]}"
  "${MISC_ARGS[@]}"
)
grpo_merge_train_args "${DEFAULT_TRAIN_ARGS[@]}"
if [[ "${GRPO_DRY_RUN}" == true ]]; then
  printf 'WEBSHOP_ROOT=%s\nWEBSHOP_DATA=%s\n' "${WEBSHOP_ROOT}" "${WEBSHOP_DATA}"
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

keys = ("WEBSHOP_ROOT", "WEBSHOP_DATA", "TMPDIR", "TMP", "TEMP",
        "MASTER_ADDR", "MASTER_PORT", "SGLANG_HEALTH_CHECK_TIMEOUT",
        "SGLANG_WARMUP_TIMEOUT", "no_proxy", "NO_PROXY")
values = {key: os.environ[key] for key in keys}
values["PYTHONPATH"] = os.pathsep.join((os.environ["REPO_ROOT"],
                                        os.environ["WEBSHOP_ROOT"],
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
