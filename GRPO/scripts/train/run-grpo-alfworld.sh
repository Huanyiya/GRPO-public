#!/usr/bin/env bash

# ALFWorld GRPO through slime's SGLang rollout and Megatron trainer.
# The environment supplies tasks, so a prompt dataset is not required.

set -euo pipefail
unset PYTORCH_CUDA_ALLOC_CONF
export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4

MODEL_SIZE=${MODEL_SIZE:-2B}
TASK_NAME=alfworld
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
cd "${REPO_ROOT}"
grpo_parse_cli "$@"

# SDA's TextWorld config resolves games and logic files through ALFWORLD_DATA.
# Keep the same data-root convention while allowing an explicit override.
ALFWORLD_DATA=$(realpath -m -- "${ALFWORLD_DATA:-${DATA_ROOT}/alfworld}")
export ALFWORLD_DATA

MODEL_DIR=${GRPO_CLI_LAST_VALUE[--hf-checkpoint]:-${MODEL_DIR:-${MODEL_PATH}}}
INIT_CKPT=${GRPO_CLI_LAST_VALUE[--ref-load]:-${INIT_CKPT:-${MODEL_ROOT}/${MODEL_NAME}_torch_dist}}
BASE_SAVE_DIR=${SAVE_DIR:-${CHECKPOINT_ROOT}/${TASK_NAME}/${MODEL_NAME}-GRPO}
LOAD_DIR=${GRPO_CLI_LAST_VALUE[--load]:-${LOAD_DIR:-${BASE_SAVE_DIR}}}
SAVE_DIR=${GRPO_CLI_LAST_VALUE[--save]:-${BASE_SAVE_DIR}}
RESUME_ITERATION=19
WANDB_RUN_ID=${WANDB_RUN_ID:-v938e9bt}
WANDB_DIR=${GRPO_CLI_LAST_VALUE[--wandb-dir]:-${WANDB_DIR:-${LOG_ROOT}/${TASK_NAME}/${MODEL_NAME}/wandb}}
UPDATE_WEIGHT_DISK_DIR=${UPDATE_WEIGHT_DISK_DIR:-${SAVE_DIR}/weight-sync}
# Ray and SGLang create Unix sockets below their temporary directories.
# Keep these short and on a local filesystem, as in the OPD launch scripts.
DEFAULT_RAY_TMP_DIR=/dev/shm/ray-alfworld-$$
RAY_TMP_DIR=${RAY_TMP_DIR:-${DEFAULT_RAY_TMP_DIR}}
RAY_LOG_ARCHIVE_DIR=${RAY_LOG_ARCHIVE_DIR:-${LOG_ROOT}/ray}
DEFAULT_WORKER_TMP_DIR=/dev/shm/alfworld-tmp-$$
WORKER_TMP_DIR=${WORKER_TMP_DIR:-${DEFAULT_WORKER_TMP_DIR}}
SGLANG_CONFIG_PATH=${WORKER_TMP_DIR}/sglang-text-only.yaml
DATA_PATH=
EVAL_DATA_PATH=

MODEL_CONFIG="${SCRIPT_DIR}/models/qwen3.5-${MODEL_SIZE}.sh"
if [[ ! -f "${MODEL_CONFIG}" ]]; then
  echo "Model configuration not found: ${MODEL_CONFIG}" >&2
  exit 1
fi
source "${MODEL_CONFIG}"

export PYTHONUNBUFFERED=1 MASTER_ADDR MASTER_PORT
export SGLANG_HEALTH_CHECK_TIMEOUT SGLANG_WARMUP_TIMEOUT
export RAY_DISABLE_DASHBOARD_GPU_METRICS=1
export RAY_TMP_DIR
export TMPDIR="${WORKER_TMP_DIR}"
export TMP="${TMPDIR}" TEMP="${TMPDIR}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ftp_proxy FTP_PROXY NO_PROXY RAY_ADDRESS
export no_proxy="localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}"
export NO_PROXY="${no_proxy}"

if [[ "${GRPO_DRY_RUN}" != true ]]; then
  if [[ "$(cat "${LOAD_DIR}/latest_checkpointed_iteration.txt" 2>/dev/null || true)" != "${RESUME_ITERATION}" || ! -f "${LOAD_DIR}/iter_0000019/.metadata" || ! -f "${LOAD_DIR}/rollout/alfworld_data_source_19.pt" ]]; then
    echo "Expected model iteration 19 and ALFWorld data state under ${LOAD_DIR}" >&2
    exit 1
  fi
  # Match the asset checks in SDA's prepare_alfworld.sh. The text environment
  # selects solvable game.tw-pddl files from the train split at runtime.
  for required_path in \
    "${ALFWORLD_DATA}/json_2.1.1/train" \
    "${ALFWORLD_DATA}/json_2.1.1/valid_seen" \
    "${ALFWORLD_DATA}/json_2.1.1/valid_unseen" \
    "${ALFWORLD_DATA}/logic/alfred.pddl" \
    "${ALFWORLD_DATA}/logic/alfred.twl2" \
    "${ALFWORLD_DATA}/detectors/mrcnn_alfred_objects_sep13_004.pth"; do
    if [[ ! -e "${required_path}" ]]; then
      echo "Required ALFWorld data path does not exist: ${required_path}" >&2
      exit 1
    fi
  done
  if [[ -z "$(find "${ALFWORLD_DATA}/json_2.1.1/train" -name game.tw-pddl -print -quit)" ]]; then
    echo "No ALFWorld TextWorld games found under ${ALFWORLD_DATA}/json_2.1.1/train" >&2
    exit 1
  fi
  for required_path in "${MODEL_DIR}" "${INIT_CKPT}" "${PYTHON_BIN}" "${RAY_BIN}" "${MEGATRON_PATH}"; do
    if [[ ! -e "${required_path}" ]]; then
      echo "Required path does not exist: ${required_path}" >&2
      exit 1
    fi
  done
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" -c 'import alfworld, textworld, yaml'
  mkdir -p "${WANDB_DIR}" "${RAY_TMP_DIR}" "${RAY_LOG_ARCHIVE_DIR}" "${WORKER_TMP_DIR}"
  # ALFWorld uses text only. Skip SGLang's multimodal processor, whose Qwen
  # module cannot import against the installed Transformers version.
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

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_DIR}"
  --ref-load "${INIT_CKPT}"
  --load "${LOAD_DIR}"
  --save "${SAVE_DIR}"
  --save-interval 10
)

ROLLOUT_ARGS=(
  --data-source-path slime_plugins.alfworld.data_source.AlfWorldDataSource
  --custom-generate-function-path slime_plugins.alfworld.rollout.generate
  --custom-reward-post-process-path slime_plugins.alfworld.rewards.normalize_step_rewards
  --apply-chat-template-kwargs '{"enable_thinking": false}'
  --num-rollout 100
  --rollout-batch-size 16
  --n-samples-per-prompt 8
  --num-steps-per-rollout 1
  --rollout-max-prompt-len 2048
  --rollout-max-response-len 512
  --rollout-max-context-len 2560
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
)

WEIGHT_SYNC_ARGS=()
case "${WEIGHT_SYNC_TRANSPORT}" in
  nccl) ;;
  disk)
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
  --wandb-run-id "${WANDB_RUN_ID}"
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
  --calculate-per-token-loss
  --log-probs-chunk-size 1024
)

SGLANG_ARGS=(
  --sglang-config "${SGLANG_CONFIG_PATH}"
  --rollout-num-gpus-per-engine 1
  --sglang-context-length 2560
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
  printf 'ALFWORLD_DATA=%s\n' "${ALFWORLD_DATA}"
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

# Archive this run's Ray logs before removing its transient runtime files.
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
  if [[ "${RAY_TMP_DIR}" == "${DEFAULT_RAY_TMP_DIR}" ]]; then
    rm -rf -- "${RAY_TMP_DIR}"
  fi
  if [[ "${WORKER_TMP_DIR}" == "${DEFAULT_WORKER_TMP_DIR}" ]]; then
    rm -rf -- "${WORKER_TMP_DIR}"
  fi
  exit "${exit_status}"
}
trap archive_ray_logs EXIT

RUNTIME_ENV_JSON=$(printf \
  '{"env_vars":{"PYTHONPATH":"%s:%s","ALFWORLD_DATA":"%s","TMPDIR":"%s","TMP":"%s","TEMP":"%s","CUDA_DEVICE_MAX_CONNECTIONS":"1","NCCL_NVLS_ENABLE":"%s","GLOO_SOCKET_IFNAME":"%s","TP_SOCKET_IFNAME":"%s","MASTER_ADDR":"%s","MASTER_PORT":"%s","RAY_DISABLE_DASHBOARD_GPU_METRICS":"1","SLIME_RELOAD_PROCESS_GROUPS":"0","SGLANG_HEALTH_CHECK_TIMEOUT":"%s","SGLANG_WARMUP_TIMEOUT":"%s","no_proxy":"%s","NO_PROXY":"%s"}}' \
  "${REPO_ROOT}" "${MEGATRON_PATH}" "${ALFWORLD_DATA}" "${WORKER_TMP_DIR}" "${WORKER_TMP_DIR}" "${WORKER_TMP_DIR}" "${HAS_NVLINK}" "${SOCKET_IFNAME}" "${SOCKET_IFNAME}" \
  "${MASTER_ADDR}" "${MASTER_PORT}" "${SGLANG_HEALTH_CHECK_TIMEOUT}" "${SGLANG_WARMUP_TIMEOUT}" "${no_proxy}" "${NO_PROXY}")

"${RAY_BIN}" job submit \
  --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${PYTHON_BIN}" train.py "${GRPO_TRAIN_ARGS[@]}"
