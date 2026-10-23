#!/usr/bin/env bash

# Qwen3.5 GRPO, single node / 4 GPUs, colocated train + rollout.
#
# Batch mapping:
#   128 prompts/rollout * 8 responses/prompt = 1,024 samples
#   1,024 samples / 1 optimizer step         = global batch size 1,024
#   140 rollouts                             = 140 optimizer steps
#
# DAPO uses `prompt` and `reward_model`; both training and evaluation prompts
# require a final answer in \boxed{...}.

set -euo pipefail

# torch_memory_saver is incompatible with expandable-segments allocation.
unset PYTORCH_CUDA_ALLOC_CONF

MODEL_SIZE=${MODEL_SIZE:-2B}
TASK_NAME=${TASK_NAME:-math}
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
# Resolve relative user paths the same way train.py will, from the repo root.
cd "${REPO_ROOT}"
grpo_parse_cli "$@"

# Paths can be overridden from the environment.
MODEL_DIR=${GRPO_CLI_LAST_VALUE[--hf-checkpoint]:-${MODEL_DIR:-${MODEL_PATH}}}
INIT_CKPT=${GRPO_CLI_LAST_VALUE[--ref-load]:-${INIT_CKPT:-${MODEL_ROOT}/${MODEL_NAME}_torch_dist}}
# Generate the boxed evaluation file with scripts/data_preprocess/prepare_math_eval_boxed.py.
EVAL_DATA_PATH=${EVAL_DATA_PATH:-${DATA_ROOT}/${TASK_NAME}/aime25-boxed.parquet}
BASE_SAVE_DIR=${SAVE_DIR:-${CHECKPOINT_ROOT}/${TASK_NAME}/${MODEL_NAME}-GRPO}
LOAD_DIR=${GRPO_CLI_LAST_VALUE[--load]:-${LOAD_DIR:-${BASE_SAVE_DIR}}}
SAVE_DIR=${GRPO_CLI_LAST_VALUE[--save]:-${BASE_SAVE_DIR}}

# DATA_PATH or --prompt-data can select another compatible boxed-answer dataset.
DAPO_DATA_PATH=${DAPO_DATA_PATH:-${DATA_ROOT}/${TASK_NAME}/dapo-math-17k-processed.parquet}
DATA_PATH=${GRPO_CLI_LAST_VALUE[--prompt-data]:-${DATA_PATH:-${DAPO_DATA_PATH}}}
echo "Training dataset: ${DATA_PATH} (input=prompt; label=reward_model)"

# On this PPU, rebuilding PCCL groups and then exporting CUDA IPC tensors can
# crash inside HGGC.  Use the OPD-proven path by default: export HF weights
# while Megatron is resident, CPU-offload it, then reload SGLang from disk.
UPDATE_WEIGHT_DISK_DIR=${UPDATE_WEIGHT_DISK_DIR:-${SAVE_DIR}/weight-sync}
WANDB_DIR=${GRPO_CLI_LAST_VALUE[--wandb-dir]:-${WANDB_DIR:-${LOG_ROOT}/${TASK_NAME}/${MODEL_NAME}/wandb}}

MODEL_CONFIG="${SCRIPT_DIR}/models/qwen3.5-${MODEL_SIZE}.sh"
if [[ ! -f "${MODEL_CONFIG}" ]]; then
  echo "Model configuration not found: ${MODEL_CONFIG}" >&2
  exit 1
fi
source "${MODEL_CONFIG}"

export PYTHONUNBUFFERED=1
export MASTER_ADDR MASTER_PORT
export SGLANG_HEALTH_CHECK_TIMEOUT SGLANG_WARMUP_TIMEOUT
# HGGC's NVML does not implement nvmlDeviceGetProcessesUtilizationInfo. Ray
# 2.55's dashboard agent otherwise exits during GPU metric collection, which
# makes raylet exit and causes `ray start` to time out.
export RAY_DISABLE_DASHBOARD_GPU_METRICS=1
# tmux servers keep the environment from the time they were created.  In
# particular, SGLang Model Gateway (Rust) also honors ALL_PROXY, not only the
# usual HTTP(S)_PROXY variables.  A proxy must never sit between the gateway
# and its local 172.18.x.x worker ports.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY ftp_proxy FTP_PROXY NO_PROXY RAY_ADDRESS
export no_proxy="localhost,127.0.0.1,0.0.0.0,${MASTER_ADDR}"
export NO_PROXY="${no_proxy}"

if [[ "${GRPO_DRY_RUN}" != true ]]; then
  mkdir -p "${WANDB_DIR}"
fi

NVLINK_COUNT=0
if [[ "${GRPO_DRY_RUN}" != true ]] && command -v nvidia-smi >/dev/null 2>&1; then
  NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l || true)
fi
if [[ "${NVLINK_COUNT}" -gt 0 ]]; then
  HAS_NVLINK=1
else
  HAS_NVLINK=0
fi
if [[ "${GRPO_DRY_RUN}" != true ]]; then
  echo "HAS_NVLINK=${HAS_NVLINK} (${NVLINK_COUNT} NVLink references detected)"
fi

CKPT_ARGS=(
  --hf-checkpoint "${MODEL_DIR}"
  # First run falls back to INIT_CKPT; later runs resume from SAVE_DIR.
  --ref-load "${INIT_CKPT}"
  --load "${LOAD_DIR}"
  --save "${SAVE_DIR}"
  --save-interval 100
)

ROLLOUT_ARGS=(
  --prompt-data "${DATA_PATH}"
  --input-key prompt
  --label-key reward_model
  --apply-chat-template
  --apply-chat-template-kwargs '{"enable_thinking": false}'
  --rollout-shuffle

  --num-rollout 100
  --rollout-batch-size 128
  --n-samples-per-prompt 8
  --num-steps-per-rollout 1

  --rollout-max-prompt-len 2048
  --rollout-max-response-len 16384
  --rollout-max-context-len 18432
  --rollout-temperature 1.0
  --rollout-top-p 1.0
  --balance-data
)

EVAL_ARGS=(
  # One rollout equals one optimizer step; this evaluates every 50 steps.
  --eval-interval 50
  --eval-prompt-data AIME25 "${EVAL_DATA_PATH}"
  --skip-eval-before-train
  --eval-input-key prompt
  --eval-label-key reward_model
  --n-samples-per-eval-prompt 8
  --eval-max-prompt-len 2048
  --eval-max-response-len 16384
  --eval-max-context-len 18432
  --eval-temperature 1.0
  --eval-top-p 0.95
  --eval-top-k 20
  --eval-presence-penalty 1.5
  --eval-repetition-penalty 1.0
)

REWARD_ARGS=(
  --custom-rm-path slime_plugins.rewards.dapo_math.reward_func
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
  nccl)
    ;;
  disk)
    if [[ "${GRPO_DRY_RUN}" != true ]]; then
      mkdir -p "${UPDATE_WEIGHT_DISK_DIR}"
    fi
    WEIGHT_SYNC_ARGS=(
      --update-weight-mode full
      --update-weight-transport disk
      --update-weight-disk-dir "${UPDATE_WEIGHT_DISK_DIR}"
    )
    ;;
  *)
    echo "Unknown WEIGHT_SYNC_TRANSPORT=${WEIGHT_SYNC_TRANSPORT}; choose nccl or disk." >&2
    exit 1
    ;;
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
  --wandb-group "dapo-${TASK_NAME}-${MODEL_NAME}-nonthinking"
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

  # Dynamic token batching replaces an infeasible physical micro-batch of
  # 128 sequences. With CP=2, a full 18,432-token sample spans two GPUs.
  --use-dynamic-batch-size
  --max-tokens-per-gpu 8192
  --calculate-per-token-loss
  --log-probs-chunk-size 1024
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-context-length 18432
  --sglang-server-concurrency 64
  --sglang-mem-fraction-static 0.7
  --sglang-mamba-scheduler-strategy extra_buffer
)

MISC_ARGS=(
  # Colocated mode normally enables both flags; keep them explicit because the
  # split disk-sync lifecycle depends on both sides being offloadable.
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
  "${EVAL_ARGS[@]}"
  "${REWARD_ARGS[@]}"
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
  grpo_print_dry_run
  exit 0
fi

# Stop only Ray processes managed by Ray; do not kill unrelated Python jobs.
"${RAY_BIN}" stop --force >/dev/null 2>&1 || true
"${RAY_BIN}" start \
  --head \
  --node-ip-address "${MASTER_ADDR}" \
  --port "${MASTER_PORT}" \
  --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats \
  --dashboard-host 0.0.0.0 \
  --dashboard-port "${RAY_DASHBOARD_PORT}"

RUNTIME_ENV_JSON=$(printf \
  '{"env_vars":{"PYTHONPATH":"%s:%s","CUDA_DEVICE_MAX_CONNECTIONS":"1","NCCL_NVLS_ENABLE":"%s","GLOO_SOCKET_IFNAME":"%s","TP_SOCKET_IFNAME":"%s","MASTER_ADDR":"%s","MASTER_PORT":"%s","RAY_DISABLE_DASHBOARD_GPU_METRICS":"1","SLIME_RELOAD_PROCESS_GROUPS":"0","SGLANG_HEALTH_CHECK_TIMEOUT":"%s","SGLANG_WARMUP_TIMEOUT":"%s","no_proxy":"%s","NO_PROXY":"%s"}}' \
  "${REPO_ROOT}" "${MEGATRON_PATH}" "${HAS_NVLINK}" "${SOCKET_IFNAME}" "${SOCKET_IFNAME}" \
  "${MASTER_ADDR}" "${MASTER_PORT}" "${SGLANG_HEALTH_CHECK_TIMEOUT}" "${SGLANG_WARMUP_TIMEOUT}" "${no_proxy}" "${NO_PROXY}")

"${RAY_BIN}" job submit \
  --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${PYTHON_BIN}" train.py "${GRPO_TRAIN_ARGS[@]}"
