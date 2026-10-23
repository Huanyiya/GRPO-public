#!/usr/bin/env bash

# Qwen3.5 GRPO, single node / 8 GPUs, colocated train + rollout.
#
# Batch mapping:
#   128 prompts/rollout * 8 responses/prompt = 1,024 samples
#   1,024 samples / 1 optimizer step         = global batch size 1,024
#   187 rollouts                             = 187 optimizer steps
#
# Polaris uses the `problem` and `answer` fields. The custom reward accepts
# scalar answers and checks final boxed answers without requiring </think>.

set -euo pipefail

# torch_memory_saver is incompatible with expandable-segments allocation.
unset PYTORCH_CUDA_ALLOC_CONF

MODEL_SIZE=${MODEL_SIZE:-2B}
TASK_NAME=${TASK_NAME:-code}
source "$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)/common.sh"
# Resolve relative user paths the same way train.py will, from the repo root.
cd "${REPO_ROOT}"
grpo_parse_cli "$@"

# Paths can be overridden from the environment.
MODEL_DIR=${GRPO_CLI_LAST_VALUE[--hf-checkpoint]:-${MODEL_DIR:-${MODEL_PATH}}}
INIT_CKPT=${GRPO_CLI_LAST_VALUE[--ref-load]:-${INIT_CKPT:-${MODEL_ROOT}/${MODEL_NAME}_torch_dist}}
BASE_SAVE_DIR=${SAVE_DIR:-${CHECKPOINT_ROOT}/${TASK_NAME}/${MODEL_NAME}-GRPO}
LOAD_DIR=${GRPO_CLI_LAST_VALUE[--load]:-${LOAD_DIR:-${BASE_SAVE_DIR}}}
SAVE_DIR=${GRPO_CLI_LAST_VALUE[--save]:-${BASE_SAVE_DIR}}
# Set SAVE_OUTPUTS=true to save one human-readable JSONL file per training
# rollout. Each line records the generated code, every testcase actually run,
# its SandboxFusion result, and the final binary reward.
SAVE_OUTPUTS=${SAVE_OUTPUTS:-true}

# Select the training data with TRAIN_DATASET=acecode (default) or eurus.
# DATA_PATH remains available as an explicit one-off override.
TRAIN_DATASET=${TRAIN_DATASET:-acecode}
ACECODE_DATA_PATH=${ACECODE_DATA_PATH:-${DATA_ROOT}/${TASK_NAME}/acecode_hard.jsonl}
EURUS_DATA_PATH=${EURUS_DATA_PATH:-${DATA_ROOT}/${TASK_NAME}/eurus-2-code-verl/data/train-00000.parquet}

case "${TRAIN_DATASET}" in
  acecode)
    DEFAULT_DATA_PATH="${ACECODE_DATA_PATH}"
    DEFAULT_REWARD_FUNC="slime_plugins.rewards.acecode.reward_func"
    # The first-stage AceCode preprocessing writes the original/signature
    # variants to `question`; `context_messages` retains only the original.
    DEFAULT_TRAIN_INPUT_KEY="question"
    DEFAULT_TRAIN_LABEL_KEY="test_cases"
    ;;
  eurus)
    DEFAULT_DATA_PATH="${EURUS_DATA_PATH}"
    DEFAULT_REWARD_FUNC="slime_plugins.rewards.code.reward_func"
    DEFAULT_TRAIN_INPUT_KEY="prompt"
    DEFAULT_TRAIN_LABEL_KEY="reward_model"
    ;;
  *)
    echo "TRAIN_DATASET must be acecode or eurus; got ${TRAIN_DATASET}." >&2
    exit 1
    ;;
esac

DATA_PATH=${GRPO_CLI_LAST_VALUE[--prompt-data]:-${DATA_PATH:-${DEFAULT_DATA_PATH}}}
REWARD_FUNC=${REWARD_FUNC:-${DEFAULT_REWARD_FUNC}}
# Kept for the shared dry-run printer; evaluation is disabled by default.
EVAL_DATA_PATH=${EVAL_DATA_PATH:-}
TRAIN_INPUT_KEY=${TRAIN_INPUT_KEY:-${DEFAULT_TRAIN_INPUT_KEY}}
TRAIN_LABEL_KEY=${TRAIN_LABEL_KEY:-${DEFAULT_TRAIN_LABEL_KEY}}
echo "Training dataset: ${TRAIN_DATASET} (${DATA_PATH}; input=${TRAIN_INPUT_KEY}; label=${TRAIN_LABEL_KEY}; reward=${REWARD_FUNC})"

SGLANG_MEM_FRACTION=${SGLANG_MEM_FRACTION:-0.7}
ROLLOUT_BATCH_SIZE=${ROLLOUT_BATCH_SIZE:-128}
LR=${LR:-1e-6}
# On this PPU, rebuilding PCCL groups and then exporting CUDA IPC tensors can
# crash inside HGGC.  Use the OPD-proven path by default: export HF weights
# while Megatron is resident, CPU-offload it, then reload SGLang from disk.
UPDATE_WEIGHT_DISK_DIR=${UPDATE_WEIGHT_DISK_DIR:-${SAVE_DIR}/weight-sync}


WANDB_DIR=${GRPO_CLI_LAST_VALUE[--wandb-dir]:-${WANDB_DIR:-${LOG_ROOT}/${TASK_NAME}/${MODEL_NAME}/wandb}}
# Leave this empty to use an existing `wandb login` session, or provide the
# key at launch with WANDB_KEY=... .  Do not commit a personal API key here.
WANDB_KEY=${WANDB_KEY:-}

# Code rewards are executed by the remote SandboxFusion service.  Do not fall
# back to executing untrusted model output on this training node.
SANDBOX_FUSION_URL=${SANDBOX_FUSION_URL:-}
CODE_SANDBOX_MAX_CONCURRENT=${CODE_SANDBOX_MAX_CONCURRENT:-64}
CODE_COMPILE_TIMEOUT=${CODE_COMPILE_TIMEOUT:-10}
CODE_RUN_TIMEOUT=${CODE_RUN_TIMEOUT:-10}
CODE_MEMORY_LIMIT_MB=${CODE_MEMORY_LIMIT_MB:-1024}

if [[ "${SAVE_OUTPUTS}" != "true" && "${SAVE_OUTPUTS}" != "false" ]]; then
  echo "SAVE_OUTPUTS must be exactly true or false; got ${SAVE_OUTPUTS}." >&2
  exit 1
fi

if [[ "${GRPO_DRY_RUN}" != true && -z "${SANDBOX_FUSION_URL}" ]]; then
  echo "SANDBOX_FUSION_URL must be set to the SandboxFusion execution endpoint." >&2
  exit 1
fi

if [[ "${GRPO_DRY_RUN}" != true ]]; then
  for required_path in "${MODEL_DIR}" "${INIT_CKPT}" "${DATA_PATH}" "${PYTHON_BIN}" "${RAY_BIN}" "${MEGATRON_PATH}"; do
    if [[ ! -e "${required_path}" ]]; then
      echo "Required path does not exist: ${required_path}" >&2
      exit 1
    fi
  done

  if [[ ! -f "${INIT_CKPT}/latest_checkpointed_iteration.txt" ]]; then
    echo "INIT_CKPT is not a Megatron torch_dist checkpoint: ${INIT_CKPT}" >&2
    echo "Generate it first, or override INIT_CKPT=/path/to/your_torch_dist_checkpoint." >&2
    exit 1
  fi
fi

if [[ "${NUM_GPUS}" -ne 8 ]]; then
  echo "This script is configured for exactly 8 GPUs; got NUM_GPUS=${NUM_GPUS}." >&2
  exit 1
fi

MODEL_CONFIG="${SCRIPT_DIR}/models/qwen3.5-${MODEL_SIZE}.sh"
if [[ ! -f "${MODEL_CONFIG}" ]]; then
  echo "Model configuration not found: ${MODEL_CONFIG}" >&2
  exit 1
fi
source "${MODEL_CONFIG}"

export PYTHONUNBUFFERED=1
export MASTER_ADDR MASTER_PORT
export SGLANG_HEALTH_CHECK_TIMEOUT SGLANG_WARMUP_TIMEOUT
export SANDBOX_FUSION_URL
export CODE_SANDBOX_MAX_CONCURRENT CODE_COMPILE_TIMEOUT CODE_RUN_TIMEOUT CODE_MEMORY_LIMIT_MB
export SAVE_OUTPUTS
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
  --save-interval 10
)

ROLLOUT_ARGS=(
  --prompt-data "${DATA_PATH}"
  --input-key "${TRAIN_INPUT_KEY}"
  --label-key "${TRAIN_LABEL_KEY}"
  --apply-chat-template
  --apply-chat-template-kwargs '{"enable_thinking": false}'
  --rollout-shuffle

  --num-rollout 187
  --rollout-batch-size "${ROLLOUT_BATCH_SIZE}"
  --n-samples-per-prompt 8
  --num-steps-per-rollout 1

  --rollout-max-prompt-len 2048
  --rollout-max-response-len 8192
  --rollout-max-context-len 10240
  --rollout-temperature 1.0
  --rollout-top-p 1.0
  --balance-data
)

if [[ "${SAVE_OUTPUTS}" == "true" ]]; then
  ROLLOUT_ARGS+=(
    --save-rollout-outputs "${SAVE_DIR}/rollout-outputs/{rollout_id}.jsonl"
  )
fi

REWARD_ARGS=(
  --custom-rm-path "${REWARD_FUNC}"
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
  --lr "${LR}"
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
  --override-opt-param-scheduler
)

WANDB_ARGS=(
  --use-wandb
  --wandb-mode online
  --wandb-project "GRPO-${TASK_NAME}-${MODEL_NAME}"
  --wandb-group "${TRAIN_DATASET}-${TASK_NAME}-${MODEL_NAME}-nonthinking"
  --wandb-dir "${WANDB_DIR}"
  --log-passrate
)

if [[ -n "${WANDB_KEY}" ]]; then
  WANDB_ARGS+=(--wandb-key "${WANDB_KEY}")
fi

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
  --max-tokens-per-gpu 6144
  --calculate-per-token-loss
  --log-probs-chunk-size 1024
  # 加到 PERF_ARGS
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-context-length 18432
  --sglang-server-concurrency 64
  --sglang-mem-fraction-static "${SGLANG_MEM_FRACTION}"
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

# Build one final argv, then exit before starting services for --dry-run.
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

# Stop only Ray processes managed by Ray before starting a new cluster.
"${RAY_BIN}" stop --force >/dev/null 2>&1 || true

# Verify the remote code executor before starting a new Ray cluster.
echo "Checking SandboxFusion endpoint..."
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" "${PYTHON_BIN}" - <<'PY'
import sys

from slime_plugins.rewards.code_sandbox import run_code

expected_output = "slime_sandboxfusion_preflight"
result = run_code(f"print({expected_output!r})", "")
if not result.success or result.stdout.strip() != expected_output:
    print(
        "SandboxFusion preflight failed: "
        f"status={result.status}, return_code={result.return_code}, stderr={result.stderr!r}, "
        f"stdout={result.stdout!r}",
        file=sys.stderr,
    )
    raise SystemExit(1)
PY
echo "SandboxFusion preflight passed."

"${RAY_BIN}" start \
  --head \
  --node-ip-address "${MASTER_ADDR}" \
  --port "${MASTER_PORT}" \
  --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats \
  --dashboard-host 0.0.0.0 \
  --dashboard-port "${RAY_DASHBOARD_PORT}"

RUNTIME_ENV_JSON=$(printf \
  '{"env_vars":{"PYTHONPATH":"%s:%s","CUDA_DEVICE_MAX_CONNECTIONS":"1","NCCL_NVLS_ENABLE":"%s","GLOO_SOCKET_IFNAME":"%s","TP_SOCKET_IFNAME":"%s","MASTER_ADDR":"%s","MASTER_PORT":"%s","RAY_DISABLE_DASHBOARD_GPU_METRICS":"1","SLIME_RELOAD_PROCESS_GROUPS":"0","SGLANG_HEALTH_CHECK_TIMEOUT":"%s","SGLANG_WARMUP_TIMEOUT":"%s","no_proxy":"%s","NO_PROXY":"%s","SANDBOX_FUSION_URL":"%s","CODE_SANDBOX_MAX_CONCURRENT":"%s","CODE_COMPILE_TIMEOUT":"%s","CODE_RUN_TIMEOUT":"%s","CODE_MEMORY_LIMIT_MB":"%s","SAVE_OUTPUTS":"%s"}}' \
  "${REPO_ROOT}" "${MEGATRON_PATH}" "${HAS_NVLINK}" "${SOCKET_IFNAME}" "${SOCKET_IFNAME}" \
  "${MASTER_ADDR}" "${MASTER_PORT}" "${SGLANG_HEALTH_CHECK_TIMEOUT}" "${SGLANG_WARMUP_TIMEOUT}" "${no_proxy}" "${NO_PROXY}" \
  "${SANDBOX_FUSION_URL}" "${CODE_SANDBOX_MAX_CONCURRENT}" \
  "${CODE_COMPILE_TIMEOUT}" "${CODE_RUN_TIMEOUT}" "${CODE_MEMORY_LIMIT_MB}" "${SAVE_OUTPUTS}")

"${RAY_BIN}" job submit \
  --address="http://127.0.0.1:${RAY_DASHBOARD_PORT}" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- "${PYTHON_BIN}" train.py "${GRPO_TRAIN_ARGS[@]}"
