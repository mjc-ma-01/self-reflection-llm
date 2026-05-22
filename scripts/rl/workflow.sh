#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HUGGINGFACE_HUB_CACHE}}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

RUN_CHECK=1
RUN_PREPARE=1
RUN_TRAIN=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only)
      RUN_PREPARE=0
      RUN_TRAIN=0
      shift
      ;;
    --prepare-only)
      RUN_TRAIN=0
      shift
      ;;
    --train)
      RUN_TRAIN=1
      shift
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "${RUN_CHECK}" == "1" ]]; then
  python -m training.rl.check_env
fi

if [[ "${RUN_PREPARE}" == "1" ]]; then
  python -m training.rl.prepare \
    --rl_dir "${RL_DIR:-datasets/rl}" \
    --output_dir "${RL_OUTPUT_DIR:-outputs/processed/rl}" \
    --test_size "${TEST_SIZE:-0.02}" \
    --seed "${SEED:-42}" \
    --limit_per_pattern "${LIMIT_PER_PATTERN:--1}"
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  python -m training.rl.check_env --strict
  python -m training.rl.run_gdpo \
    --model_path "${MODEL_PATH:-checkpoints/sft/merged}" \
    --train_file "${RL_TRAIN_FILE:-outputs/processed/rl/train.parquet}" \
    --val_file "${RL_VAL_FILE:-outputs/processed/rl/test.parquet}" \
    --output_dir "${RL_OUTPUT_CHECKPOINT_DIR:-checkpoints/rl}" \
    "${EXTRA_ARGS[@]:-}"
fi
