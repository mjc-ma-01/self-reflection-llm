#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HUGGINGFACE_HUB_CACHE}}"
export WANDB_DISABLED="${WANDB_DISABLED:-true}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

python -m training.rl.run_gdpo \
  --model_path "${MODEL_PATH:-checkpoints/sft/merged}" \
  --train_file "${RL_TRAIN_FILE:-outputs/processed/rl/train.parquet}" \
  --val_file "${RL_VAL_FILE:-outputs/processed/rl/test.parquet}" \
  --output_dir "${RL_OUTPUT_CHECKPOINT_DIR:-checkpoints/rl}" \
  "$@"
