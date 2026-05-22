#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HUGGINGFACE_HUB_CACHE}}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED="${WANDB_DISABLED:-true}"

MODEL_PATH="${MODEL_PATH:-meta-llama/Meta-Llama-3-8B-Instruct}"
TRAIN_FILE="${SFT_TRAIN_FILE:-datasets/sft/train.jsonl}"
EVAL_FILE="${SFT_EVAL_FILE:-datasets/sft/test.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/sft}"
NGPU="${NGPU:-$(python - <<'PY'
import torch
print(torch.cuda.device_count() if torch.cuda.is_available() else 0)
PY
)}"
STRATEGY="$(PYTHONPATH="${ROOT}:${PYTHONPATH:-}" python - <<'PY'
from utils.resources import choose_sft_plan
print(choose_sft_plan().strategy)
PY
)"

if [[ "${NGPU}" -gt 1 && "${STRATEGY}" == "full_deepspeed_zero2" ]]; then
  torchrun --standalone --nproc_per_node "${NGPU}" \
    -m training.sft.train \
    --model_path "${MODEL_PATH}" \
    --train_file "${TRAIN_FILE}" \
    --eval_file "${EVAL_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    "$@"
else
  echo "[sft] strategy=${STRATEGY}; launching single-process trainer"
  python -m training.sft.train \
    --model_path "${MODEL_PATH}" \
    --train_file "${TRAIN_FILE}" \
    --eval_file "${EVAL_FILE}" \
    --output_dir "${OUTPUT_DIR}" \
    "$@"
fi
