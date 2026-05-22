#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HUGGINGFACE_HUB_CACHE}}"

MODEL_PATH="${MODEL_PATH:-checkpoints/sft/merged}"
EVAL_DATASETS="${EVAL_DATASETS:-strongreject xstest wildchat donot gsm8k general}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
BATCH_SIZE="${BATCH_SIZE:-4}"

python -m evaluation.run \
  --model_path "${MODEL_PATH}" \
  --datasets ${EVAL_DATASETS} \
  --num_samples "${NUM_SAMPLES}" \
  --batch_size "${BATCH_SIZE}" \
  --output_dir outputs/results \
  "$@"
