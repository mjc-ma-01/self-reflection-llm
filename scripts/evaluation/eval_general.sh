#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-${ROOT}/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HUGGINGFACE_HUB_CACHE}}"

python -m evaluation.run \
  --model_path "${MODEL_PATH:-checkpoints/sft/merged}" \
  --datasets general \
  --num_samples "${NUM_SAMPLES:-50}" \
  --batch_size "${BATCH_SIZE:-4}" \
  --output_dir outputs/results \
  "$@"
