#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
EVAL_DATASETS="${EVAL_DATASETS:-strongreject xstest gsm8k}"
BATCH_SIZE="${BATCH_SIZE:-4}"

if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "MODEL_PATH is required." >&2
  exit 2
fi

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1

for dataset in ${EVAL_DATASETS}; do
  save_path="${RESULTS_DIR:-${ROOT}/results/model_answer}/sft_${dataset}.json"
  "${PYTHON_BIN}" -m self_reflection_llm.evaluation.generate_answers \
    --model_path "${MODEL_PATH}" \
    --eval_ds "${dataset}" \
    --save_path "${save_path}" \
    --batch_size "${BATCH_SIZE}"

  if [[ "${RUN_SCORE:-1}" == "1" ]]; then
    "${PYTHON_BIN}" -m self_reflection_llm.evaluation.score_answers "${dataset}" \
      --input_path "${save_path}" \
      --output_dir "${RESULTS_DIR:-${ROOT}/results/model_answer}" || {
        echo "Scoring failed for ${dataset}; generated answers remain at ${save_path}." >&2
        exit 1
      }
  fi
done
