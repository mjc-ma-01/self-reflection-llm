#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-${ROOT}/data/sft/ready}"
LIMIT_PER_FILE="${LIMIT_PER_FILE:-200}"
HELPFUL_LIMIT="${HELPFUL_LIMIT:-400}"
TEST_SIZE="${TEST_SIZE:-0.1}"
SEED="${SEED:-42}"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1

args=(
  --output_dir "${SFT_OUTPUT_DIR}"
  --limit_per_file "${LIMIT_PER_FILE}"
  --helpful_limit "${HELPFUL_LIMIT}"
  --test_size "${TEST_SIZE}"
  --seed "${SEED}"
)

if [[ "${INCLUDE_HF_MATH:-0}" == "1" ]]; then
  args+=(--include_hf_math --hf_math_samples_per_type "${HF_MATH_SAMPLES_PER_TYPE:-200}")
fi

"${PYTHON_BIN}" -m self_reflection_llm.datasets.sft_data \
  "${args[@]}"
