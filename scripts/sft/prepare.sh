#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

python -m training.sft.prepare \
  --source_dir "${SFT_SOURCE_DIR:-datasets/sft/source}" \
  --output_dir "${SFT_OUTPUT_DIR:-datasets/sft}" \
  --test_size "${TEST_SIZE:-0.1}" \
  --seed "${SEED:-42}" \
  --limit_per_pattern "${LIMIT_PER_PATTERN:--1}" \
  "$@"

