#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

python -m training.rl.prepare \
  --rl_dir "${RL_DIR:-datasets/rl}" \
  --output_dir "${RL_OUTPUT_DIR:-outputs/processed/rl}" \
  --test_size "${TEST_SIZE:-0.02}" \
  --seed "${SEED:-42}" \
  --limit_per_pattern "${LIMIT_PER_PATTERN:--1}" \
  "$@"

