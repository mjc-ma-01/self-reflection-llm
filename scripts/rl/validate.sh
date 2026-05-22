#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

python -m training.rl.check_env
python -m training.rl.prepare \
  --rl_dir datasets/rl \
  --output_dir outputs/processed/rl \
  --limit_per_pattern "${LIMIT_PER_PATTERN:-16}"

