#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

python -m utils.cleanup --execute \
  data \
  src \
  config \
  docs \
  requirements-rl.txt \
  readme.md \
  scripts/sft/prepare_data.sh \
  scripts/sft/build_source_data.sh \
  scripts/rl/run_gdpo.sh \
  scripts/rl/check_env.py \
  scripts/analysis \
  .pytest_cache
