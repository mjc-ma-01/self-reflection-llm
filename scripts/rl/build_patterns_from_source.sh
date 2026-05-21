#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
LIMIT_PER_SOURCE="${LIMIT_PER_SOURCE:--1}"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1

"${PYTHON_BIN}" -m self_reflection_llm.datasets.source_patterns \
  --general_output "${ROOT}/data/rl/general_pattern.json" \
  --harmful_output "${ROOT}/data/rl/harmful_pattern.json" \
  --limit_per_source "${LIMIT_PER_SOURCE}"
