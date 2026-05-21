#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SFT_SOURCE_OUTPUT_DIR="${SFT_SOURCE_OUTPUT_DIR:-${ROOT}/data/sft/source}"
LIMIT_PER_SOURCE="${LIMIT_PER_SOURCE:--1}"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1

"${PYTHON_BIN}" -m self_reflection_llm.datasets.sft_source_data \
  --output_dir "${SFT_SOURCE_OUTPUT_DIR}" \
  --limit_per_source "${LIMIT_PER_SOURCE}"
