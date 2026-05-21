#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SFT_OUTPUT_DIR="${SFT_OUTPUT_DIR:-${ROOT}/data/processed/reflector_sft}"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1

"${PYTHON_BIN}" -m self_reflection_llm.datasets.sft_data \
  --output_dir "${SFT_OUTPUT_DIR}"
