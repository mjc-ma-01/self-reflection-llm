#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_FILE="${INPUT_FILE:-${ROOT}/data/rl/harmful_pattern.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/data/processed/reflector_generation}"
LIMIT="${LIMIT:--1}"
SEED="${SEED:-42}"
TI_FILE="${TI_FILE:-${OUTPUT_DIR}/ti.jsonl}"
TGR_FILE="${TGR_FILE:-${OUTPUT_DIR}/tgr.jsonl}"
RTC_FILE="${RTC_FILE:-${OUTPUT_DIR}/reflection_trajectories.jsonl}"

export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1

mkdir -p "${OUTPUT_DIR}"

echo "[SFT prepare][TI] Trajectory Initialization"
ti_args=(
  --input_file "${INPUT_FILE}"
  --output_file "${TI_FILE}"
  --limit "${LIMIT}"
  --seed "${SEED}"
)
if [[ -n "${MODEL_TRAJECTORY_FIELD:-}" ]]; then
  ti_args+=(--model_trajectory_field "${MODEL_TRAJECTORY_FIELD}")
fi
"${PYTHON_BIN}" -m self_reflection_llm.generation.trajectory_initialization "${ti_args[@]}"

echo "[SFT prepare][TGR] Teacher-Guided Reflection Generation"
"${PYTHON_BIN}" -m self_reflection_llm.generation.teacher_guided_reflection \
  --input_file "${TI_FILE}" \
  --output_file "${TGR_FILE}" \
  --limit "${LIMIT}"

echo "[SFT prepare][RTC] Reflection-Based Trajectory Construction"
"${PYTHON_BIN}" -m self_reflection_llm.generation.reflection_trajectory_construction \
  --input_file "${TGR_FILE}" \
  --output_file "${RTC_FILE}" \
  --limit "${LIMIT}"

echo "[SFT prepare] wrote ${RTC_FILE}"
