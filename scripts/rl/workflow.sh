#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_CHECK="${RUN_CHECK:-1}"
RUN_PREPARE_RL="${RUN_PREPARE_RL:-1}"
RUN_PREPARE_EVAL="${RUN_PREPARE_EVAL:-1}"
RUN_REWARD_SMOKE="${RUN_REWARD_SMOKE:-1}"
RUN_TRAIN="${RUN_TRAIN:-0}"

RL_OUTPUT_DIR="${RL_OUTPUT_DIR:-${ROOT}/data/processed/reflector_rl}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${ROOT}/data/processed/reflector_eval}"
EVAL_DATASETS="${EVAL_DATASETS:-xstest strongreject wildchat donot gsm8k simpleqa}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-200}"
LIMIT_PER_FILE="${LIMIT_PER_FILE:--1}"
TEST_SIZE="${TEST_SIZE:-0.02}"
SEED="${SEED:-42}"

export PYTHONPATH="${ROOT}/src:${VERL_ROOT:-}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1

usage() {
  cat <<'EOF'
Usage:
  scripts/rl/workflow.sh [--check-only] [--prepare-only] [--train]

Environment overrides:
  PYTHON_BIN                 Python executable in the training environment.
  VERL_ROOT                  Optional path to a local verl checkout.
  MODEL_PATH                 Required when --train or RUN_TRAIN=1.
  RL_OUTPUT_DIR              Default: data/processed/reflector_rl.
  EVAL_OUTPUT_DIR            Default: data/processed/reflector_eval.
  EVAL_DATASETS              Space-separated validation datasets.
  EVAL_NUM_SAMPLES           Default: 200.
  LIMIT_PER_FILE             Limit source RL examples for smoke runs.
  RM_BASE_URL / RM_MODEL     Optional OpenAI-compatible reward judge.

Examples:
  PYTHON_BIN=python scripts/rl/workflow.sh --prepare-only
  MODEL_PATH=<sft-checkpoint> VERL_ROOT=<verl-checkout> scripts/rl/workflow.sh --train
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only)
      RUN_PREPARE_RL=0
      RUN_PREPARE_EVAL=0
      RUN_REWARD_SMOKE=1
      RUN_TRAIN=0
      shift
      ;;
    --prepare-only)
      RUN_TRAIN=0
      shift
      ;;
    --train)
      RUN_TRAIN=1
      shift
      ;;
    --skip-eval)
      RUN_PREPARE_EVAL=0
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

echo "[reflector-rl] root=${ROOT}"
echo "[reflector-rl] python=${PYTHON_BIN}"

if [[ "${RUN_CHECK}" == "1" ]]; then
  "${PYTHON_BIN}" "${ROOT}/scripts/rl/check_env.py"
fi

if [[ "${RUN_PREPARE_RL}" == "1" ]]; then
  "${PYTHON_BIN}" -m self_reflection_llm.datasets.rl_data \
    --general_path "${ROOT}/data/rl/general_pattern.json" \
    --harmful_path "${ROOT}/data/rl/harmful_pattern.json" \
    --output_dir "${RL_OUTPUT_DIR}" \
    --test_size "${TEST_SIZE}" \
    --seed "${SEED}" \
    --limit_per_file "${LIMIT_PER_FILE}"
fi

if [[ "${RUN_PREPARE_EVAL}" == "1" ]]; then
  # shellcheck disable=SC2086
  "${PYTHON_BIN}" -m self_reflection_llm.datasets.eval_data \
    --datasets ${EVAL_DATASETS} \
    --output_dir "${EVAL_OUTPUT_DIR}" \
    --num_samples "${EVAL_NUM_SAMPLES}" \
    --seed "${SEED}"
fi

if [[ "${RUN_REWARD_SMOKE}" == "1" ]]; then
  "${PYTHON_BIN}" - <<'PY'
from self_reflection_llm.rl.reward import MARK_CONTINUE, MARK_REFLECT, compute_score_segmented

result = compute_score_segmented(
    data_source="reflector_harmful",
    solution_str=f"unsafe draft {MARK_REFLECT} this is unsafe {MARK_CONTINUE} I cannot help with that.",
    ground_truth="refuse",
    extra_info={"prompt_type": "harmful", "expected_behavior": "refuse", "question": "bad request"},
)
assert result["reward_component_correctness"] == 1.0, result
assert result["reward_component_reflect"] == 1.0, result
print("[reflector-rl] reward smoke passed")
PY
fi

if [[ "${RUN_TRAIN}" == "1" ]]; then
  "${PYTHON_BIN}" "${ROOT}/scripts/rl/check_env.py" --strict
  if [[ -z "${MODEL_PATH:-}" ]]; then
    echo "MODEL_PATH is required for training." >&2
    exit 2
  fi
  TRAIN_FILES="${TRAIN_FILES:-['${RL_OUTPUT_DIR}/train.parquet']}" \
  VAL_FILES="${VAL_FILES:-['${RL_OUTPUT_DIR}/test.parquet']}" \
  PYTHON_BIN="${PYTHON_BIN}" \
  VERL_ROOT="${VERL_ROOT:-}" \
  bash "${ROOT}/scripts/rl/run_gdpo.sh"
fi

echo "[reflector-rl] workflow complete"
