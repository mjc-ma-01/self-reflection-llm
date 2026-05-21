#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RL_OUTPUT_DIR="${RL_OUTPUT_DIR:-${ROOT}/data/processed/reflector_rl}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${ROOT}/data/processed/reflector_eval}"
EVAL_DATASETS="${EVAL_DATASETS:-xstest strongreject wildchat donot gsm8k simpleqa}"
EVAL_NUM_SAMPLES="${EVAL_NUM_SAMPLES:-200}"

export PYTHONPATH="${ROOT}/src:${VERL_ROOT:-}:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1

"${PYTHON_BIN}" "${ROOT}/scripts/rl/check_env.py"

"${PYTHON_BIN}" -m self_reflection_llm.datasets.rl_data \
  --general_path "${ROOT}/data/rl/general_pattern.json" \
  --harmful_path "${ROOT}/data/rl/harmful_pattern.json" \
  --output_dir "${RL_OUTPUT_DIR}" \
  --test_size "${TEST_SIZE:-0.02}" \
  --seed "${SEED:-42}" \
  --limit_per_file "${LIMIT_PER_FILE:--1}"

# shellcheck disable=SC2086
"${PYTHON_BIN}" -m self_reflection_llm.datasets.eval_data \
  --datasets ${EVAL_DATASETS} \
  --output_dir "${EVAL_OUTPUT_DIR}" \
  --num_samples "${EVAL_NUM_SAMPLES}" \
  --seed "${SEED:-42}"

"${PYTHON_BIN}" - <<'PY'
from self_reflection_llm.rl.gdpo import group_decoupled_advantages
from self_reflection_llm.rl.reward import MARK_CONTINUE, MARK_REFLECT, compute_score_segmented

score = compute_score_segmented(
    data_source="reflector_harmful",
    solution_str=f"unsafe draft {MARK_REFLECT} this is unsafe {MARK_CONTINUE} I cannot help with that.",
    ground_truth="refuse",
    extra_info={"prompt_type": "harmful", "expected_behavior": "refuse", "question": "unsafe request"},
)
assert score["reward_component_correctness"] == 1.0, score
assert score["reward_component_reflect"] == 1.0, score

advantages = group_decoupled_advantages(
    group_ids=["a", "a", "b", "b"],
    reward_components={
        "reward_component_correctness": [1.0, 0.0, 1.0, 1.0],
        "reward_component_reflect": [1.0, -1.0, 0.0, 1.0],
    },
)
assert len(advantages) == 4, advantages
print("[reflector-rl] validation smoke passed")
PY
