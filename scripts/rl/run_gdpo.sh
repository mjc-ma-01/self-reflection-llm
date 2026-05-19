#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Requires a verl build with GDPO support, matching the paper's trainer setup.
if [[ -z "${MODEL_PATH:-}" ]]; then
  echo "MODEL_PATH must point to the SFT checkpoint or base model." >&2
  exit 2
fi

TRAIN_FILES="${TRAIN_FILES:-['${ROOT}/data/processed/reflector_rl/train.parquet']}"
VAL_FILES="${VAL_FILES:-['${ROOT}/data/processed/reflector_rl/test.parquet']}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-reflector_gdpo}"
PROJECT_NAME="${PROJECT_NAME:-reflector}"
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
ROLLOUT_N="${ROLLOUT_N:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-4096}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-8}"
LR="${LR:-1.0e-6}"
REFLECT_BONUS_CORRECT="${REFLECT_BONUS_CORRECT:-0.3}"
REFLECT_PENALTY_WRONG="${REFLECT_PENALTY_WRONG:--0.3}"
REFLECT_REWARD_PROMPT_TYPES="${REFLECT_REWARD_PROMPT_TYPES:-harmful}"
PYTHON_BIN="${PYTHON_BIN:-python}"
VERL_ROOT="${VERL_ROOT:-}"

if [[ -n "${VERL_ROOT}" ]]; then
  export PYTHONPATH="${ROOT}/src:${VERL_ROOT}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"
fi
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"

"${PYTHON_BIN}" -m verl.trainer.main_ppo \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE_PER_GPU}" \
  actor_rollout_ref.actor.optim.lr="${LR}" \
  algorithm.adv_estimator=gdpo \
  algorithm.gdpo_reward_keys='[reward_component_correctness,reward_component_reflect]' \
  algorithm.gdpo_reward_weights='[1.0,0.3]' \
  custom_reward_function.path="${ROOT}/src/self_reflection_llm/rl/reward.py" \
  custom_reward_function.name=compute_score_segmented \
  custom_reward_function.reward_kwargs.reflect_bonus_correct="${REFLECT_BONUS_CORRECT}" \
  custom_reward_function.reward_kwargs.reflect_penalty_wrong="${REFLECT_PENALTY_WRONG}" \
  custom_reward_function.reward_kwargs.reflect_reward_prompt_types="${REFLECT_REWARD_PROMPT_TYPES}" \
  trainer.project_name="${PROJECT_NAME}" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.nnodes="${NNODES}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE}" \
  "$@"
