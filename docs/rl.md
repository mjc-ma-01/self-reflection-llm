# Reflector RL Workflow

This document covers the compact RL integration for the Reflector release. The
repository keeps paper-specific data conversion, reward logic, and GDPO helpers
locally, while distributed rollout and PPO/GDPO training are delegated to a
GDPO-capable `verl` checkout.

## Components

- `src/self_reflection_llm/datasets/rl_data.py` converts
  `data/rl/general_pattern.json` and `data/rl/harmful_pattern.json` into
  verl-compatible train/test parquet files.
- `src/self_reflection_llm/datasets/eval_data.py` prepares validation parquet
  files for safety, math, and general QA checks.
- `src/self_reflection_llm/rl/reward.py` implements the segmented trajectory
  reward used by the RL stage: final safety/correctness plus a reflection
  component based on `<|reflect|>`, `<|explore|>`, and `<|continue|>`.
- `src/self_reflection_llm/rl/gdpo.py` contains a framework-neutral GDPO helper
  for group reward-decoupled normalization.
- `scripts/rl/workflow.sh` is the end-to-end entry point.
- `scripts/rl/run_gdpo.sh` is the lower-level verl launch script.

## One-Command Preparation

Use a Python executable from an environment with `datasets`, `pyarrow`, `torch`,
`transformers`, and the intended `verl` checkout available.

```bash
PYTHON_BIN=python \
VERL_ROOT=<verl-checkout> \
scripts/rl/workflow.sh --prepare-only
```

This runs environment checks, builds RL parquet under
`data/processed/reflector_rl`, builds validation parquet under
`data/processed/reflector_eval`, and runs a reward smoke test.

## Prepare RL Parquet

```bash
PYTHONPATH=src python -m self_reflection_llm.datasets.rl_data \
  --general_path data/rl/general_pattern.json \
  --harmful_path data/rl/harmful_pattern.json \
  --output_dir data/processed/reflector_rl
```

The output records use verl's standard fields: `data_source`, `prompt`,
`ability`, `reward_model`, and `extra_info`. The train split mixes general and
harmful prompts; the test split is sampled with `--test_size`.

Useful smoke-run overrides:

```bash
PYTHONPATH=src python -m self_reflection_llm.datasets.rl_data \
  --limit_per_file 16 \
  --output_dir data/processed/reflector_rl_smoke
```

## Prepare Validation Parquet

```bash
PYTHONPATH=src python -m self_reflection_llm.datasets.eval_data \
  --datasets xstest strongreject wildchat donot gsm8k simpleqa \
  --output_dir data/processed/reflector_eval \
  --num_samples 200
```

The generated files can be added to `VAL_FILES` for verl validation.

## Reward Judge

For paper-style safety and helpfulness rewards, start an OpenAI-compatible judge
service and set:

```bash
export RM_BASE_URL=http://127.0.0.1:30000
export RM_MODEL=<reward-judge-model>
```

If no remote judge is configured, harmful prompts use a refusal-keyword fallback
and math prompts use local boxed-answer matching. General helpfulness prompts
need the remote judge for meaningful rewards.

## Run GDPO

```bash
MODEL_PATH=<reflector-sft-checkpoint> \
PYTHON_BIN=python \
VERL_ROOT=<verl-checkout> \
N_GPUS_PER_NODE=8 \
scripts/rl/workflow.sh --train
```

The launch script requires a `verl` checkout that recognizes
`algorithm.adv_estimator=gdpo`. The repository-local `gdpo.py` keeps the
decoupled normalization logic small and testable.

Useful overrides:

```bash
TRAIN_FILES="['data/processed/reflector_rl/train.parquet']" \
VAL_FILES="['data/processed/reflector_rl/test.parquet','data/processed/reflector_eval/xstest.parquet','data/processed/reflector_eval/gsm8k.parquet']" \
REFLECT_BONUS_CORRECT=0.3 \
REFLECT_PENALTY_WRONG=-0.3 \
ROLLOUT_N=8 \
bash scripts/rl/run_gdpo.sh trainer.total_epochs=1
```

The reward function emits:

- `reward_component_correctness`: final answer safety/correctness.
- `reward_component_reflect`: `+1` for effective reflection, `-1` for
  reflection followed by an unsafe or incorrect final answer, and `0` otherwise.

With GDPO weights `[1.0, 0.3]`, this matches the paper's dual-reward design.
