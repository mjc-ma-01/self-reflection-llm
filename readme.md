# Reflector

This repository implements the two-stage Reflector pipeline described in
`7030_Reflector_Internalizing_S.pdf`: supervised fine-tuning first teaches a
model explicit step-wise reflection patterns, then reinforcement learning
internalizes those patterns with outcome and reflection-validity rewards.

The repository is organized as package-first code under `src/self_reflection_llm`.
Run commands with `PYTHONPATH=src python -m ...`; root-level compatibility
wrappers are intentionally not kept.

## Two-Stage Design

Reflector targets delayed or indirect jailbreaks, where harmful content can
appear after an initially benign-looking generation trajectory. The training
flow is therefore split into two modules.

1. **SFT stage**: teacher-guided reflection data is converted into supervised
   examples containing structured markers such as `<|reflect|>`,
   `<|explore|>`, and `<|continue|>`. This stage establishes the response
   format and teaches the base model how to revise unsafe intermediate
   trajectories.
2. **RL stage**: a GDPO-compatible `verl` run optimizes the SFT checkpoint with
   two reward components: final outcome correctness or safety, and validity of
   the reflection trajectory. This stage encourages autonomous reflection rather
   than only imitating teacher traces.

## Repository Layout

```text
.
├── config/
│   └── deepspeed_zero2.yaml
├── data/
│   ├── source/                  # released source and SFT data assets
│   ├── rl/                      # released RL prompt data
│   └── processed/               # generated train/eval artifacts, gitignored
├── docs/
│   └── rl.md
├── requirements-rl.txt
├── results/
│   ├── DRA_processed/           # processed reflection JSONL used by SFT
│   └── model_answer/            # generated evaluation answers and scores
├── scripts/
│   ├── analysis/
│   ├── sft/
│   │   ├── prepare_data.sh
│   │   ├── train_sft.sh
│   │   └── validate_sft.sh
│   └── rl/
│       ├── check_env.py
│       ├── run_gdpo.sh
│       ├── validate_rl.sh
│       └── workflow.sh
└── src/self_reflection_llm/
    ├── datasets/                # SFT mixtures and RL/eval parquet builders
    ├── evaluation/              # answer generation, harmfulness scoring, judges
    ├── generation/              # reflection-data generation utilities
    ├── hub/                     # Hugging Face Hub utilities
    ├── rl/                      # segmented reward and GDPO helpers
    ├── training/                # SFT entry points
    └── paths.py
```

Path handling is centralized in `src/self_reflection_llm/paths.py`.

## Data Assets

- `data/source/` contains released reflective source data, benign/safety
  imitation data, and auxiliary generation inputs.
- `results/DRA_processed/` contains reflection JSONL files consumed by the SFT
  mixture.
- `data/rl/general_pattern.json` and `data/rl/harmful_pattern.json` are the
  default prompt sets for RL internalization.
- Generated JSONL/parquet artifacts are written under `data/processed/`.

## Stage 1: SFT

Prepare the SFT train/test JSONL files and metadata:

```bash
bash scripts/sft/prepare_data.sh
```

This runs:

```bash
PYTHONPATH=src python -m self_reflection_llm.datasets.sft_data \
  --output_dir data/processed/reflector_sft
```

The SFT dataset is implemented in
`src/self_reflection_llm/datasets/mixture.py` and combines:

- processed reflective safety data from `results/DRA_processed/`
- released source data from `data/source/`
- math reasoning data from Hugging Face
- benign instruction-following data from Hugging Face

Train the SFT model:

```bash
MODEL_PATH=<base-model-or-checkpoint> \
OUTPUT_ROOT=outputs/reflect_models \
bash scripts/sft/train_sft.sh
```

The underlying module is:

```bash
PYTHONPATH=src accelerate launch \
  --config_file config/deepspeed_zero2.yaml \
  -m self_reflection_llm.training.sft \
  --model_path <base-model-or-checkpoint>
```

Validate an SFT checkpoint by generating answers and running post-hoc scoring:

```bash
MODEL_PATH=outputs/reflector_sft/checkpoint-best \
EVAL_DATASETS="strongreject xstest gsm8k" \
bash scripts/sft/validate_sft.sh
```

## Stage 2: RL

Install the lightweight RL preparation dependencies:

```bash
pip install -r requirements-rl.txt
```

Prepare RL train/test parquet files and validation parquet files:

```bash
PYTHON_BIN=python scripts/rl/workflow.sh --prepare-only
```

Equivalent direct RL data command:

```bash
PYTHONPATH=src python -m self_reflection_llm.datasets.rl_data \
  --general_path data/rl/general_pattern.json \
  --harmful_path data/rl/harmful_pattern.json \
  --output_dir data/processed/reflector_rl
```

Run RL validation preparation and reward smoke tests:

```bash
PYTHON_BIN=python scripts/rl/validate_rl.sh
```

Run GDPO training with a GDPO-capable `verl` checkout:

```bash
MODEL_PATH=<reflector-sft-checkpoint> \
PYTHON_BIN=python \
VERL_ROOT=<verl-checkout> \
RM_BASE_URL=http://127.0.0.1:30000 \
RM_MODEL=<reward-judge-model> \
scripts/rl/workflow.sh --train
```

The RL module keeps paper-specific logic local:

- `src/self_reflection_llm/datasets/rl_data.py` builds verl-compatible
  train/test parquet files.
- `src/self_reflection_llm/datasets/eval_data.py` builds validation parquet
  files for safety, math, and general QA.
- `src/self_reflection_llm/rl/reward.py` implements the segmented reward:
  final correctness/safety plus the reflection component.
- `src/self_reflection_llm/rl/gdpo.py` implements group reward-decoupled
  normalization helpers for tests and integration.

See `docs/rl.md` for reward details, judge fallback behavior, and common
overrides.

## Reflective Data Generation

Generation utilities live in `src/self_reflection_llm/generation/`.

```bash
PYTHONPATH=src python -m self_reflection_llm.generation.generate
PYTHONPATH=src python -m self_reflection_llm.generation.generate_reflection_benign_data
PYTHONPATH=src python -m self_reflection_llm.generation.generate_reflection_data_paragraph
PYTHONPATH=src python -m self_reflection_llm.generation.generate_reflection_data_code
PYTHONPATH=src python -m self_reflection_llm.generation.generate_reflection_data_table
```

For local reflection-critic generation, set:

```bash
export REFLECTOR_CRITIC_MODEL=<critic-model-or-checkpoint>
```

Some generation scripts expect external intermediate attack files from the
original data construction workflow. The released processed files needed for SFT
are already included under `results/DRA_processed/`.

## Standalone Evaluation

Generate model answers:

```bash
PYTHONPATH=src python -m self_reflection_llm.evaluation.generate_answers \
  --model_path <model-or-checkpoint> \
  --eval_ds strongreject
```

Run harmfulness or task scoring:

```bash
PYTHONPATH=src python -m self_reflection_llm.evaluation.score_answers strongreject
```

For OpenAI-compatible generation or judging scripts, set:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
```

## Quick Checks

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m compileall -q src scripts tests
bash -n scripts/sft/prepare_data.sh scripts/sft/train_sft.sh scripts/sft/validate_sft.sh
bash -n scripts/rl/workflow.sh scripts/rl/run_gdpo.sh scripts/rl/validate_rl.sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest tests/test_rl_reward.py
```
