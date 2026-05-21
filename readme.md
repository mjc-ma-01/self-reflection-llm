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
│   ├── source/                  # released source assets
│   ├── sft/                     # prepared SFT train/test plus manifest
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
    ├── datasets/                # RL/eval parquet builders and legacy loaders
    ├── evaluation/              # answer generation, harmfulness scoring, judges
    ├── generation/              # TI/TGR/RTC reflection-data generation
    ├── hub/                     # Hugging Face Hub utilities
    ├── rl/                      # segmented reward and GDPO helpers
    ├── training/                # SFT entry points
    └── paths.py
```

Path handling is centralized in `src/self_reflection_llm/paths.py`.

## Data Assets

- `data/sft/train.jsonl` and `data/sft/test.jsonl` are the prepared SFT files.
- `data/sft/manifest.json` records the source files used to produce them.
- `data/source/` contains released auxiliary source assets.
- `data/rl/general_pattern.json` and `data/rl/harmful_pattern.json` are merged
  from `data/source/` by source-level `ability` and are the default prompt sets
  for RL internalization.
- Generated JSONL/parquet artifacts are written under `data/processed/`.

## Stage 1: SFT

In this repository, **prepare means generating reflection trajectories**. The
checked-in `data/sft/train.jsonl` and `data/sft/test.jsonl` files are the
default SFT inputs, while this optional reproduction module regenerates raw
reflection trajectories with the original OpenAI prompt templates restored in
`src/self_reflection_llm/generation/prompt_config.py`.

1. **TI: Trajectory Initialization** calls GPT-5 with one restored prompt
   preset and parses the generated `query` plus `reflect_answer`.
2. **TGR: Teacher-Guided Reflection Generation** extracts `z_reflect` and
   `z_explore` from the generated reflection markers.
3. **RTC: Reflection-Based Trajectory Construction** writes the final
   reflection trajectory in the SFT JSONL schema.

Run all three generation stages:

```bash
OPENAI_API_KEY=<key> \
OPENAI_MODEL=gpt-5 \
PROMPT_PRESET=dra \
INPUT_FILE=data/rl/harmful_pattern.json \
bash scripts/sft/prepare_data.sh
```

Available `PROMPT_PRESET` values are `dra`, `drattack`,
`dra_answer_benign`, `drattack_answer_benign`, `dra_benign`, and
`drattack_benign`.

The final generated reflection trajectories are written to:

```text
data/processed/reflector_generation/reflection_trajectories.jsonl
```

The stage modules can also be run independently:

```bash
PYTHONPATH=src python -m self_reflection_llm.generation.trajectory_initialization \
  --input_file data/rl/harmful_pattern.json \
  --output_file data/processed/reflector_generation/ti.jsonl

PYTHONPATH=src python -m self_reflection_llm.generation.teacher_guided_reflection \
  --input_file data/processed/reflector_generation/ti.jsonl \
  --output_file data/processed/reflector_generation/tgr.jsonl

PYTHONPATH=src python -m self_reflection_llm.generation.reflection_trajectory_construction \
  --input_file data/processed/reflector_generation/tgr.jsonl \
  --output_file data/processed/reflector_generation/reflection_trajectories.jsonl
```

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
  --model_path <base-model-or-checkpoint> \
  --train_file data/sft/train.jsonl \
  --eval_file data/sft/test.jsonl
```

Validate an SFT checkpoint by generating answers and running post-hoc scoring:

```bash
MODEL_PATH=<sft-output>/checkpoint-best \
EVAL_DATASETS="strongreject xstest gsm8k" \
bash scripts/sft/validate_sft.sh
```

## Stage 2: RL

Install the lightweight RL preparation dependencies:

```bash
pip install -r requirements-rl.txt
```

Rebuild the merged RL pattern files from `data/source/` whenever source data is
changed:

```bash
bash scripts/rl/build_patterns_from_source.sh
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

The generation folder intentionally contains only the OpenAI query helper and
the three paper stages:

- `query_openai.py`
- `trajectory_initialization.py` (`TI`)
- `teacher_guided_reflection.py` (`TGR`)
- `reflection_trajectory_construction.py` (`RTC`)

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
