# Reflector

This repository contains the code and released data assets for the Reflector
pipeline: reflective data construction, supervised fine-tuning, RL
internalization, and evaluation.

The repository uses a package-first layout. Runnable commands call package
modules with `python -m`; root-level Python wrapper scripts are intentionally not
kept.

## Repository Layout

```text
.
├── config/
│   └── deepspeed_zero2.yaml
├── data/
│   ├── source/                  # released source/SFT data assets
│   └── rl/                      # released RL prompt data
├── docs/
│   └── rl.md
├── requirements-rl.txt
├── results/
│   ├── DRA_processed/           # processed reflection data used by SFT
│   └── model_answer/            # released evaluation examples
├── scripts/
│   ├── analysis/
│   ├── rl/
│   │   ├── check_env.py
│   │   ├── run_gdpo.sh
│   │   └── workflow.sh
│   └── sft/
│       └── train_sft.sh
└── src/self_reflection_llm/
    ├── datasets/                # SFT mixtures and RL/eval parquet builders
    ├── evaluation/              # generation, scoring, and judges
    ├── generation/              # reflection-data generation utilities
    ├── hub/                     # Hugging Face Hub utilities
    ├── rl/                      # reward and GDPO helpers
    ├── training/                # SFT entry points
    └── paths.py
```

Path handling is centralized in `src/self_reflection_llm/paths.py`.
Repository data lives under `data/source/` and `data/rl/`; generated parquet
files are written to `data/processed/`, which is ignored by git.

## Data Assets

- `data/source/` contains the released reflective source data, benign/safety
  imitation data, and auxiliary generation inputs.
- `data/rl/general_pattern.json` and `data/rl/harmful_pattern.json` are the
  default prompt sets for the RL internalization stage.
- `results/DRA_processed/` contains processed reflection JSONL files consumed by
  the SFT dataset mixture.

## Reflective Data Generation

Generation utilities live in `src/self_reflection_llm/generation/`.

Common entry points:

```bash
PYTHONPATH=src python -m self_reflection_llm.generation.generate
PYTHONPATH=src python -m self_reflection_llm.generation.generate_reflection_benign_data
PYTHONPATH=src python -m self_reflection_llm.generation.generate_reflection_data_paragraph
PYTHONPATH=src python -m self_reflection_llm.generation.generate_reflection_data_code
PYTHONPATH=src python -m self_reflection_llm.generation.generate_reflection_data_table
```

For local reflection-critic generation, set the critic model explicitly:

```bash
export REFLECTOR_CRITIC_MODEL=<critic-model-or-checkpoint>
```

Some generation scripts expect external intermediate attack files from the
original data construction workflow. The released processed files needed for SFT
are already included under `results/DRA_processed/`.

## Supervised Fine-Tuning

The SFT stage trains on the dataset mixture implemented in
`src/self_reflection_llm/datasets/mixture.py`. The mixture combines:

- processed reflective safety data from `results/DRA_processed/`
- released source data from `data/source/`
- math reasoning data from Hugging Face
- benign instruction-following data from Hugging Face

Launch SFT with:

```bash
bash scripts/sft/train_sft.sh
```

The underlying module is:

```bash
PYTHONPATH=src accelerate launch \
  --config_file config/deepspeed_zero2.yaml \
  -m self_reflection_llm.training.sft \
  --model_path <base-model-or-checkpoint>
```

The default base model in code is `meta-llama/Llama-3.1-8B-Instruct`. Override
`--model_path` for the checkpoint used in your environment.

## Reinforcement Learning

The RL stage uses `verl` for rollout/training and keeps paper-specific logic in
this repository:

- `src/self_reflection_llm/datasets/rl_data.py` builds train/test parquet files
  from `data/rl/`.
- `src/self_reflection_llm/datasets/eval_data.py` builds validation parquet
  files for safety, math, and general QA checks.
- `src/self_reflection_llm/rl/reward.py` implements the segmented reward:
  final correctness/safety plus a reflection component based on
  `<|reflect|>`, `<|explore|>`, and `<|continue|>`.
- `src/self_reflection_llm/rl/gdpo.py` contains the small framework-neutral
  GDPO normalization helper.

Install the light RL preparation dependencies:

```bash
pip install -r requirements-rl.txt
```

Run the runnable preparation and smoke-test flow:

```bash
PYTHON_BIN=python scripts/rl/workflow.sh --prepare-only
```

This checks the environment, converts `data/rl/*.json` to parquet under
`data/processed/reflector_rl/`, optionally builds validation parquet under
`data/processed/reflector_eval/`, and runs a reward smoke test.

Prepare only the RL parquet files:

```bash
PYTHONPATH=src python -m self_reflection_llm.datasets.rl_data \
  --general_path data/rl/general_pattern.json \
  --harmful_path data/rl/harmful_pattern.json \
  --output_dir data/processed/reflector_rl
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

See `docs/rl.md` for reward details, judge fallback behavior, and useful
overrides.

## Evaluation

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

Run the auxiliary OpenAI-compatible judge:

```bash
PYTHONPATH=src python -m self_reflection_llm.evaluation.openai_judge \
  --input results/model_answer/gsm8k.json
```

For OpenAI-compatible generation or judging scripts, set:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
```

## Quick Checks

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m compileall -q src scripts tests
bash -n scripts/rl/workflow.sh scripts/rl/run_gdpo.sh scripts/sft/train_sft.sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest tests/test_rl_reward.py
```
