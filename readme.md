# ReSAM Framework (Anonymous)

This repository contains the anonymized implementation of the ReSAM framework used in our paper for reflective data generation, supervised fine-tuning, and post-training safety evaluation.

> This repository is anonymized for double-blind review.

---

## Repository Status

The original project appears to have started as a flat script-based repository. The current checkout has since been partially reorganized into a package under `src/self_reflection_llm/`.

To preserve the original workflow while matching the current code:

- the real implementations now live in `src/self_reflection_llm/`
- root-level compatibility entry points are kept for the main scripts named in the original README
- path handling in the package uses `src/self_reflection_llm/paths.py`, so outputs resolve relative to the repository root instead of hard-coded absolute paths

In other words, this README keeps the original logical structure, but the documented paths below reflect the code that is actually present in this repository.

---

## Project Structure

```text
.
├── config/
│   └── deepspeed_zero2.yaml
├── data_src/
│   ├── DRA_safe_imitation_data_1k_complete.json
│   ├── DrAttack_safe_imitation_data_500_complete.json
│   ├── DRA(benign)_safe_imitation_data_200_complete.json
│   ├── DrAttack(benign)_safe_imitation_data_300_complete.json
│   ├── GPT_reflect/
│   ├── dataset.json
│   └── ...
├── RL_data/
│   ├── general_pattern.json
│   └── harmful_pattern.json
├── results/
│   ├── DRA_processed/
│   └── model_answer/
├── sft.sh
├── generate.py
├── generate_reflection_benign_data.py
├── generate_reflection_data.py
├── eval_llm.py
├── eval_json.py
├── sft_llm_cot.py
└── src/
    └── self_reflection_llm/
        ├── data.py
        ├── eval_json.py
        ├── eval_llm.py
        ├── harmbench_utils.py
        ├── openai_judge.py
        ├── paths.py
        ├── sft_llm_cot.py
        ├── utils.py
        ├── data_src/
        │   ├── clean_data.py
        │   ├── count_data.py
        │   ├── generate.py
        │   ├── generate_harm.py
        │   ├── generate_harm_wrong.py
        │   ├── generate_reflection_benign_data.py
        │   ├── generate_reflection_data_code.py
        │   ├── generate_reflection_data_paragraph.py
        │   ├── generate_reflection_data_table.py
        │   ├── load_hf.py
        │   ├── prompt_config.py
        │   ├── test_openai.py
        │   ├── test_openai_benign.py
        │   └── test_openai_benign_dra.py
        └── results/
            └── DRA_processed/
                └── add_tag.py
```

---

## 1. Data Generation

This repository contains multiple data construction pipelines for producing reflective safe responses from unsafe or partially unsafe model outputs. The high-level logic remains the same as in the original version: generate or collect unsafe or borderline responses, ask another model to reflect on them, insert structured control tokens such as `<|reflect|>`, `<|explore|>`, and `<|continue|>`, and save the resulting training examples.

### Main entry points

- `generate.py`
  Compatibility wrapper for `src/self_reflection_llm/data_src/generate.py`.
  This is the early single-example generation prototype. It contains the core reflection object definition, prompt templates, and token insertion logic:
  - `add_reflect_continue(...)`
  - `add_reflect_continue_explore(...)`
  - `CriticGenerator.generate(...)`

- `generate_reflection_benign_data.py`
  Compatibility wrapper for `src/self_reflection_llm/data_src/generate_reflection_benign_data.py`.
  This script processes attack results and builds both harmful-to-safe and benign-to-safe reflective samples.

  Current behavior in code:
  - reads `DRA/results/attack/batch_llama2_13b_0_120_result.json`
  - iterates over `results -> <original_query> -> qa`
  - keeps harmful items when all of the following hold:
    - `jailbreak_check is True`
    - `em == 1`
    - `harmbench is True`
  - keeps benign items when all of the following hold:
    - `jailbreak_check is False`
    - `em == 0`
    - `harmbench is False`
    - `answer` starts with `"I"`
  - writes incremental JSONL output to `results/DRA_processed/filtered_attack_with_reflections_benign.jsonl`
  - deduplicates items by a SHA256 id derived from query and answer fields

- `generate_reflection_data.py`
  Compatibility dispatcher added for the current repository layout.
  The original project referred to a single `generate_reflection_data.py`, but the current codebase has split that functionality into three task-specific modules:
  - `src/self_reflection_llm/data_src/generate_reflection_data_paragraph.py`
  - `src/self_reflection_llm/data_src/generate_reflection_data_code.py`
  - `src/self_reflection_llm/data_src/generate_reflection_data_table.py`

  Each of the three scripts:
  - loads a pre-collected ReLLM-style source file from a `DRA/...` path
  - calls `CriticGenerator.generate(..., type3=...)`
  - writes JSONL outputs under `results/DRA_processed/`
  - skips previously processed queries already present in the output file

### Supporting generation utilities

- `src/self_reflection_llm/data_src/test_openai.py`
  Generates attack-oriented samples through the OpenAI-compatible API endpoint configured by `OPENAI_API_KEY` and `OPENAI_BASE_URL`.

- `src/self_reflection_llm/data_src/test_openai_benign.py`
  Generates benign imitation data.

- `src/self_reflection_llm/data_src/test_openai_benign_dra.py`
  Reads local `data_src/dataset.json`, filters entries with label `DRA`, and generates benign DRA-style query data.

- `src/self_reflection_llm/data_src/load_hf.py`
  Downloads and restructures subsets from `Deep1994/ReNeLLM-Jailbreak`, then saves separated `latex_table`, `python_code`, and `paragraph` source files for later reflection generation.

- `src/self_reflection_llm/data_src/clean_data.py`
  Cleans generated JSONL or JSON artifacts.

- `src/self_reflection_llm/data_src/count_data.py`
  Counts and inspects generated data distributions.

- `src/self_reflection_llm/data_src/prompt_config.py`
  Stores prompt templates used by the OpenAI-based generation scripts.

### Data format used for training

Most generated samples are normalized into:

```json
{
  "query": "...",
  "reflect_answer": "...\n<|reflect|>\n...\n<|explore|>\n...\n<|continue|>\n..."
}
```

The exact insertion policy differs by script:

- type-2 style samples insert `<|reflect|>` and `<|continue|>`
- type-3 style samples insert `<|reflect|>`, `<|explore|>`, and `<|continue|>`

### Important external dependencies for generation

Several generation scripts expect extra assets that are not present in this checkout but are referenced by code:

- `DRA/results/attack/batch_llama2_13b_0_120_result.json`
- `DRA/data/ReLLM_paragraph.json`
- `DRA/data/ReLLM_python_code.json`
- `DRA/data_source/ReLLM_latex_table.json`

So the code structure is present here, but some generation pipelines require external intermediate files from the original experimental workspace.

---

## 2. Model Training

Training is implemented as supervised fine-tuning over reflective response data.

### `sft.sh`

This is the main launch script used in the current repository.

It sets:

- `PYTHONPATH=src`
- `config/deepspeed_zero2.yaml`
- `ngpu=8`
- `per_device_train_batch_size=1`
- `gradient_accumulation_steps=4`
- `num_train_epochs=5`
- `learning_rate=5e-6`

and launches:

```bash
accelerate launch --config_file config/deepspeed_zero2.yaml \
  -m self_reflection_llm.sft_llm_cot \
  ...
```

The script also computes a derived training schedule from `total_samples=2800`, which matches the dataset composition currently implemented in `ReflectDataset`.

### `sft_llm_cot.py`

Compatibility wrapper for `src/self_reflection_llm/sft_llm_cot.py`.

The core training file currently does the following:

- loads a base causal LM from `model_path`
- optionally enables 4-bit loading
- optionally enables Flash Attention 2
- loads the training dataset through `ReflectDataset`
- converts each example into a chat-style prompt with:
  - system: `"You are a helpful and harmless assistant."`
  - user: `row["question"]`
  - assistant: `row["label"]`
- masks prompt tokens in `labels` so that only the assistant response contributes to loss
- trains with Hugging Face `Trainer`
- saves either `checkpoint-best` or `checkpoint-final`

### Actual training data composition in code

`src/self_reflection_llm/data.py` currently builds the SFT set by concatenating:

1. Local reflective safety data from nine local files:
   - three JSONL files in `results/DRA_processed/`
   - four JSON files in `data_src/`
   - two JSON files in `data_src/GPT_reflect/`
   - each file contributes up to 200 samples

2. Mathematical reasoning data from `Satori-reasoning/Satori_FT_data`
   - types:
     - `type_III_I_data`
     - `type_II_I_data`
     - `type_II_II_data`
   - up to 200 examples per type

3. Benign instruction-following data from `tatsu-lab/alpaca_eval`
   - 400 examples
   - processed by inserting `<|continue|>` markers into the reference output

The combined dataset is shuffled and split with `test_size=0.1`.

This means the current code is not training on only one monolithic reflective file; it is explicitly mixing reflective safety data, math data, and general helpfulness data.

---

## 3. Evaluation

The evaluation stage is split into answer generation and harmfulness scoring.

### `eval_llm.py`

Compatibility wrapper for `src/self_reflection_llm/eval_llm.py`.

This script:

- loads a causal LM from `--model_path`
- formats evaluation prompts using the same system message as training
- generates answers with `max_new_tokens=2048`, `do_sample=True`, `temperature=0.7`
- writes outputs to `results/model_answer/<dataset>.json` by default

Supported datasets in the current code:

- `xstest`
- `strongreject`
- `wildchat`
- `donot`
- `simpleqa`
- `gsm8k`
- `advglue`

These datasets are assembled by helper classes in `src/self_reflection_llm/data.py`, mostly via Hugging Face `load_dataset(...)`.

### `eval_json.py`

Compatibility wrapper for `src/self_reflection_llm/eval_json.py`.

This script evaluates generated answers for harmfulness. It:

- reads `results/model_answer/<dataset>.json`
- extracts `question` and `answer`
- runs HarmBench-style binary classification through `predict(...)` in `harmbench_utils.py`
- runs heuristic jailbreak checks:
  - GCG-style refusal prefix check
  - AutoDAN-style refusal prefix check
  - EM-style lexical overlap check
- writes processed results to:
  - `results/model_answer/<dataset>_check.json`
  - or for `xstest`:
    - `results/model_answer/xstest_harmful_check.json`
    - `results/model_answer/xstest_benign_check.json`

For `gsm8k`, the same script also computes answer correctness by extracting final numeric answers from the model output and comparing them to labels.

### `harmbench_utils.py`

The HarmBench-style classifier is implemented locally with:

- classifier model: `cais/HarmBench-Llama-2-13b-cls`
- 4-bit loading via `BitsAndBytesConfig`
- one-token `yes` or `no` generation for binary harmfulness decisions

This is the main harmfulness scoring backend used by `eval_json.py`.

### `openai_judge.py`

This is a separate OpenAI-based judge used for GSM8K-style numerical consistency checking. It is not the main default evaluation path, but it is included in the repository as an auxiliary judging script.

---

## 4. Result Files

The `results/` directory contains the processed artifacts used by the current codebase.

### `results/DRA_processed/`

This directory stores reflective data derived from DRA or ReLLM-style attack samples.

Examples currently present:

- `GPT_ReLLM(code)310_filtered_attack_with_reflections.jsonl`
- `GPT_ReLLM(paragraph)470_filtered_attack_with_reflections.jsonl`
- `GPT_ReLLM(table)440_filtered_attack_with_reflections.jsonl`
- `LLaMA70b_DRA_filtered_attack_with_reflections.jsonl`
- `LLaMA70b_DRA_filtered_attack_with_reflections_benign.jsonl`

These files are consumed directly by `ReflectDataset` during training.

### `results/model_answer/`

This directory stores model generations and corresponding post-processed evaluations.

Examples:

- `strongreject.json`
  Raw model responses on the StrongReject dataset

- `strongreject_check.json`
  Post-processed harmfulness evaluation results including:
  - HarmBench classifier output
  - jailbreak string checks
  - EM overlap check

- `gsm8k.json`
  Raw model responses on GSM8K

- `gsm8k_stats.json` or `gsm8k_evaluated.json`
  Auxiliary numeric evaluation artifacts generated by the GSM8K judging pipeline

In the processed harmfulness files:

- `True` means the response is classified as harmful or successful under the corresponding check
- `False` means the response is classified as safe or unsuccessful under that check

---

## 5. Running the Code

### Environment

At minimum, the current code expects:

- Python with PyTorch and Transformers
- `accelerate`
- `datasets`
- `peft`
- `bitsandbytes` for 4-bit paths
- a CUDA environment for the default training and HarmBench evaluation paths

For OpenAI-compatible generation or judging scripts, also set:

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=...
```

### Canonical commands

Training:

```bash
PYTHONPATH=src accelerate launch --config_file config/deepspeed_zero2.yaml -m self_reflection_llm.sft_llm_cot ...
```

Generate evaluation answers:

```bash
PYTHONPATH=src python -m self_reflection_llm.eval_llm --eval_ds=strongreject
```

Run harmfulness evaluation:

```bash
PYTHONPATH=src python -m self_reflection_llm.eval_json strongreject
```

### Compatibility commands

The root-level wrappers allow the original script naming style:

```bash
python eval_llm.py --eval_ds=strongreject
python eval_json.py strongreject
python generate.py
python generate_reflection_benign_data.py
python generate_reflection_data.py --format paragraph
```

The package-style commands are still the canonical implementation path.

---

## 6. Notes on Differences from the Original README

Compared with the original flat README description, the current repository differs in three important ways:

1. `generate_reflection_data.py` is no longer a single implementation file.
   It has been split into `code`, `paragraph`, and `table` variants.

2. Training is more concrete than the original summary implied.
   The current code mixes:
   - local reflective safety data
   - math reasoning data
   - Alpaca-style benign data

3. Evaluation is partly model-based and partly heuristic.
   Harmfulness is not only checked by prompting an LLM; the repository also runs lexical jailbreak checks and exact-match-style overlap checks.

This README therefore preserves the original workflow narrative, but updates the details to match the actual code paths and files in the repository.

---

## TODO

- Investigate what types of generalization capabilities emerge from the framework.
- Analyze how internal model parameters and representations change after training.
- Consolidate the remaining external `DRA/` dependencies so the full generation pipeline can be reproduced from a single checkout.
