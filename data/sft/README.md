# Reflector SFT Data

This directory contains the prepared SFT data and its source manifest in one
place.

The main training path does not need to run generation. It reads the ready JSONL
files directly:

- `train.jsonl`
- `test.jsonl`
- `metadata.json`
- `manifest.json`
- `source/harmful_pattern.json`
- `source/general_pattern.json`

The released ready data is rebuilt from:

- processed teacher-reflection files under `results/DRA_processed/`
- released JSON/JSONL source files under `data/source/`
- optional Hugging Face math reflection data when `INCLUDE_HF_MATH=1`

The `source/` files are the SFT source records before train/test split. They
follow the local `mixture.py` counts: 200 rows per harmful source, 200
`correct-correct` general rows, and 400 `alpaca_eval` general rows.

The teacher-reflection generation utilities remain under
`src/self_reflection_llm/generation/` for full data-construction reproduction.
