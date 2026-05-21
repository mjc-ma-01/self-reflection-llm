# Reflector SFT Raw Sources

This directory documents the released inputs used to rebuild
`data/sft/ready/train.jsonl` and `data/sft/ready/test.jsonl`.

The main training path does not need to run generation. It reads the ready JSONL
files directly. Rebuild them only when you want to reproduce or modify the SFT
data mixture:

```bash
bash scripts/sft/reproduce_data.sh
```

The released ready data is rebuilt from:

- processed teacher-reflection files under `results/DRA_processed/`
- released JSON/JSONL source files under `data/source/`
- optional Hugging Face math reflection data when `INCLUDE_HF_MATH=1`

The teacher-reflection generation utilities remain under
`src/self_reflection_llm/generation/` for full data-construction reproduction.
