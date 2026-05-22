from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from utils.paths import SFT_DIR, SFT_SOURCE_DIR, repo_relative
from utils.patterns import load_sft_patterns, sft_row, split_records, write_jsonl

DEFAULT_SYSTEM_PROMPT = "You are a helpful and harmless assistant."


def prepare_sft_data(
    *,
    source_dir: Path = SFT_SOURCE_DIR,
    output_dir: Path = SFT_DIR,
    test_size: float = 0.1,
    seed: int = 42,
    limit_per_pattern: int = -1,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> dict[str, object]:
    records = load_sft_patterns(source_dir, limit_per_pattern=limit_per_pattern)
    rows = [sft_row(record, system_prompt=system_prompt) for record in records]
    train_rows, test_rows = split_records(rows, test_size=test_size, seed=seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "train.jsonl", train_rows)
    write_jsonl(output_dir / "test.jsonl", test_rows)

    metadata = {
        "schema_version": "pattern-v1",
        "source_dir": repo_relative(source_dir),
        "train_file": repo_relative(output_dir / "train.jsonl"),
        "test_file": repo_relative(output_dir / "test.jsonl"),
        "records": len(rows),
        "train_records": len(train_rows),
        "test_records": len(test_rows),
        "pattern_counts": {
            "harmful": sum(row["pattern_type"] == "harmful" for row in rows),
            "general": sum(row["pattern_type"] == "general" for row in rows),
        },
        "args": {
            "test_size": test_size,
            "seed": seed,
            "limit_per_pattern": limit_per_pattern,
            "system_prompt": system_prompt,
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": "pattern-v1",
        "description": "SFT data built only from harmful_pattern.json and general_pattern.json.",
        "sources": [
            {"pattern_type": "harmful", "path": repo_relative(source_dir / "harmful_pattern.json")},
            {"pattern_type": "general", "path": repo_relative(source_dir / "general_pattern.json")},
        ],
        "train_file": repo_relative(output_dir / "train.jsonl"),
        "test_file": repo_relative(output_dir / "test.jsonl"),
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build SFT train/test JSONL from harmful/general pattern files.")
    parser.add_argument("--source_dir", type=Path, default=SFT_SOURCE_DIR)
    parser.add_argument("--output_dir", type=Path, default=SFT_DIR)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_per_pattern", type=int, default=-1)
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    metadata = prepare_sft_data(**vars(args))
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
