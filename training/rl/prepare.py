from __future__ import annotations

import argparse
import json
from pathlib import Path

from utils.paths import OUTPUTS_DIR, RL_DIR, repo_relative
from utils.patterns import load_rl_patterns, rl_row, split_records

DEFAULT_OUTPUT_DIR = OUTPUTS_DIR / "processed" / "rl"


def _write_parquet(rows: list[dict], path: Path) -> None:
    import datasets as hf_datasets

    path.parent.mkdir(parents=True, exist_ok=True)
    hf_datasets.Dataset.from_list(rows).to_parquet(str(path))


def prepare_rl_data(
    *,
    rl_dir: Path = RL_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    test_size: float = 0.02,
    seed: int = 42,
    limit_per_pattern: int = -1,
    system_prompt: str | None = None,
) -> dict[str, object]:
    records = load_rl_patterns(rl_dir, limit_per_pattern=limit_per_pattern)
    rows = [rl_row(record, system_prompt=system_prompt) for record in records]
    train_rows, test_rows = split_records(rows, test_size=test_size, seed=seed)
    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"
    _write_parquet(train_rows, train_path)
    _write_parquet(test_rows, test_path)
    metadata = {
        "schema_version": "pattern-v1-verl",
        "rl_dir": repo_relative(rl_dir),
        "train_file": repo_relative(train_path),
        "test_file": repo_relative(test_path),
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
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare GDPO/RL parquet from harmful/general pattern files.")
    parser.add_argument("--rl_dir", type=Path, default=RL_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--test_size", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_per_pattern", type=int, default=-1)
    parser.add_argument("--system_prompt", default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    print(json.dumps(prepare_rl_data(**vars(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
