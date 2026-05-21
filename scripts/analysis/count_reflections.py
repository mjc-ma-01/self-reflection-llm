#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path


DEFAULT_FILES = ("strongreject_check.json", "wildchat_check.json", "xstest_harmful_check.json")


def count_file(path: Path) -> tuple[int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    reflect_count = 0
    harmbench_true = 0
    for content in data.values():
        for item in content.get("detail", []):
            if item.get("harmbench") is True:
                harmbench_true += 1
                if "<|reflect|>" in item.get("answer", ""):
                    reflect_count += 1
    return reflect_count, harmbench_true


def main() -> None:
    parser = argparse.ArgumentParser(description="Count reflection markers in checked model-answer JSON files.")
    parser.add_argument("--base_dir", type=Path, default=Path("results/model_answer/llama8b-harm-sft"))
    parser.add_argument("--files", nargs="+", default=list(DEFAULT_FILES))
    args = parser.parse_args()

    print("Reflection marker analysis")
    print("-" * 60)

    total_reflect_count = 0
    total_harmbench_true = 0
    for filename in args.files:
        path = args.base_dir / filename
        try:
            reflect_count, harmbench_true = count_file(path)
        except FileNotFoundError:
            print(f"Missing file: {path}\n")
            continue
        except json.JSONDecodeError:
            print(f"Failed to parse JSON: {path}\n")
            continue

        percentage = (reflect_count / harmbench_true * 100) if harmbench_true > 0 else 0
        print(f"File: {path.name}")
        print(f"  harmbench=true: {harmbench_true}")
        print(f"  contains <|reflect|>: {reflect_count}")
        print(f"  ratio: {percentage:.2f}%\n")
        total_reflect_count += reflect_count
        total_harmbench_true += harmbench_true

    if total_harmbench_true > 0:
        total_percentage = total_reflect_count / total_harmbench_true * 100
        print("-" * 60)
        print("Total")
        print(f"  harmbench=true: {total_harmbench_true}")
        print(f"  contains <|reflect|>: {total_reflect_count}")
        print(f"  ratio: {total_percentage:.2f}%")
    else:
        print("No harmbench=true entries found.")


if __name__ == "__main__":
    main()
