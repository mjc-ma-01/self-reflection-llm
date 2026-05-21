"""Export the Reflector SFT mixture to train/test JSONL files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from self_reflection_llm.datasets.mixture import ReflectDataset
from self_reflection_llm.paths import PROCESSED_DATA_DIR

MARKERS = ("<|reflect|>", "<|explore|>", "<|continue|>")


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _split_to_records(split) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(split):
        question = str(row["question"])
        answer = str(row["label"])
        records.append(
            {
                "id": idx,
                "question": question,
                "label": answer,
                "has_reflect": "<|reflect|>" in answer,
                "has_explore": "<|explore|>" in answer,
                "has_continue": "<|continue|>" in answer,
            }
        )
    return records


def _marker_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for marker in MARKERS:
        counts[marker] = sum(marker in row["label"] for row in records)
    return counts


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=PROCESSED_DATA_DIR / "reflector_sft")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir.expanduser()

    dataset = ReflectDataset().get_dataset()
    train_records = _split_to_records(dataset["train"])
    test_records = _split_to_records(dataset["test"])

    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"
    metadata_path = output_dir / "metadata.json"
    _write_jsonl(train_records, train_path)
    _write_jsonl(test_records, test_path)

    metadata = {
        "train_records": len(train_records),
        "test_records": len(test_records),
        "train_marker_counts": _marker_counts(train_records),
        "test_marker_counts": _marker_counts(test_records),
        "schema": {
            "question": "user instruction or attack prompt",
            "label": "assistant response containing Reflector markers when applicable",
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {train_path} ({len(train_records)} rows)")
    print(f"Wrote {test_path} ({len(test_records)} rows)")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
