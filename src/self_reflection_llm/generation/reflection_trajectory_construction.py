"""RTC: Reflection-Based Trajectory Construction for Reflector.

This stage converts the reflection-augmented raw answer from TI/TGR into the
SFT JSONL schema. It does not call another prompt; the prepared trajectory comes
from the original OpenAI prompt template restored in ``prompt_config.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from self_reflection_llm.paths import PROCESSED_DATA_DIR


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _valid(text: str) -> bool:
    return "<|reflect|>" in text and "<|explore|>" in text and "<|continue|>" in text


def _reflect_answer(row: dict[str, Any]) -> str:
    raw_reflect_answer = row.get("raw_reflect_answer")
    if isinstance(raw_reflect_answer, str) and raw_reflect_answer.strip():
        return raw_reflect_answer.strip()
    return (
        f"{row['y_before']}\n\n<|reflect|>\n{row['z_reflect']}"
        f"\n<|explore|>\n{row['z_explore']}\n<|continue|>"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=Path, default=PROCESSED_DATA_DIR / "reflector_generation" / "tgr.jsonl")
    parser.add_argument("--output_file", type=Path, default=PROCESSED_DATA_DIR / "reflector_generation" / "reflection_trajectories.jsonl")
    parser.add_argument("--limit", type=int, default=-1)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = _read_jsonl(args.input_file)
    if args.limit >= 0:
        rows = rows[: args.limit]
    args.output_file.expanduser().parent.mkdir(parents=True, exist_ok=True)

    with args.output_file.expanduser().open("w", encoding="utf-8") as handle:
        for row in rows:
            reflect_answer = _reflect_answer(row)
            out = {
                "id": row["id"],
                "query": row["query"],
                "reflect_answer": reflect_answer,
                "prompt": [{"role": "user", "content": row["query"]}],
                "response": reflect_answer,
                "data_source": "reflector_sft_generated",
                "ability": "safety",
                "extra_info": {
                    "stage": "reflection_trajectory_construction",
                    "truncation_index": row["truncation_index"],
                    "valid_markers": _valid(reflect_answer),
                    "source": row.get("source"),
                },
                "ti": {"trajectory": row["trajectory"], "y_before": row["y_before"]},
                "tgr": {"z_reflect": row["z_reflect"], "z_explore": row["z_explore"]},
            }
            if not out["extra_info"]["valid_markers"]:
                raise ValueError(f"constructed trajectory missing required markers: {out['id']}")
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[RTC] wrote {out['id']}")


if __name__ == "__main__":
    main()
