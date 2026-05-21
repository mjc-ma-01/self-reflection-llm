"""RTC: Reflection-Based Trajectory Construction for Reflector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from self_reflection_llm.generation.query_openai import GPT5Client, require_text
from self_reflection_llm.paths import PROCESSED_DATA_DIR

SYSTEM = """You are GPT-5 completing REFLECTOR supervised trajectories.
Continue from reflection guidance into a safe final answer. Return JSON only."""

USER_TEMPLATE = """Stage: Reflection-Based Trajectory Construction (RTC)

Paper specification:
Assemble a complete reflection-augmented trajectory tau_tilde = (y_before, z, y_after).
The continuation y_after must follow z_reflect and z_explore and terminate in a safe state.

Safety Generation Prompt Template:
Your task is to continue writing a few brief sentences, adhering to helpful,
respectful, and honest principles. You must strictly refrain from offering methods,
details, or instructions to carry out illegal, violent, or harmful actions. In your
continuation, you must:
- explicitly and briefly refuse to assist with unsafe activity when applicable;
- provide specific legal alternatives such as law enforcement, legal aid, mediation,
  cybersecurity hardening, or professional assistance;
- remain logically consistent with the reflection and exploration.

Original query x:
{query}

Prefix y_before:
{y_before}

Reflection z_reflect:
{z_reflect}

Exploration z_explore:
{z_explore}

Required JSON keys:
- y_after: safe continuation text only. Do not repeat the special tokens."""


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
    client = GPT5Client()
    args.output_file.expanduser().parent.mkdir(parents=True, exist_ok=True)

    with args.output_file.expanduser().open("w", encoding="utf-8") as handle:
        for row in rows:
            response = client.json_completion(
                system=SYSTEM,
                user=USER_TEMPLATE.format(
                    query=row["query"],
                    y_before=row["y_before"],
                    z_reflect=row["z_reflect"],
                    z_explore=row["z_explore"],
                ),
                max_tokens=1600,
                temperature=0.2,
            )
            y_after = require_text(response, "y_after")
            reflect_answer = (
                f"{row['y_before']}\n\n<|reflect|>\n{row['z_reflect']}"
                f"\n<|explore|>\n{row['z_explore']}\n<|continue|>\n{y_after}"
            )
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
