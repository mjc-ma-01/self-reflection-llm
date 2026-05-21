"""TGR: Teacher-Guided Reflection Generation for Reflector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from self_reflection_llm.generation.query_openai import GPT5Client, require_text
from self_reflection_llm.paths import PROCESSED_DATA_DIR

SYSTEM = """You are GPT-5 acting as the teacher model in the REFLECTOR paper.
You generate structured self-critique for a truncated trajectory. Return JSON only."""

USER_TEMPLATE = """Stage: Teacher-Guided Reflection Generation (TGR)

Paper specification:
Given an indirect jailbreak query x and a truncated prefix y_before, generate
z = (z_reflect, z_explore). z_reflect identifies the precise ethical/safety breach.
z_explore charts a safe path forward for subsequent generation.

Self-Critique & Reflection Template:
You are a student. Review your incorrect partial solution, reflecting on mistakes and
proposing alternative ethical solutions.
1. Analyze the provided step-by-step solution.
2. Produce one concise first-person critique and one concise alternative ethical claim.
Do not include meta-text outside the required JSON.

Original query x:
{query}

Truncated prefix y_before:
{y_before}

Required JSON keys:
- z_reflect: first-person reflection identifying the exact mistake/safety breach.
- z_explore: first-person safe exploration plan that redirects toward a lawful, harmless response.
"""


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=Path, default=PROCESSED_DATA_DIR / "reflector_generation" / "ti.jsonl")
    parser.add_argument("--output_file", type=Path, default=PROCESSED_DATA_DIR / "reflector_generation" / "tgr.jsonl")
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
                user=USER_TEMPLATE.format(query=row["query"], y_before=row["y_before"]),
                max_tokens=1200,
                temperature=0.2,
            )
            z_reflect = require_text(response, "z_reflect")
            z_explore = require_text(response, "z_explore")
            out = {
                **row,
                "z_reflect": z_reflect,
                "z_explore": z_explore,
                "reflection": f"<|reflect|>\n{z_reflect}\n<|explore|>\n{z_explore}",
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[TGR] wrote {out['id']}")


if __name__ == "__main__":
    main()
