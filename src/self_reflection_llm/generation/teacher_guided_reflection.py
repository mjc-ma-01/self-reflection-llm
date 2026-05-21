"""TGR: Teacher-Guided Reflection Generation for Reflector.

This stage extracts teacher reflection and exploration spans from the raw
GPT-5 output produced by TI. It intentionally reuses the original prompt output
instead of introducing a second, incompatible prompt format.
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


def _until_marker(text: str) -> str:
    candidates = [idx for marker in ("<|explore|>", "<|continue|>", "<|reflect|>") if (idx := text.find(marker)) >= 0]
    return text[: min(candidates)].strip() if candidates else text.strip()


def _extract_reflection(raw_reflect_answer: str) -> tuple[str, str]:
    if "<|reflect|>" not in raw_reflect_answer:
        raise ValueError("raw_reflect_answer is missing <|reflect|>.")
    after_reflect = raw_reflect_answer.split("<|reflect|>", 1)[1]
    if "<|explore|>" in after_reflect:
        z_reflect, after_explore = after_reflect.split("<|explore|>", 1)
        z_explore = _until_marker(after_explore)
    else:
        z_reflect = _until_marker(after_reflect)
        z_explore = ""
    if not z_reflect.strip():
        raise ValueError("empty z_reflect extracted from raw_reflect_answer.")
    return z_reflect.strip(), z_explore.strip()


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
    args.output_file.expanduser().parent.mkdir(parents=True, exist_ok=True)

    with args.output_file.expanduser().open("w", encoding="utf-8") as handle:
        for row in rows:
            raw_reflect_answer = row.get("raw_reflect_answer")
            if not isinstance(raw_reflect_answer, str) or not raw_reflect_answer.strip():
                raise ValueError(f"TI row {row.get('id')} is missing raw_reflect_answer.")
            z_reflect, z_explore = _extract_reflection(raw_reflect_answer)
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
