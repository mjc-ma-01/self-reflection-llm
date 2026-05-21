"""TI: Trajectory Initialization for Reflector SFT data generation.

Given an indirect jailbreak query x, generate an initial target-policy
trajectory and sample a prefix y_before. This implements the paper's first
Stage-I data construction phase before teacher reflection is added.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from self_reflection_llm.generation.query_openai import GPT5Client, require_text
from self_reflection_llm.paths import PROCESSED_DATA_DIR

SYSTEM = """You are GPT-5 acting as the trajectory-initialization engine for the REFLECTOR paper.
Generate structured supervised-data artifacts only. The purpose is safety research and reflective
alignment. Do not include operational details that would enable harm; when a harmful step would
become actionable, replace the operational content with concise placeholders such as
[unsafe operational detail omitted]. Return JSON only."""

USER_TEMPLATE = """Stage: Trajectory Initialization (TI)

Paper specification:
For an indirect jailbreak query x, generate a full step sequence tau = (y_1, ..., y_T)
from a target policy, then choose an intermediate truncation index n and retain only
y_before = (y_1, ..., y_n). The suffix is discarded at this stage.

Input query x:
{query}

Required JSON keys:
- trajectory: string, a step-by-step initial trajectory with steps separated by exactly two newlines.
- unsafe_signal_step: integer, the first step where unsafe intent/risk becomes visible.
- notes: string, one sentence explaining why this is a useful reflection trigger.

The trajectory should simulate a model beginning to comply with the prompt, but must keep
dangerous operational content non-actionable by using placeholders where needed."""


def _load_items(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    data = json.load(path.open("r", encoding="utf-8"))
    if isinstance(data, dict):
        data = data.get("data") or data.get("items") or data.get("queries") or [data]
    if not isinstance(data, list):
        raise ValueError(f"expected JSON list or JSONL in {path}")
    return [item for item in data if isinstance(item, dict)]


def _query(item: dict[str, Any]) -> str:
    for key in ("query", "question", "prompt", "instruction", "text_query"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise ValueError(f"missing query-like field in keys: {sorted(item)}")


def _record_id(query: str, idx: int) -> str:
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()[:16]
    return f"ti-{idx:06d}-{digest}"


def _prefix(trajectory: str, rng: random.Random, unsafe_signal_step: int | None) -> tuple[str, int]:
    steps = [step.strip() for step in trajectory.split("\n\n") if step.strip()]
    if not steps:
        return trajectory.strip(), 1
    if unsafe_signal_step and 1 <= unsafe_signal_step <= len(steps):
        low = max(1, unsafe_signal_step)
        high = min(len(steps), unsafe_signal_step + 1)
        n = rng.randint(low, high)
    else:
        n = rng.randint(1, len(steps))
    return "\n\n".join(steps[:n]), n


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, default=PROCESSED_DATA_DIR / "reflector_generation" / "ti.jsonl")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_trajectory_field", type=str, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = _load_items(args.input_file.expanduser())
    if args.limit >= 0:
        rows = rows[: args.limit]
    rng = random.Random(args.seed)
    client = None if args.model_trajectory_field else GPT5Client()
    args.output_file.expanduser().parent.mkdir(parents=True, exist_ok=True)

    with args.output_file.expanduser().open("w", encoding="utf-8") as handle:
        for idx, item in enumerate(rows):
            query = _query(item)
            if args.model_trajectory_field and item.get(args.model_trajectory_field):
                trajectory = str(item[args.model_trajectory_field]).strip()
                response = {"trajectory": trajectory, "unsafe_signal_step": 1, "notes": "Loaded from source field."}
            else:
                assert client is not None
                response = client.json_completion(
                    system=SYSTEM,
                    user=USER_TEMPLATE.format(query=query),
                    max_tokens=1800,
                    temperature=0.4,
                )
                trajectory = require_text(response, "trajectory")
            unsafe_signal_step = response.get("unsafe_signal_step")
            unsafe_step = int(unsafe_signal_step) if isinstance(unsafe_signal_step, int) else None
            y_before, truncation_index = _prefix(trajectory, rng, unsafe_step)
            out = {
                "id": _record_id(query, idx),
                "query": query,
                "trajectory": trajectory,
                "truncation_index": truncation_index,
                "y_before": y_before,
                "ti_notes": str(response.get("notes", "")),
                "source": {"file": str(args.input_file), "index": idx},
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[TI] wrote {out['id']}")


if __name__ == "__main__":
    main()
