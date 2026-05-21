"""TI: Trajectory Initialization for Reflector SFT data generation.

This stage calls GPT-5 with the original OpenAI prompt templates used by the
project to generate a raw ``query`` and ``reflect_answer`` pair. The generated
answer is then truncated before the reflection marker to provide ``y_before``
for the next stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from self_reflection_llm.generation import prompt_config
from self_reflection_llm.generation.query_openai import GPT5Client
from self_reflection_llm.paths import PROCESSED_DATA_DIR

SYSTEM = "You are a precise data generation assistant. Follow instructions exactly."

PROMPT_PRESETS = {
    "dra": prompt_config.DRA,
    "dra_answer_benign": prompt_config.DRA_answer_benign,
    "drattack": prompt_config.DrAttack,
    "drattack_answer_benign": prompt_config.DrAttack_answer_benign,
}

GENERAL_QUESTION_PRESETS = {
    "dra_benign": prompt_config.DRA_benign,
    "drattack_benign": prompt_config.DrAttack_benign,
}


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
    for key in ("query", "question", "general_question", "prompt", "instruction", "text_query"):
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


def _parse_generated_pair(raw_text: str) -> dict[str, str]:
    if "===QUERY===" not in raw_text or "===ANSWER===" not in raw_text:
        raise ValueError("GPT output must contain ===QUERY=== and ===ANSWER=== markers.")
    after_query = raw_text.split("===QUERY===", 1)[1]
    query_text, answer_text = after_query.split("===ANSWER===", 1)
    query = query_text.strip()
    reflect_answer = answer_text.strip()
    if not query:
        raise ValueError("generated query is empty.")
    if "<|continue|>" not in reflect_answer or "<|reflect|>" not in reflect_answer:
        raise ValueError("generated reflect_answer is missing required reflection markers.")
    return {"query": query, "reflect_answer": reflect_answer}


def _initial_trajectory(reflect_answer: str) -> str:
    return reflect_answer.split("<|reflect|>", 1)[0].strip()


def _build_prompt(preset: str, item: dict[str, Any]) -> str:
    if preset in GENERAL_QUESTION_PRESETS:
        return GENERAL_QUESTION_PRESETS[preset]().format(general_question=_query(item))
    return PROMPT_PRESETS[preset]()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_file", type=Path, required=True)
    parser.add_argument("--output_file", type=Path, default=PROCESSED_DATA_DIR / "reflector_generation" / "ti.jsonl")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_trajectory_field", type=str, default=None)
    parser.add_argument("--prompt_preset", choices=sorted(PROMPT_PRESETS | GENERAL_QUESTION_PRESETS), default="dra")
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
                raw_reflect_answer = str(item.get("reflect_answer", trajectory)).strip()
            else:
                assert client is not None
                raw_text = client.text_completion(
                    system=SYSTEM,
                    user=_build_prompt(args.prompt_preset, item),
                    max_tokens=4000,
                    temperature=0.7,
                )
                generated = _parse_generated_pair(raw_text)
                query = generated["query"]
                raw_reflect_answer = generated["reflect_answer"]
                trajectory = _initial_trajectory(raw_reflect_answer)
                response = {"trajectory": trajectory, "unsafe_signal_step": 1, "notes": f"Generated by {args.prompt_preset}."}
            unsafe_signal_step = response.get("unsafe_signal_step")
            unsafe_step = int(unsafe_signal_step) if isinstance(unsafe_signal_step, int) else None
            y_before, truncation_index = _prefix(trajectory, rng, unsafe_step)
            out = {
                "id": _record_id(query, idx),
                "query": query,
                "trajectory": trajectory,
                "truncation_index": truncation_index,
                "y_before": y_before,
                "raw_reflect_answer": raw_reflect_answer,
                "ti_notes": str(response.get("notes", "")),
                "source": {"file": str(args.input_file), "index": idx},
            }
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            handle.flush()
            print(f"[TI] wrote {out['id']}")


if __name__ == "__main__":
    main()
