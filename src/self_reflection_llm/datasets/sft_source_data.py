"""Build SFT source pattern files before the train/test split.

The source grouping follows ``mixture.py``:
- the eight requested ReflectDataset-style files are grouped as harmful/safety;
- the remaining local general mixture uses correct-correct plus alpaca_eval.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from self_reflection_llm.paths import DATA_SRC_DIR, DRA_PROCESSED_DIR, PROJECT_ROOT


SFT_SOURCE_DIR = PROJECT_ROOT / "data" / "sft" / "source"


@dataclass(frozen=True)
class SourceSpec:
    data_source: str
    path: Path
    ability: str
    limit: int


HARMFUL_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        "sft_rellm_code",
        DRA_PROCESSED_DIR / "GPT_ReLLM(code)310_filtered_attack_with_reflections.jsonl",
        "safety",
        200,
    ),
    SourceSpec(
        "sft_rellm_paragraph",
        DRA_PROCESSED_DIR / "GPT_ReLLM(paragraph)470_filtered_attack_with_reflections.jsonl",
        "safety",
        200,
    ),
    SourceSpec(
        "sft_rellm_table",
        DRA_PROCESSED_DIR / "GPT_ReLLM(table)440_filtered_attack_with_reflections.jsonl",
        "safety",
        200,
    ),
    SourceSpec(
        "sft_dra_safe_imitation",
        DATA_SRC_DIR / "DRA_safe_imitation_data_1k_complete.json",
        "safety",
        200,
    ),
    SourceSpec(
        "sft_drattack_safe_imitation",
        DATA_SRC_DIR / "DrAttack_safe_imitation_data_500_complete.json",
        "safety",
        200,
    ),
    SourceSpec(
        "sft_drattack_benign_imitation",
        DATA_SRC_DIR / "DrAttack(benign)_safe_imitation_data_300_complete.json",
        "safety",
        200,
    ),
    SourceSpec(
        "sft_dra_benign_imitation",
        DATA_SRC_DIR / "DRA(benign)_safe_imitation_data_200_complete.json",
        "safety",
        200,
    ),
    SourceSpec(
        "sft_gpt_wrong_correct",
        DATA_SRC_DIR / "GPT_reflect" / "wrong-correct.json",
        "safety",
        200,
    ),
)

GENERAL_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec(
        "sft_gpt_correct_correct",
        DATA_SRC_DIR / "GPT_reflect" / "correct-correct.json",
        "general",
        200,
    ),
    SourceSpec("sft_alpaca_eval", DATA_SRC_DIR / "alpaca_eval.json", "general", 400),
)

QUERY_KEYS = ("query", "question", "instruction", "prompt", "text_query")
LABEL_KEYS = ("reflect_answer", "output", "answer", "response", "label")


def _repo_display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        rows.append(item)
        return rows
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON list in {path}")
    return [item for item in data if isinstance(item, dict)]


def _first_text(item: dict[str, Any], keys: Iterable[str]) -> tuple[str | None, str | None]:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return key, value.strip()
    return None, None


def _insert_continue_tokens(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    sentences = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    first_part, rest_part = (sentences[0], "") if len(sentences) == 1 else sentences
    first_part = first_part + "\n<|continue|>\n"
    if not rest_part:
        return first_part
    chunk_size = len(rest_part) // 4
    new_rest = ""
    for idx in range(3):
        new_rest += rest_part[idx * chunk_size : (idx + 1) * chunk_size]
        new_rest += "<|continue|>"
    new_rest += rest_part[3 * chunk_size :]
    return first_part + new_rest


def _record(spec: SourceSpec, item: dict[str, Any], source_idx: int) -> dict[str, Any] | None:
    input_field, query = _first_text(item, QUERY_KEYS)
    _, label = _first_text(item, LABEL_KEYS)
    if not query or not label:
        return None
    if spec.data_source == "sft_alpaca_eval":
        label = _insert_continue_tokens(label)
    return {
        "query": query,
        "label": label,
        "data_source": spec.data_source,
        "ability": spec.ability,
        "source_file": _repo_display_path(spec.path),
        "source_idx": source_idx,
        "input_field": input_field,
    }


def build_source_data(limit_per_source: int = -1) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    harmful: list[dict[str, Any]] = []
    general: list[dict[str, Any]] = []
    for spec, target in [(spec, harmful) for spec in HARMFUL_SOURCES] + [(spec, general) for spec in GENERAL_SOURCES]:
        if not spec.path.exists():
            raise FileNotFoundError(spec.path)
        effective_limit = spec.limit if limit_per_source < 0 else min(spec.limit, limit_per_source)
        for source_idx, item in enumerate(_load_records(spec.path)):
            if source_idx >= effective_limit:
                break
            record = _record(spec, item, source_idx)
            if record:
                target.append(record)
    return harmful, general


def _write_json(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=SFT_SOURCE_DIR)
    parser.add_argument("--limit_per_source", type=int, default=-1)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir.expanduser()
    harmful, general = build_source_data(limit_per_source=args.limit_per_source)
    _write_json(harmful, output_dir / "harmful_pattern.json")
    _write_json(general, output_dir / "general_pattern.json")
    metadata = {
        "description": "SFT source data before train/test split, grouped by ReflectDataset mixture logic.",
        "harmful_records": len(harmful),
        "general_records": len(general),
        "harmful_sources": [spec.__dict__ | {"path": _repo_display_path(spec.path)} for spec in HARMFUL_SOURCES],
        "general_sources": [spec.__dict__ | {"path": _repo_display_path(spec.path)} for spec in GENERAL_SOURCES],
    }
    _write_json(metadata, output_dir / "metadata.json")
    print(f"Wrote {output_dir / 'harmful_pattern.json'} ({len(harmful)} rows)")
    print(f"Wrote {output_dir / 'general_pattern.json'} ({len(general)} rows)")
    print(f"Wrote {output_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
