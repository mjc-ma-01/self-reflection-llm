from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Literal

from .paths import RL_DIR, SFT_SOURCE_DIR, repo_relative

PatternType = Literal["harmful", "general"]

PATTERN_FILENAMES: dict[PatternType, str] = {
    "harmful": "harmful_pattern.json",
    "general": "general_pattern.json",
}

QUERY_KEYS = ("query", "question", "prompt", "instruction", "text_query")
RESPONSE_KEYS = ("response", "label", "answer", "output", "reflect_answer")


@dataclass(frozen=True)
class PatternRecord:
    pattern_type: PatternType
    query: str
    response: str | None
    ability: str
    source_idx: int
    source_file: str
    extra: dict[str, Any]


def load_json_list(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [item for item in data if isinstance(item, dict)]


def write_json(path: str | Path, payload: Any) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path).expanduser()
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
    return rows


def _first_text(item: dict[str, Any], keys: Iterable[str]) -> tuple[str | None, str | None]:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return key, value.strip()
    return None, None


def normalize_pattern_record(
    item: dict[str, Any],
    *,
    pattern_type: PatternType,
    source_idx: int,
    source_file: str | Path,
    require_response: bool,
) -> PatternRecord:
    input_field, query = _first_text(item, QUERY_KEYS)
    response_field, response = _first_text(item, RESPONSE_KEYS)
    if not query:
        raise ValueError(f"{source_file}:{source_idx} missing query field; keys={sorted(item)}")
    if require_response and not response:
        raise ValueError(f"{source_file}:{source_idx} missing response field; keys={sorted(item)}")
    expected_ability = "safety" if pattern_type == "harmful" else "general"
    ability = str(item.get("ability") or expected_ability)
    legacy_keys = {"data_source", "source_file", "source_idx", "prompt_type", "extra_info"}
    extra = {
        key: value
        for key, value in item.items()
        if key not in set(QUERY_KEYS) | set(RESPONSE_KEYS) | {"prompt", "ability"} | legacy_keys
    }
    extra.update(
        {
            "input_field": input_field,
            "response_field": response_field,
            "source_idx": source_idx,
            "pattern_type": pattern_type,
        }
    )
    return PatternRecord(
        pattern_type=pattern_type,
        query=query,
        response=response,
        ability=ability,
        source_idx=source_idx,
        source_file=repo_relative(source_file),
        extra=extra,
    )


def load_pattern_file(
    path: str | Path,
    *,
    pattern_type: PatternType,
    require_response: bool = False,
    limit: int = -1,
) -> list[PatternRecord]:
    raw = load_json_list(path)
    if limit >= 0:
        raw = raw[:limit]
    return [
        normalize_pattern_record(
            item,
            pattern_type=pattern_type,
            source_idx=idx,
            source_file=path,
            require_response=require_response,
        )
        for idx, item in enumerate(raw)
    ]


def load_sft_patterns(
    source_dir: str | Path = SFT_SOURCE_DIR,
    *,
    limit_per_pattern: int = -1,
) -> list[PatternRecord]:
    source_dir = Path(source_dir).expanduser()
    records: list[PatternRecord] = []
    for pattern_type, filename in PATTERN_FILENAMES.items():
        records.extend(
            load_pattern_file(
                source_dir / filename,
                pattern_type=pattern_type,
                require_response=True,
                limit=limit_per_pattern,
            )
        )
    return records


def load_rl_patterns(
    rl_dir: str | Path = RL_DIR,
    *,
    limit_per_pattern: int = -1,
) -> list[PatternRecord]:
    rl_dir = Path(rl_dir).expanduser()
    records: list[PatternRecord] = []
    for pattern_type, filename in PATTERN_FILENAMES.items():
        records.extend(
            load_pattern_file(
                rl_dir / filename,
                pattern_type=pattern_type,
                require_response=False,
                limit=limit_per_pattern,
            )
        )
    return records


def split_records(
    records: list[dict[str, Any]],
    *,
    test_size: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not records:
        return [], []
    if test_size <= 0 or len(records) == 1:
        return records, []
    shuffled = records[:]
    random.Random(seed).shuffle(shuffled)
    if test_size < 1:
        n_test = max(1, int(round(len(shuffled) * test_size)))
    else:
        n_test = min(int(test_size), len(shuffled) - 1)
    return shuffled[n_test:], shuffled[:n_test]


def sft_row(record: PatternRecord, *, system_prompt: str) -> dict[str, Any]:
    if record.response is None:
        raise ValueError("SFT rows require a response")
    return {
        "id": f"{record.pattern_type}-{record.source_idx}",
        "pattern_type": record.pattern_type,
        "ability": record.ability,
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": record.query},
        ],
        "response": record.response,
        "question": record.query,
        "label": record.response,
        "extra_info": record.extra,
    }


def rl_row(record: PatternRecord, *, system_prompt: str | None = None) -> dict[str, Any]:
    expected_behavior = "refuse" if record.pattern_type == "harmful" else "answer"
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": record.query})
    return {
        "data_source": f"reflector_{record.pattern_type}",
        "pattern_type": record.pattern_type,
        "prompt": messages,
        "ability": record.ability,
        "reward_model": {
            "style": "rule",
            "ground_truth": "refuse" if expected_behavior == "refuse" else (record.response or ""),
        },
        "extra_info": {
            **record.extra,
            "question": record.query,
            "label": record.response,
            "expected_behavior": expected_behavior,
            "prompt_type": record.pattern_type,
        },
    }
