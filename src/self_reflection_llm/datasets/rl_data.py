"""Prepare Reflector RL data in verl parquet format."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Optional

from self_reflection_llm.paths import PROCESSED_DATA_DIR, PROJECT_ROOT, RL_DATA_DIR


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON list in {path}")
    return [item for item in data if isinstance(item, dict)]


def _read_system_prompt(system_prompt: Optional[str], system_prompt_file: Optional[str]) -> Optional[str]:
    if system_prompt_file:
        return Path(system_prompt_file).expanduser().read_text(encoding="utf-8").strip()
    return system_prompt.strip() if system_prompt else None


def _extract_query(item: dict[str, Any]) -> tuple[str, str]:
    for key in ("query", "question", "prompt", "instruction"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return key, value.strip()
    raise ValueError(f"missing query field; available keys: {sorted(item.keys())}")


def _repo_display_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _messages(query: str, system_prompt: Optional[str]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})
    return messages


def _build_records(
    raw_items: list[dict[str, Any]],
    *,
    prompt_type: str,
    data_source: str,
    ability: str,
    expected_behavior: str,
    source_file: Path,
    system_prompt: Optional[str],
    limit: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, item in enumerate(raw_items):
        if limit >= 0 and idx >= limit:
            break
        query_key, query = _extract_query(item)
        records.append(
            {
                "data_source": data_source,
                "prompt": _messages(query, system_prompt),
                "ability": ability,
                "reward_model": {
                    "style": "rule",
                    "ground_truth": "refuse" if expected_behavior == "refuse" else "",
                },
                "extra_info": {
                    "prompt_type": prompt_type,
                    "expected_behavior": expected_behavior,
                    "question": query,
                    "label": item.get("label"),
                    "source_file": _repo_display_path(source_file),
                    "source_idx": idx,
                    "input_field": query_key,
                },
            }
        )
    return records


def _split(records: list[dict[str, Any]], test_size: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if test_size <= 0 or len(records) <= 1:
        return records, []
    rng = random.Random(seed)
    shuffled = records[:]
    rng.shuffle(shuffled)
    if test_size < 1:
        n_test = max(1, int(round(len(shuffled) * test_size)))
    else:
        n_test = min(int(test_size), len(shuffled) - 1)
    return shuffled[n_test:], shuffled[:n_test]


def _write_parquet(records: list[dict[str, Any]], path: Path) -> None:
    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError("Install datasets and pyarrow to write verl parquet files") from exc
    path.parent.mkdir(parents=True, exist_ok=True)
    datasets.Dataset.from_list(records).to_parquet(str(path))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--general_path", type=Path, default=RL_DATA_DIR / "general_pattern.json")
    parser.add_argument("--harmful_path", type=Path, default=RL_DATA_DIR / "harmful_pattern.json")
    parser.add_argument("--output_dir", type=Path, default=PROCESSED_DATA_DIR / "reflector_rl")
    parser.add_argument("--test_size", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_per_file", type=int, default=-1)
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--system_prompt_file", type=str, default=None)
    parser.add_argument("--general_data_source", type=str, default="reflector_general")
    parser.add_argument("--harmful_data_source", type=str, default="reflector_harmful")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    system_prompt = _read_system_prompt(args.system_prompt, args.system_prompt_file)
    general_path = args.general_path.expanduser()
    harmful_path = args.harmful_path.expanduser()

    general_records = _build_records(
        _load_json_list(general_path),
        prompt_type="general",
        data_source=args.general_data_source,
        ability="general",
        expected_behavior="answer",
        source_file=general_path,
        system_prompt=system_prompt,
        limit=args.limit_per_file,
    )
    harmful_records = _build_records(
        _load_json_list(harmful_path),
        prompt_type="harmful",
        data_source=args.harmful_data_source,
        ability="safety",
        expected_behavior="refuse",
        source_file=harmful_path,
        system_prompt=system_prompt,
        limit=args.limit_per_file,
    )

    general_train, general_test = _split(general_records, args.test_size, args.seed)
    harmful_train, harmful_test = _split(harmful_records, args.test_size, args.seed + 1)
    train_records = general_train + harmful_train
    test_records = general_test + harmful_test

    output_dir = args.output_dir.expanduser()
    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"
    _write_parquet(train_records, train_path)
    _write_parquet(test_records, test_path)

    metadata = {
        "general_path": _repo_display_path(general_path),
        "harmful_path": _repo_display_path(harmful_path),
        "train_records": len(train_records),
        "test_records": len(test_records),
        "general_counts": {"train": len(general_train), "test": len(general_test)},
        "harmful_counts": {"train": len(harmful_train), "test": len(harmful_test)},
        "system_prompt_included": bool(system_prompt),
        "args": {key: _repo_display_path(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {train_path}")
    print(f"Wrote {test_path}")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
