"""Build and load ready-to-train Reflector SFT JSONL files.

The main training path consumes ``data/sft/ready/train.jsonl`` and
``data/sft/ready/test.jsonl`` directly. This module is the optional
reproduction step that rebuilds those files from released local sources.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterable, Optional

from self_reflection_llm.paths import DATA_SRC_DIR, DRA_PROCESSED_DIR, PROJECT_ROOT, SFT_READY_DIR

MARKERS = ("<|reflect|>", "<|explore|>", "<|continue|>")
SYSTEM_PROMPT = "You are a helpful and harmless assistant."

REFLECTIVE_SOURCES = (
    ("rellm_code", DRA_PROCESSED_DIR / "GPT_ReLLM(code)310_filtered_attack_with_reflections.jsonl", "safety"),
    ("rellm_paragraph", DRA_PROCESSED_DIR / "GPT_ReLLM(paragraph)470_filtered_attack_with_reflections.jsonl", "safety"),
    ("rellm_table", DRA_PROCESSED_DIR / "GPT_ReLLM(table)440_filtered_attack_with_reflections.jsonl", "safety"),
    ("dra_harmful", DRA_PROCESSED_DIR / "LLaMA70b_DRA_filtered_attack_with_reflections.jsonl", "safety"),
    ("dra_benign", DRA_PROCESSED_DIR / "LLaMA70b_DRA_filtered_attack_with_reflections_benign.jsonl", "general"),
    ("dra_safe_imitation", DATA_SRC_DIR / "DRA_safe_imitation_data_1k_complete.json", "safety"),
    ("drattack_safe_imitation", DATA_SRC_DIR / "DrAttack_safe_imitation_data_500_complete.json", "safety"),
    ("drattack_benign_imitation", DATA_SRC_DIR / "DrAttack(benign)_safe_imitation_data_300_complete.json", "general"),
    ("dra_benign_imitation", DATA_SRC_DIR / "DRA(benign)_safe_imitation_data_200_complete.json", "general"),
    ("gpt_wrong_correct", DATA_SRC_DIR / "GPT_reflect" / "wrong-correct.json", "safety"),
    ("gpt_correct_correct", DATA_SRC_DIR / "GPT_reflect" / "correct-correct.json", "general"),
    ("harm_refuse", DATA_SRC_DIR / "harm_refuse_gptgenerate.json", "safety"),
    ("harm_wrong_refuse", DATA_SRC_DIR / "harm_wrong_refuse_gptgenerate.json", "safety"),
)


def _repo_display_path(path: Path) -> str:
    if str(path).startswith("hf:"):
        return str(path)
    resolved = path.expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"expected a JSON list in {path}")
    return [item for item in data if isinstance(item, dict)]


def _first_text(item: dict[str, Any], keys: Iterable[str]) -> Optional[str]:
    for key in keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _messages(question: str, system_prompt: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


def _record(
    *,
    record_id: str,
    question: str,
    response: str,
    data_source: str,
    ability: str,
    source_file: Path,
    source_idx: int,
    system_prompt: str,
) -> dict[str, Any]:
    extra = {
        "stage": "sft",
        "question": question,
        "source_file": _repo_display_path(source_file),
        "source_idx": source_idx,
        "has_reflect": "<|reflect|>" in response,
        "has_explore": "<|explore|>" in response,
        "has_continue": "<|continue|>" in response,
    }
    return {
        "id": record_id,
        "data_source": data_source,
        "prompt": _messages(question, system_prompt),
        "response": response,
        "ability": ability,
        "extra_info": extra,
        # Backward-compatible aliases used by older SFT scripts.
        "question": question,
        "label": response,
    }


def _insert_continue_tokens(text: str) -> str:
    text = text.strip()
    if not text:
        return text
    sentences = text.split(". ", 1)
    if len(sentences) == 1:
        return f"{text}\n<|continue|>\n"
    first, rest = sentences
    chunks = []
    if rest:
        chunk_size = max(1, len(rest) // 4)
        for idx in range(3):
            chunks.append(rest[idx * chunk_size : (idx + 1) * chunk_size])
            chunks.append("<|continue|>")
        chunks.append(rest[3 * chunk_size :])
    return f"{first}.\n<|continue|>\n{''.join(chunks)}"


def build_local_sft_records(
    *,
    limit_per_file: int = 200,
    helpful_limit: int = 400,
    system_prompt: str = SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    """Build SFT records from released local sources only."""

    records: list[dict[str, Any]] = []

    for source_name, path, ability in REFLECTIVE_SOURCES:
        if not path.exists():
            raise FileNotFoundError(path)
        for source_idx, item in enumerate(_load_records(path)):
            if limit_per_file >= 0 and source_idx >= limit_per_file:
                break
            question = _first_text(item, ("query", "question", "prompt", "instruction", "text_query"))
            response = _first_text(item, ("reflect_answer", "response", "answer", "output"))
            if not question or not response:
                continue
            records.append(
                _record(
                    record_id=f"{source_name}-{source_idx}",
                    question=question,
                    response=response,
                    data_source=f"sft_{source_name}",
                    ability=ability,
                    source_file=path,
                    source_idx=source_idx,
                    system_prompt=system_prompt,
                )
            )

    alpaca_path = DATA_SRC_DIR / "alpaca_eval.json"
    if alpaca_path.exists() and helpful_limit != 0:
        for source_idx, item in enumerate(_load_records(alpaca_path)):
            if helpful_limit >= 0 and source_idx >= helpful_limit:
                break
            question = _first_text(item, ("instruction", "query", "question", "prompt"))
            output = _first_text(item, ("output", "answer", "response"))
            if not question or not output:
                continue
            records.append(
                _record(
                    record_id=f"alpaca_eval-{source_idx}",
                    question=question,
                    response=_insert_continue_tokens(output),
                    data_source="sft_alpaca_eval",
                    ability="general",
                    source_file=alpaca_path,
                    source_idx=source_idx,
                    system_prompt=system_prompt,
                )
            )

    return records


def build_hf_math_records(*, num_samples_per_type: int, system_prompt: str = SYSTEM_PROMPT) -> list[dict[str, Any]]:
    """Optionally add online math reflection data used in the original mixture."""

    if num_samples_per_type <= 0:
        return []
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets to add Hugging Face math records") from exc

    ds = load_dataset("Satori-reasoning/Satori_FT_data", split="train")
    target_types = ("type_III_I_data", "type_II_I_data", "type_II_II_data")
    records: list[dict[str, Any]] = []
    for target_type in target_types:
        selected = [row for row in ds if row.get("type") == target_type]
        for source_idx, item in enumerate(selected[:num_samples_per_type]):
            question = _first_text(item, ("query", "question", "prompt"))
            response = _first_text(item, ("response", "reflect_answer", "answer"))
            if not question or not response:
                continue
            records.append(
                _record(
                    record_id=f"satori_math-{target_type}-{source_idx}",
                    question=question,
                    response=response,
                    data_source="sft_satori_math",
                    ability="math",
                    source_file=Path("hf://Satori-reasoning/Satori_FT_data"),
                    source_idx=source_idx,
                    system_prompt=system_prompt,
                )
            )
    return records


def split_records(records: list[dict[str, Any]], test_size: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
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


def write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.expanduser().open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def marker_counts(records: list[dict[str, Any]]) -> dict[str, int]:
    return {marker: sum(marker in row.get("response", row.get("label", "")) for row in records) for marker in MARKERS}


def load_sft_dataset(train_file: str | Path, eval_file: str | Path):
    """Load ready SFT JSONL files as a Hugging Face DatasetDict."""

    try:
        from datasets import Dataset, DatasetDict
    except ImportError as exc:
        raise RuntimeError("Install datasets to load SFT JSONL files") from exc

    train_path = Path(train_file).expanduser()
    eval_path = Path(eval_file).expanduser()
    if not train_path.exists() or not eval_path.exists():
        missing = [str(path) for path in (train_path, eval_path) if not path.exists()]
        raise FileNotFoundError(
            "Missing ready SFT data: "
            + ", ".join(missing)
            + ". Run `bash scripts/sft/prepare_data.sh` to rebuild it."
        )
    return DatasetDict(
        {
            "train": Dataset.from_list(read_jsonl(train_path)),
            "test": Dataset.from_list(read_jsonl(eval_path)),
        }
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, default=SFT_READY_DIR)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_per_file", type=int, default=200)
    parser.add_argument("--helpful_limit", type=int, default=400)
    parser.add_argument("--system_prompt", type=str, default=SYSTEM_PROMPT)
    parser.add_argument("--include_hf_math", action="store_true")
    parser.add_argument("--hf_math_samples_per_type", type=int, default=200)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir.expanduser()

    records = build_local_sft_records(
        limit_per_file=args.limit_per_file,
        helpful_limit=args.helpful_limit,
        system_prompt=args.system_prompt,
    )
    if args.include_hf_math:
        records.extend(
            build_hf_math_records(
                num_samples_per_type=args.hf_math_samples_per_type,
                system_prompt=args.system_prompt,
            )
        )

    train_records, test_records = split_records(records, args.test_size, args.seed)
    train_path = output_dir / "train.jsonl"
    test_path = output_dir / "test.jsonl"
    metadata_path = output_dir / "metadata.json"
    write_jsonl(train_records, train_path)
    write_jsonl(test_records, test_path)

    metadata = {
        "train_records": len(train_records),
        "test_records": len(test_records),
        "total_records": len(records),
        "train_marker_counts": marker_counts(train_records),
        "test_marker_counts": marker_counts(test_records),
        "schema": {
            "data_source": "source bucket for the supervised example",
            "prompt": "chat messages used as the supervised prompt",
            "response": "assistant target text",
            "ability": "safety, general, or math",
            "extra_info": "source path, source index, and marker flags",
            "question": "backward-compatible user text alias",
            "label": "backward-compatible assistant target alias",
        },
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Wrote {train_path} ({len(train_records)} rows)")
    print(f"Wrote {test_path} ({len(test_records)} rows)")
    print(f"Wrote {metadata_path}")


if __name__ == "__main__":
    main()
