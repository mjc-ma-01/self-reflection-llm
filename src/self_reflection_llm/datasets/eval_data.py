"""Prepare validation datasets in the same parquet schema used by verl."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

from self_reflection_llm.paths import PROCESSED_DATA_DIR

DEFAULT_EVAL_DATASETS = ("xstest", "strongreject", "wildchat", "donot", "gsm8k", "simpleqa")


def _require_datasets():
    try:
        import datasets
    except ImportError as exc:
        raise RuntimeError("Install datasets and pyarrow to prepare validation parquet files") from exc
    return datasets


def _select(ds, num_samples: int, seed: int):
    if num_samples <= 0 or len(ds) <= num_samples:
        return ds
    return ds.shuffle(seed=seed).select(range(num_samples))


def _row(
    *,
    data_source: str,
    question: str,
    ability: str,
    prompt_type: str,
    ground_truth: str,
    expected_behavior: str = "answer",
    extra: dict | None = None,
) -> dict:
    info = {
        "prompt_type": prompt_type,
        "question": question,
        "expected_behavior": expected_behavior,
        "eval_method": "rm" if prompt_type != "math" else "local_or_rm",
    }
    if extra:
        info.update(extra)
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": question}],
        "images": None,
        "ability": ability,
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": info,
    }


def prepare_xstest(num_samples: int, seed: int):
    datasets = _require_datasets()
    ds = _select(datasets.load_dataset("walledai/XSTest", split="test"), num_samples, seed)

    def process(row):
        is_unsafe = row["label"] == "unsafe"
        return _row(
            data_source="eval_xstest",
            question=row["prompt"],
            ability="safety",
            prompt_type="harmful",
            ground_truth="refuse" if is_unsafe else "answer",
            expected_behavior="refuse" if is_unsafe else "answer",
            extra={"label": row["label"]},
        )

    return ds.map(process, remove_columns=ds.column_names)


def prepare_strongreject(num_samples: int, seed: int):
    datasets = _require_datasets()
    ds = _select(datasets.load_dataset("walledai/StrongREJECT", split="train"), num_samples, seed)

    def process(row):
        return _row(
            data_source="eval_strongreject",
            question=row["prompt"],
            ability="safety",
            prompt_type="harmful",
            ground_truth="refuse",
            expected_behavior="refuse",
            extra={"category": row.get("category", "")},
        )

    return ds.map(process, remove_columns=ds.column_names)


def prepare_wildchat(num_samples: int, seed: int):
    datasets = _require_datasets()
    ds = datasets.load_dataset("ToxicityPrompts/PolygloToxicityPrompts", "wildchat-en", split="wildchat")
    ds = _select(ds.filter(lambda row: row["prompt_toxicity"] >= 0.5), num_samples, seed)

    def process(row):
        return _row(
            data_source="eval_wildchat",
            question=row["prompt"],
            ability="safety",
            prompt_type="harmful",
            ground_truth="refuse",
            expected_behavior="refuse",
            extra={"toxicity": row.get("prompt_toxicity")},
        )

    return ds.map(process, remove_columns=ds.column_names)


def prepare_donot(num_samples: int, seed: int):
    datasets = _require_datasets()
    ds = _select(datasets.load_dataset("LibrAI/do-not-answer", split="train"), num_samples, seed)

    def process(row):
        return _row(
            data_source="eval_donot",
            question=row["question"],
            ability="safety",
            prompt_type="harmful",
            ground_truth="refuse",
            expected_behavior="refuse",
            extra={"harm_type": row.get("types_of_harm", "")},
        )

    return ds.map(process, remove_columns=ds.column_names)


def prepare_gsm8k(num_samples: int, seed: int):
    datasets = _require_datasets()
    ds = _select(datasets.load_dataset("openai/gsm8k", "main", split="test"), num_samples, seed)

    def process(row):
        answer = row["answer"].split("####")[-1].strip() if "####" in row["answer"] else row["answer"].strip()
        question = (
            "Please solve this math problem step by step. Put your final answer in the format: "
            "Answer: \\boxed{your answer here}\n\n"
            + row["question"]
        )
        return _row(
            data_source="eval_gsm8k",
            question=question,
            ability="math",
            prompt_type="math",
            ground_truth=answer,
            expected_behavior="answer",
            extra={"raw_question": row["question"]},
        )

    return ds.map(process, remove_columns=ds.column_names)


def prepare_simpleqa(num_samples: int, seed: int):
    datasets = _require_datasets()
    ds = _select(datasets.load_dataset("basicv8vc/SimpleQA", split="test"), num_samples, seed)

    def process(row):
        return _row(
            data_source="eval_simpleqa",
            question=row["problem"],
            ability="general",
            prompt_type="general",
            ground_truth=row["answer"],
            expected_behavior="answer",
        )

    return ds.map(process, remove_columns=ds.column_names)


PREPARE_FNS: dict[str, Callable[[int, int], object]] = {
    "xstest": prepare_xstest,
    "strongreject": prepare_strongreject,
    "wildchat": prepare_wildchat,
    "donot": prepare_donot,
    "gsm8k": prepare_gsm8k,
    "simpleqa": prepare_simpleqa,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_EVAL_DATASETS), choices=sorted(PREPARE_FNS))
    parser.add_argument("--output_dir", type=Path, default=PROCESSED_DATA_DIR / "reflector_eval")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir.expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.datasets:
        ds = PREPARE_FNS[name](args.num_samples, args.seed)
        out_path = output_dir / f"{name}.parquet"
        ds.to_parquet(str(out_path))
        print(f"Wrote {out_path} ({len(ds)} rows)")


if __name__ == "__main__":
    main()
