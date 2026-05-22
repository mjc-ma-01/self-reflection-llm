from __future__ import annotations

from typing import Callable

from utils.paths import RL_DIR
from utils.patterns import load_pattern_file

BENCHMARK_DISPLAY_NAMES = {
    "strongreject": "StrongREJECT",
    "xstest": "XSTest",
    "wildchat": "ToxicityPrompts",
    "donot": "Do-Not-Answer",
    "gsm8k": "GSM8K",
    "general": "General Benchmark",
}


def _select(ds, num_samples: int, seed: int):
    if num_samples <= 0 or len(ds) <= num_samples:
        return ds
    return ds.shuffle(seed=seed).select(range(num_samples))


def _record(dataset: str, question: str, label: str = "", prompt_type: str = "general", expected_behavior: str = "answer", **extra):
    return {
        "dataset": dataset,
        "question": question,
        "label": label,
        "prompt_type": prompt_type,
        "expected_behavior": expected_behavior,
        "extra_info": extra,
    }


def load_xstest(num_samples: int, seed: int) -> list[dict]:
    import datasets as hf_datasets

    ds = _select(hf_datasets.load_dataset("walledai/XSTest", split="test"), num_samples, seed)
    return [
        _record(
            "xstest",
            row["prompt"],
            row["label"],
            "harmful" if row["label"] == "unsafe" else "general",
            "refuse" if row["label"] == "unsafe" else "answer",
        )
        for row in ds
    ]


def load_strongreject(num_samples: int, seed: int) -> list[dict]:
    import datasets as hf_datasets

    ds = _select(hf_datasets.load_dataset("walledai/StrongREJECT", split="train"), num_samples, seed)
    return [_record("strongreject", row["prompt"], row.get("category", ""), "harmful", "refuse") for row in ds]


def load_wildchat(num_samples: int, seed: int) -> list[dict]:
    import datasets as hf_datasets

    ds = hf_datasets.load_dataset("ToxicityPrompts/PolygloToxicityPrompts", "ptp-en", split="small")
    ds = _select(ds.filter(lambda row: float(row.get("toxicity") or 0.0) >= 0.5), num_samples, seed)
    return [_record("wildchat", row["prompt"], str(row.get("toxicity", "")), "harmful", "refuse") for row in ds]


def load_donot(num_samples: int, seed: int) -> list[dict]:
    import datasets as hf_datasets

    ds = _select(hf_datasets.load_dataset("LibrAI/do-not-answer", split="train"), num_samples, seed)
    return [_record("donot", row["question"], str(row.get("types_of_harm", "")), "harmful", "refuse") for row in ds]


def load_gsm8k(num_samples: int, seed: int) -> list[dict]:
    import datasets as hf_datasets

    ds = _select(hf_datasets.load_dataset("openai/gsm8k", "main", split="test"), num_samples, seed)
    rows = []
    for row in ds:
        label = row["answer"].split("####")[-1].strip() if "####" in row["answer"] else row["answer"].strip()
        question = (
            "Please solve this math problem step by step. Put your final answer in the format: "
            "Answer: \\boxed{your answer here}\n\n"
            + row["question"]
        )
        rows.append(_record("gsm8k", question, label, "math", "answer", raw_question=row["question"]))
    return rows


def load_simpleqa(num_samples: int, seed: int) -> list[dict]:
    import datasets as hf_datasets

    ds = _select(hf_datasets.load_dataset("basicv8vc/SimpleQA", split="test"), num_samples, seed)
    return [_record("simpleqa", row["problem"], row["answer"], "general", "answer") for row in ds]


def load_general_eval(num_samples: int, seed: int) -> list[dict]:
    general = load_pattern_file(RL_DIR / "general_pattern.json", pattern_type="general", require_response=False, limit=num_samples)
    return [
        _record("general", item.query, item.response or "", "general", "answer", source_idx=item.source_idx)
        for item in general
    ]


LOADERS: dict[str, Callable[[int, int], list[dict]]] = {
    "xstest": load_xstest,
    "strongreject": load_strongreject,
    "wildchat": load_wildchat,
    "donot": load_donot,
    "gsm8k": load_gsm8k,
    "general": load_general_eval,
}


def load_benchmark(name: str, *, num_samples: int, seed: int) -> list[dict]:
    if name not in LOADERS:
        raise ValueError(f"Unsupported benchmark {name!r}. Available: {sorted(LOADERS)}")
    return LOADERS[name](num_samples, seed)


def benchmark_display_name(name: str) -> str:
    return BENCHMARK_DISPLAY_NAMES.get(name, name)
