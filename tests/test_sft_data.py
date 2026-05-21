from pathlib import Path

from self_reflection_llm.datasets.sft_data import load_sft_dataset, marker_counts, read_jsonl


def test_ready_sft_data_loads():
    dataset = load_sft_dataset("data/sft/ready/train.jsonl", "data/sft/ready/test.jsonl")

    assert len(dataset["train"]) > 0
    assert len(dataset["test"]) > 0
    row = dataset["train"][0]
    assert row["prompt"][-1]["role"] == "user"
    assert row["response"] == row["label"]
    assert row["question"] == row["prompt"][-1]["content"]
    assert row["extra_info"]["stage"] == "sft"


def test_ready_sft_data_contains_reflection_markers():
    rows = read_jsonl(Path("data/sft/ready/train.jsonl"))
    counts = marker_counts(rows)

    assert counts["<|continue|>"] > 0
    assert counts["<|reflect|>"] > 0
