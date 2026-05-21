from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI

from ..paths import MODEL_ANSWER_DIR


JUDGE_PROMPT = """You are a data evaluation expert. Compare the MODEL ANSWER with the STANDARD ANSWER by following these strict rules:

RULES:
1. Extract the final numerical answer from the model response.
2. Remove all non-digit characters except decimal points, minus signs, and fraction symbols.
3. Ignore formatting differences such as dollar signs, commas, or spaces in numbers.
4. Compare only the final numerical values.

Model response:
{model_answer}

Standard answer:
{label}

If the numerical values match, output "yes". Otherwise, output "no".
Output strictly only "yes" or "no"."""


def extract_answer_from_label(label_text: str) -> str:
    parts = str(label_text).split("####")
    text = parts[-1] if parts else str(label_text)
    match = re.search(r"-?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+(?:\.\d+)?)?", text)
    return match.group(0).replace(",", "") if match else text.strip()


def judge_item(client: OpenAI, item: dict[str, Any], model: str, timeout: float) -> str:
    model_answer = str(item.get("answer", ""))
    label = extract_answer_from_label(str(item.get("label", "")))
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a data evaluation expert. Follow instructions exactly."},
            {
                "role": "user",
                "content": JUDGE_PROMPT.format(model_answer=model_answer.split("\n\n")[-1], label=label),
            },
        ],
        max_tokens=10,
        temperature=0,
        timeout=timeout,
    )
    evaluation = response.choices[0].message.content.strip().lower()
    return evaluation if evaluation in {"yes", "no"} else "no"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Use an OpenAI-compatible judge for GSM8K answer consistency.")
    parser.add_argument("--input", type=Path, default=MODEL_ANSWER_DIR / "gsm8k.json")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--stats_output", type=Path, default=None)
    parser.add_argument("--model", default=os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o"))
    parser.add_argument("--timeout", type=float, default=20.0)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_path = args.input.expanduser()
    output_path = args.output or input_path.with_name(f"{input_path.stem}_evaluated{input_path.suffix}")
    stats_path = args.stats_output or input_path.with_name(f"{input_path.stem}_stats.json")

    data_list = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(data_list, list):
        raise ValueError(f"Expected a JSON list in {input_path}")

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=os.getenv("OPENAI_BASE_URL"))
    yes_count = 0
    total_count = 0
    failed_items: list[int] = []

    for idx, item in enumerate(data_list):
        if not isinstance(item, dict):
            failed_items.append(idx + 1)
            continue
        if not item.get("answer") or not item.get("label"):
            failed_items.append(idx + 1)
            item["evaluation"] = "error"
            continue
        try:
            evaluation = judge_item(client, item, args.model, args.timeout)
        except Exception:
            failed_items.append(idx + 1)
            item["evaluation"] = "error"
            continue
        item["evaluation"] = evaluation
        yes_count += int(evaluation == "yes")
        total_count += 1
        print(f"{idx + 1}/{len(data_list)}: {evaluation}")

    stats = {
        "total_data": len(data_list),
        "successfully_evaluated": total_count,
        "failed_evaluations": len(failed_items),
        "failed_indices": failed_items,
        "consistent_answers": yes_count,
        "consistency_rate": yes_count / total_count if total_count else 0,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data_list, indent=2, ensure_ascii=False), encoding="utf-8")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(f"Wrote {stats_path}")


if __name__ == "__main__":
    main()
