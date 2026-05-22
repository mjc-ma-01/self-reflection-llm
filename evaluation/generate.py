from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from evaluation.benchmarks import load_benchmark
from utils.paths import RESULTS_DIR, set_hf_cache_env

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_BENCHMARKS = ["strongreject", "xstest", "wildchat", "donot", "gsm8k", "general"]
DEFAULT_SYSTEM_PROMPT = "You are a helpful and harmless assistant."


def generate_answers(
    *,
    model_path: str,
    datasets: list[str],
    output_dir: Path = RESULTS_DIR / "answers",
    num_samples: int = 200,
    seed: int = 42,
    batch_size: int = 4,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    load_in_4bit: bool = False,
) -> list[Path]:
    set_hf_cache_env()
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    quantization_config = None
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        cache_dir=os.environ.get("HUGGINGFACE_HUB_CACHE"),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir=os.environ.get("HUGGINGFACE_HUB_CACHE"), padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    written: list[Path] = []
    for dataset_name in datasets:
        rows = load_benchmark(dataset_name, num_samples=num_samples, seed=seed)
        results: list[dict] = []
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": row["question"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for row in batch
            ]
            encoded = tokenizer(prompts, padding=True, return_tensors="pt")
            encoded = {key: value.to(model.device) for key, value in encoded.items()}
            with torch.no_grad():
                output_ids = model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=temperature > 0,
                    temperature=temperature if temperature > 0 else None,
                    pad_token_id=tokenizer.pad_token_id,
                )
            answers = tokenizer.batch_decode(output_ids[:, encoded["input_ids"].shape[1] :], skip_special_tokens=True)
            for row, answer in zip(batch, answers):
                results.append({**row, "answer": answer, "model": model_path})
            print(f"[evaluation] {dataset_name}: generated {min(start + batch_size, len(rows))}/{len(rows)}")
        path = output_dir / f"{Path(model_path).name}_{dataset_name}.json"
        path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return written


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate answers for multiple evaluation datasets.")
    parser.add_argument("--model_path", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL))
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_BENCHMARKS)
    parser.add_argument("--output_dir", type=Path, default=RESULTS_DIR / "answers")
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--load_in_4bit", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    paths = generate_answers(**vars(args))
    for path in paths:
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
