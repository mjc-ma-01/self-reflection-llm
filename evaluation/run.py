from __future__ import annotations

import argparse
from pathlib import Path

from evaluation.generate import DEFAULT_BENCHMARKS, generate_answers
from evaluation.score import score_answer_files
from utils.paths import RESULTS_DIR


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full generation and scoring evaluation.")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_BENCHMARKS)
    parser.add_argument("--output_dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--num_samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--load_in_4bit", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    answer_paths = generate_answers(
        model_path=args.model_path,
        datasets=args.datasets,
        output_dir=args.output_dir / "answers",
        num_samples=args.num_samples,
        seed=args.seed,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        load_in_4bit=args.load_in_4bit,
    )
    paths = score_answer_files(answer_paths, output_dir=args.output_dir)
    for kind, path in paths.items():
        print(f"{kind}: {path}")


if __name__ == "__main__":
    main()
