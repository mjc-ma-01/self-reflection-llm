from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upload a HuggingFace model card without storing credentials in the repo.")
    parser.add_argument("--repo_id", default="krystal7/llama-8b-reflect-sft")
    parser.add_argument("--card", type=Path, default=Path("model_cards/llama-8b-reflect-sft/README.md"))
    parser.add_argument("--revision", default="main")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if not args.card.exists():
        raise FileNotFoundError(args.card)
    token = os.getenv("HF_TOKEN")
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=str(args.card),
        path_in_repo="README.md",
        repo_id=args.repo_id,
        repo_type="model",
        revision=args.revision,
        commit_message="Update Reflector SFT model card",
    )
    print(f"Uploaded {args.card} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
