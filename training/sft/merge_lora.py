from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.paths import set_hf_cache_env


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Merge a LoRA SFT adapter into its base model.")
    parser.add_argument("--adapter_path", type=Path, default=Path("checkpoints/sft/checkpoint-best"))
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints/sft/merged"))
    parser.add_argument("--torch_dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    return parser


def main() -> None:
    set_hf_cache_env()
    args = build_arg_parser().parse_args()
    peft_config = PeftConfig.from_pretrained(args.adapter_path)
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.torch_dtype]
    base_model = AutoModelForCausalLM.from_pretrained(
        peft_config.base_model_name_or_path,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model = model.merge_and_unload()
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output_dir, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(args.output_dir)
    metadata = {
        "adapter_path": str(args.adapter_path),
        "base_model_name_or_path": peft_config.base_model_name_or_path,
        "output_dir": str(args.output_dir),
        "torch_dtype": args.torch_dtype,
    }
    (args.output_dir / "merge_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
