from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.utils import is_flash_attn_2_available

from utils.paths import CHECKPOINTS_DIR, SFT_DIR, set_hf_cache_env
from utils.patterns import read_jsonl
from utils.resources import choose_sft_plan, detect_resources

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_SYSTEM_PROMPT = "You are a helpful and harmless assistant."


def _load_dataset(train_file: str | Path, eval_file: str | Path) -> DatasetDict:
    train_rows = read_jsonl(train_file)
    eval_rows = read_jsonl(eval_file)
    if not train_rows:
        raise ValueError(f"No SFT training rows found in {train_file}")
    if not eval_rows:
        raise ValueError(f"No SFT eval rows found in {eval_file}")
    return DatasetDict({"train": Dataset.from_list(train_rows), "test": Dataset.from_list(eval_rows)})


def _extract_question_response(row: dict[str, Any]) -> tuple[str, str]:
    question = row.get("question")
    response = row.get("response") or row.get("label")
    prompt = row.get("prompt")
    if not question and isinstance(prompt, list):
        for message in reversed(prompt):
            if isinstance(message, dict) and message.get("role") == "user":
                question = message.get("content")
                break
    if not question or not response:
        raise ValueError(f"SFT row must contain question/prompt and response/label: {sorted(row)}")
    return str(question), str(response)


def _maybe_peft_model(model, *, use_peft: bool, load_in_4bit: bool):
    if not use_peft:
        return model
    try:
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    except ImportError as exc:
        raise RuntimeError("PEFT mode requires `peft`. Install requirements/requirements.txt.") from exc
    if load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
        r=64,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    return get_peft_model(model, config)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train Llama 3 8B SFT on harmful/general pattern data.")
    parser.add_argument("--model_path", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL))
    parser.add_argument("--train_file", type=Path, default=SFT_DIR / "train.jsonl")
    parser.add_argument("--eval_file", type=Path, default=SFT_DIR / "test.jsonl")
    parser.add_argument("--output_dir", type=Path, default=CHECKPOINTS_DIR / "sft")
    parser.add_argument("--system_prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--auto_config", action="store_true", default=True)
    parser.add_argument("--no_auto_config", action="store_false", dest="auto_config")
    parser.add_argument("--use_peft", action="store_true", default=None)
    parser.add_argument("--load_in_4bit", action="store_true", default=None)
    parser.add_argument("--num_train_epochs", type=float, default=None)
    parser.add_argument("--per_device_train_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--deepspeed", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    set_hf_cache_env()
    args = build_arg_parser().parse_args()
    set_seed(args.seed)

    resources = detect_resources()
    plan = choose_sft_plan(resources) if args.auto_config else None
    use_peft = args.use_peft if args.use_peft is not None else (plan.use_peft if plan else False)
    load_in_4bit = args.load_in_4bit if args.load_in_4bit is not None else (plan.load_in_4bit if plan else False)
    precision = plan.precision if plan else ("bf16" if resources.bf16 else "fp16")
    deepspeed = args.deepspeed if args.deepspeed is not None else (plan.deepspeed_config if plan else None)
    num_train_epochs = args.num_train_epochs if args.num_train_epochs is not None else (plan.num_train_epochs if plan else 3.0)
    train_bs = args.per_device_train_batch_size if args.per_device_train_batch_size is not None else (plan.per_device_train_batch_size if plan else 1)
    grad_accum = args.gradient_accumulation_steps if args.gradient_accumulation_steps is not None else (plan.gradient_accumulation_steps if plan else 4)
    learning_rate = args.learning_rate if args.learning_rate is not None else (plan.learning_rate if plan else 5e-6)

    quantization_config = None
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("4-bit loading requires transformers BitsAndBytesConfig and bitsandbytes.") from exc
        if not torch.cuda.is_available():
            raise RuntimeError("4-bit training requires CUDA.")
        quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    torch_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    attn_implementation = "flash_attention_2" if torch.cuda.is_available() and is_flash_attn_2_available() else None
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        cache_dir=os.environ.get("HUGGINGFACE_HUB_CACHE"),
        torch_dtype=torch_dtype if torch.cuda.is_available() else torch.float32,
        quantization_config=quantization_config,
        device_map="auto" if torch.cuda.is_available() and load_in_4bit and not deepspeed else None,
        attn_implementation=attn_implementation,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, cache_dir=os.environ.get("HUGGINGFACE_HUB_CACHE"), padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False
    model = _maybe_peft_model(model, use_peft=use_peft, load_in_4bit=load_in_4bit)
    if use_peft and hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    dataset = _load_dataset(args.train_file, args.eval_file)

    def tokenize(row: dict[str, Any]) -> dict[str, list[int]]:
        question, response = _extract_question_response(row)
        messages = [
            {"role": "system", "content": args.system_prompt},
            {"role": "user", "content": question},
        ]
        prompt_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        messages.append({"role": "assistant", "content": response})
        full_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True)
        if len(full_ids) > args.max_length:
            full_ids = full_ids[-args.max_length :]
            prompt_ids = prompt_ids[-min(len(prompt_ids), args.max_length) :]
        labels = full_ids.copy()
        prompt_len = min(len(prompt_ids), len(labels))
        labels[:prompt_len] = [-100] * prompt_len
        return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}

    dataset = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        overwrite_output_dir=True,
        run_name="llama3_8b_reflector_sft",
        seed=args.seed,
        per_device_train_batch_size=train_bs,
        gradient_accumulation_steps=grad_accum,
        per_device_eval_batch_size=1,
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.05,
        bf16=(precision == "bf16"),
        fp16=(precision == "fp16"),
        gradient_checkpointing=True,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=0.25,
        save_strategy="steps",
        save_steps=0.25,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        deepspeed=deepspeed,
        remove_unused_columns=False,
        group_by_length=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "resource_plan.json").write_text(
        json.dumps(
            {
                "resources": resources.__dict__,
                "plan": plan.__dict__ if plan else None,
                "effective_batch_size": (plan.effective_batch_size if plan else train_bs * grad_accum),
                "model_path": args.model_path,
                "hf_cache": os.environ.get("HUGGINGFACE_HUB_CACHE"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, pad_to_multiple_of=8, return_tensors="pt"),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "checkpoint-best"))
    tokenizer.save_pretrained(str(args.output_dir / "checkpoint-best"))


if __name__ == "__main__":
    main()
