import json
import os
from dataclasses import dataclass
from typing import Literal, Optional, Union

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, HfArgumentParser

from .data import Adv_GLUE, DoNotAnswer, GSM8k, SimpleQA, StrongReject, WildChat, XSTest
from .paths import MODEL_ANSWER_DIR


def extract_train_name_from_path(model_path: str) -> str:
    parts = model_path.split("train:")
    if len(parts) > 1:
        return parts[1].split("/")[0]
    return ""


def build_eval_dataset(eval_ds: str):
    if eval_ds == "xstest":
        return XSTest(num_samples=100).get_dataset()
    if eval_ds == "strongreject":
        return StrongReject(num_samples=100).get_dataset()
    if eval_ds == "donot":
        return DoNotAnswer(num_samples=200).get_dataset()
    if eval_ds == "wildchat":
        return WildChat(num_samples=200).get_dataset()
    if eval_ds == "simpleqa":
        return SimpleQA(num_samples=200).get_dataset()
    if eval_ds == "gsm8k":
        return GSM8k(num_samples=500).get_dataset()
    if eval_ds == "advglue":
        return Adv_GLUE().get_dataset()
    raise ValueError(f"Unsupported eval dataset: {eval_ds}")


@dataclass
class ScriptArgument:
    model_path: str = (
        "/mnt/shared-storage-user/majiachen/models/reflect_results/"
        "model:sft_mllm_llama_8b/train:reflect_cot_1k_w400benign_400harm_300math_"
        "lr5e-6_epoch5_warmup0.05/checkpoint-best"
    )
    eval_ds: Union[
        Literal["xstest", "strongreject", "wildchat", "donot", "simpleqa", "gsm8k", "advglue"],
        str,
    ] = "wildchat"
    save_path: Optional[str] = None
    seed: int = 42
    load_in_4bit: bool = False
    use_flash_attention_2: bool = True
    batch_size: int = 4


def main() -> None:
    parser = HfArgumentParser(ScriptArgument)
    args = parser.parse_args_into_dataclasses()[0]

    transformers.set_seed(args.seed)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        quantization_config=(
            BitsAndBytesConfig(load_in_4bit=True)
            if args.load_in_4bit and transformers.utils.is_bitsandbytes_available()
            else None
        ),
        attn_implementation=(
            "flash_attention_2"
            if args.use_flash_attention_2 and transformers.utils.is_flash_attn_2_available()
            else None
        ),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left")
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token

    eval_dataset = build_eval_dataset(args.eval_ds)

    def llm_collator(row, system_prompt: str = "You are a helpful and harmless assistant.") -> dict:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["question"]},
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return {"prompt": prompt}

    eval_dataset = eval_dataset.map(llm_collator)

    results = []
    for i in range(0, len(eval_dataset), args.batch_size):
        batch = eval_dataset.select(range(i, min(i + args.batch_size, len(eval_dataset))))
        prompts = batch["prompt"]
        inputs = tokenizer(prompts, padding=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda")
        attention_mask = inputs["attention_mask"].to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=2048,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
            )

        generated_texts = tokenizer.batch_decode(
            generated_ids[:, input_ids.shape[-1] :], skip_special_tokens=True
        )
        for ex, answer in zip(batch, generated_texts):
            results.append(
                {
                    "question": ex["question"],
                    "answer": answer,
                    "label": ex.get("label", None),
                }
            )
        print(f"num_{i}_examples")
        print(results[i])

    if not args.save_path:
        args.save_path = str(MODEL_ANSWER_DIR / f"{args.eval_ds}.json")

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    with open(args.save_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print(f"Results saved to {args.save_path}")


if __name__ == "__main__":
    main()
