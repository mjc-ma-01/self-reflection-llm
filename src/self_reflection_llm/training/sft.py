import os
import json
from dataclasses import dataclass
from pathlib import Path
import torch
import transformers
from accelerate import PartialState
from datasets import Dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer

SYSTEM_PROMPT = "You are a helpful and harmless assistant."
DEFAULT_SFT_DIR = Path("data/sft")


def _read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(
            f"Missing SFT JSONL file: {path}. Generate reflection trajectories with "
            "`bash scripts/sft/prepare_data.sh` or pass --train_file/--eval_file explicitly."
        )
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_sft_dataset(train_file: str | Path, eval_file: str | Path) -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_list(_read_jsonl(train_file)),
            "test": Dataset.from_list(_read_jsonl(eval_file)),
        }
    )

def extract_train_name_from_path(model_path: str) -> str:
    parts = model_path.split("train:")
    if len(parts) > 1:
        # 获取train:后面的部分，再按"/"分割取第一部分
        after_train = parts[1]
        train_name = after_train.split("/")[0]
        return train_name
    return ""
    
@dataclass
class ModelArguments:
    model_path:       str = "meta-llama/Llama-3.1-8B-Instruct"
    load_in_4bit:     bool = False
    use_flash_attention_2: bool = True

@dataclass
class PeftArguments:
    use_peft:       bool  = True
    target_modules: str   = "all-linear"
    r:              int   = 64
    lora_alpha:     int   = 64
    lora_dropout:   float = 0.05
    bias:           str   = "none"
    task_type:      str   = "CAUSAL_LM"

@dataclass
class DataArguments:
    train_file: str = str(DEFAULT_SFT_DIR / "train.jsonl")
    eval_file: str = str(DEFAULT_SFT_DIR / "test.jsonl")
    system_prompt: str = SYSTEM_PROMPT

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = "outputs/reflect_models"
    run_name: str = "reflect_sft"
    report_to: str = "wandb"
    overwrite_output_dir: bool = True
    seed: int = 42
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    per_device_eval_batch_size: int = 4
    learning_rate: float = 1e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    weight_decay: float = 0.05
    bf16: bool = True
    num_train_epochs: float = 2.0
    logging_steps: float = 10
    save_steps: float = 0.25
    eval_steps: float = 0.25
    eval_strategy: str = "steps"
    save_only_model: bool = False
    load_best_model_at_end: bool = True


def train():
    parser = transformers.HfArgumentParser((ModelArguments, PeftArguments, DataArguments, TrainingArguments))
    model_args, peft_args, data_args, training_args = parser.parse_args_into_dataclasses()
        # **(
        #     # {"device_map": {"": PartialState().local_process_index}}
        #     "device_map"="auto"
        #     if not transformers.modeling_utils.is_deepspeed_zero3_enabled()
        #     else {}
        # ),
        #        device_map="auto",
    # loading model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_path,
        torch_dtype=torch.bfloat16,
        **(
            {"device_map": {"": PartialState().local_process_index}} if not transformers.modeling_utils.is_deepspeed_zero3_enabled()
            else {}
        ),
        quantization_config=(
            BitsAndBytesConfig(load_in_4bit=True)
            if model_args.load_in_4bit and transformers.utils.is_bitsandbytes_available()
            else None
        ),
        attn_implementation=(
            "flash_attention_2"
            if model_args.use_flash_attention_2 and transformers.utils.is_flash_attn_2_available()
            else None
        ),
    )    
    
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_path, padding_side="right")
    if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token
    # if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.unk_token
    
    dataset = load_sft_dataset(data_args.train_file, data_args.eval_file)

    # breakpoint()

    def _question_and_label(row: dict) -> tuple[str, str]:
        question = row.get("question")
        label = row.get("label") or row.get("response")
        if not question:
            prompt = row.get("prompt") or []
            for message in reversed(prompt):
                if isinstance(message, dict) and message.get("role") == "user":
                    question = message.get("content")
                    break
        if not question or not label:
            raise ValueError(f"SFT row must contain question/prompt and label/response: {row.keys()}")
        return str(question), str(label)

    def sft_map(row, label_pad_token_id: int = -100) -> dict:
        question, label = _question_and_label(row)
        messages = [
            {"role": "system", "content": data_args.system_prompt},
            {"role": "user", "content": question},]
        
        prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        prompt_len = len(prompt_tokens)
        messages.append({"role": "assistant", "content": label})
        prompt_response_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True)
        labels = prompt_response_tokens.copy()
        labels[:prompt_len] = [label_pad_token_id] * prompt_len
        return {
            "input_ids": prompt_response_tokens,
            "attention_mask": [1]*len(prompt_response_tokens),
            "labels": labels,
        }
        
    dataset = dataset.map(sft_map)

    remove_columns = [column for column in dataset["train"].column_names if column not in {"input_ids", "attention_mask", "labels"}]
    dataset = dataset.remove_columns(remove_columns)

    print(f"-----------------------------Finished——loading——dataset--------------------------------")
    for split in ['train', 'test']:
        for i, example in enumerate(dataset[split]):
            labels = example['labels']
            if not isinstance(labels, list) or len(labels) == 0:
                print(f"发现问题的样本在 {split} 集的第 {i} 个：labels = {labels}, 类型 = {type(labels)}")
                
    # initiate trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(tokenizer,pad_to_multiple_of=8, return_tensors="pt", padding=True)
    )
    # train and save
    trainer.train()
    save_name = "checkpoint-best" if training_args.load_best_model_at_end else "checkpoint-final"
    trainer.save_model(os.path.join(training_args.output_dir, save_name))


if __name__ == "__main__":
    train()
