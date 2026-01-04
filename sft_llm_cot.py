# dev: WANDB_MODE=offline WANDB_PROJECT=system-prompt-steerability-dev PYTHONPATH=. srun -p mllm-align --quotatype=reserved --gres=gpu:1 --cpus-per-task=16 --time=300 accelerate launch --config_file configs/accelerate_configs/single_gpu.yaml src/train.py
import os
import functools
from dataclasses import dataclass
import copy
import omegaconf
import torch
import transformers
from accelerate import PartialState
from peft import get_peft_model, LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer, BitsAndBytesConfig, Trainer
from datasets import concatenate_datasets, Dataset
from data import *

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
    model_path:       str = "/c23030/ckj/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
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
class TrainingArguments(transformers.TrainingArguments):
    output_dir: str = "/c23030/ckj/code/vlm/models/reflect_results"
    run_name: str = "/c23030/ckj/code/vlm/models/reflect_results"
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
    parser = transformers.HfArgumentParser((ModelArguments, PeftArguments, TrainingArguments))
    model_args, peft_args, training_args = parser.parse_args_into_dataclasses()
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
    
    data = ReflectDataset()
    dataset = data.get_dataset()

    # breakpoint()

    def sft_map(row, system_prompt: str = "You are a helpful and harmless assistant.",label_pad_token_id: int = -100) -> dict:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": row["question"]},]
        
        prompt_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        prompt_len = len(prompt_tokens)
        messages.append({"role": "assistant", "content": row["label"]})
        prompt_response_tokens = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=True)
        labels = prompt_response_tokens.copy()
        labels[:prompt_len] = [label_pad_token_id] * prompt_len
        return {
            "input_ids": prompt_response_tokens,
            "attention_mask": [1]*len(prompt_response_tokens),
            "labels": labels,
        }
        
    dataset = dataset.map(sft_map)

    dataset = dataset.remove_columns(["label","question"])

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
