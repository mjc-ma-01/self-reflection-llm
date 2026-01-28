
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import torch
import torch
import torch
import torch.nn.functional as F
import omegaconf
from datasets import Dataset
from torch.utils.data import DataLoader
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedTokenizer, HfArgumentParser
from dataclasses import dataclass
from PIL import Image
from typing import Optional,Union,Literal
import os, json
from data import *
# from harmbench_utils import *

def extract_train_name_from_path(model_path: str) -> str:
    parts = model_path.split("train:")
    if len(parts) > 1:
        # 获取train:后面的部分，再按"/"分割取第一部分
        after_train = parts[1]
        train_name = after_train.split("/")[0]
        return train_name
    return ""

def map_fn(
    row,
    tokenizer: PreTrainedTokenizer,
    system_prompt: str = "You are a helpful and harmless assistant.", 
) -> dict:
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user", 
            "content": row["question"]
        },
    ]
    prompt = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        tokenize=False
    )

    return {"prompt": prompt}


def llama_evaluate(model, tokenizer, eval_dataset, bs):
    ## 生成答案    
    results = []
    eval_dataset = eval_dataset.map(map_fn,fn_kwargs={"tokenizer":tokenizer})
    
    with torch.no_grad(): 
        for idx in range(0, len(eval_dataset), bs):
            batch = eval_dataset.select(range(idx, min(idx + bs, len(eval_dataset))))
            
            prompts = batch["prompt"]

            # tokenize the prompts
            inputs = tokenizer(prompts, padding=True, return_tensors="pt")
            input_ids = inputs["input_ids"].to("cuda")
            attention_mask = inputs["attention_mask"].to("cuda")

            # 生成文本答案
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=2048,  # 控制最大生成长度
                    num_return_sequences=1,  # 生成一个答案
                    do_sample=True,  # 采用贪心解码（可以改成True用于采样）
                    temperature=0.7,  # 控制生成的多样性（仅在 do_sample=True 时有效）
                )
            generated_texts = tokenizer.batch_decode(generated_ids[:, input_ids.shape[-1]:], skip_special_tokens=True) 

            for ex, answer in zip(batch, generated_texts):
                results.append({
                    "question": ex["question"],
                    "answer": answer,
                    "label": ex.get("label", None)})
            print(f"num_{idx}_examples")
            print(results[idx])
    return results

@dataclass
class ScriptArgument:


    # model_path: str = "thu-ml/STAIR-Llama-3.1-8B-SFT"
    model_path: str = "thu-ml/STAIR-Llama-3.1-8B-DPO-3"

    # 可以选用给定的字符串，也可以输入自定义路径（str）

    eval_ds: Union[
        Literal["xstest", "strongreject","wildchat", "donot", "simpleqa","gsm8k","advglue"],
        str
    ] = "wildchat"   # 默认值

    save_path: str = f"/results/model_answer/{eval_ds}.json"
    seed: int = 42
    load_in_4bit: bool = False
    use_flash_attention_2: bool = True
    batch_size: int = 4 ## 推理时一个批次的数量


if __name__ == "__main__":
    parser = HfArgumentParser(ScriptArgument)
    args = parser.parse_args_into_dataclasses()[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformers.set_seed(args.seed)

    # loading model and tokenizer
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
    if not tokenizer.pad_token: tokenizer.pad_token = tokenizer.eos_token
    
    
    if args.eval_ds == "xstest":
        c1 = XSTest(num_samples=100)
        eval_dataset = c1.get_dataset()  

    elif args.eval_ds == "strongreject":
        c2 = StrongReject(num_samples=100)
        eval_dataset = c2.get_dataset()
    elif args.eval_ds == "donot":
        c4 = DoNotAnswer(num_samples=200)
        eval_dataset = c4.get_dataset()  

    elif args.eval_ds == "wildchat":
        c4 = WildChat(num_samples=200)
        eval_dataset = c4.get_dataset() 

    elif args.eval_ds == "simpleqa":
        c4 = SimpleQA(num_samples=200)
        eval_dataset = c4.get_dataset()  

    elif args.eval_ds == "gsm8k":
        c4 = GSM8k(num_samples=500)
        eval_dataset = c4.get_dataset()  

    elif args.eval_ds == "advglue":
        c4 = Adv_GLUE()
        eval_dataset = c4.get_dataset()  

    def llm_collator(row, system_prompt: str="You are a helpful and harmless assistant.") -> dict:
        messages = [
            {"role": "system", "content": system_prompt},
            { "role": "user", "content": row["question"]},]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        return {"prompt": prompt}

    eval_dataset = eval_dataset.map(llm_collator)
     ## 生成答案
    results = []
    bs = 4
    for i in range(0, len(eval_dataset), bs):
        batch = eval_dataset.select(range(i, min(i + bs, len(eval_dataset))))
        prompts = batch["prompt"]
        inputs = tokenizer(prompts, padding=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to("cuda")
        attention_mask = inputs["attention_mask"].to("cuda")
        # 生成文本答案
        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=2048,  # 控制最大生成长度
                num_return_sequences=1,  # 生成一个答案
                do_sample=True,  # 采用贪心解码（可以改成True用于采样）
                temperature=0.7,  # 控制生成的多样性（仅在 do_sample=True 时有效）
            )
        generated_texts = tokenizer.batch_decode(generated_ids[:, input_ids.shape[-1]:], skip_special_tokens=True) 
        for ex, answer in zip(batch, generated_texts):
            results.append({
                "question": ex["question"],
                "answer": answer,
                "label": ex.get("label", None)})
        print(f"num_{i}_examples")
        print(results[i])
    args.save_path = f"/results/model_answer/{args.eval_ds}.json"

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    with open(args.save_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4, ensure_ascii=False)

    print(f"Results saved to {args.save_path}")
    

