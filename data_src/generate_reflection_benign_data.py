

import os
import re
import math
import torch
import anthropic,openai
from enum import Enum
# from dotenv import load_dotenv
from dataclasses import dataclass
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
# load_dotenv()
import json
import hashlib
from typing import List, Dict, Any
from pathlib import Path
import random
class MODELS(Enum):
    LLAMA = "llama"
    CLAUDE = "claude"
    OPENAI = "openai"

@dataclass
class Reflection:
    num: str ## number thinking steps
    context: str ## verification to correctness / correction to mistakes
    have_solution: bool = None ## only the incorrect solution have the alternative solutions
    solution: str = "" ## the alternative solutions


#### Refer to Satori (https://arxiv.org/pdf/2502.02508) in Section Multi-agent COAT Data Synthesis. 


def add_reflect_continue(output: str, reflect: list[Reflection], num_reflect_step: int):
    data = ""
    sentences = output.split("\n\n")

    # 创建反思映射表，映射步骤编号到反思对象
    reflection_map = {}
    for r in reflect:
        try:
            step_indices = int(r.num)
            if step_indices < len(sentences) - 1:  # 不插入最后一步
                reflection_map[step_indices] = r
        except ValueError:
            continue  # 如果反思的步骤编号无效，跳过该反思
    
    
    reflect_step = None
    # 随机选择一个步骤插入反思
    if reflection_map:
        reflect_step = random.choice(list(reflection_map.keys()))  # 这里 keys() 是 int

    inserted = False
    for i, sentence in enumerate(sentences):
        data += sentence

        if i == len(sentences) - 1:
            break
        # 随机选的一步插入 reflection
        if not inserted and i == reflect_step:
            r = reflection_map[i]
            data += "\n\n<|reflect|>\n" + r.context + "\n<|continue|>\n"
            inserted = True
        else:
            data += "\n<|continue|>\n"

    return data


def add_reflect_continue_explore(output: str, reflect: list[Reflection], num_reflect_step: int):
    data = ""
    sentences = output.split("\n\n")

    # 创建反思映射表，映射步骤编号到反思对象
    reflection_map = {}
    for r in reflect:
        try:
            step_idx = int(r.num)
            reflection_map[step_idx] = r
        except ValueError:
            continue  # 如果反思的步骤编号无效，跳过该反思

    inserted = False  # 标记是否已插入反思
    reflect_range = range(1, 5)  # 只在第 3 到第 6 步之间插入反思（即索引 2 到 5）

    for i, sentence in enumerate(sentences):
        data += sentence
        
        # 如果到了最后一步，停止
        if i == len(sentences) - 1:
            break

        # 只在第 3 到第 6 步之间插入反思
        if i in reflect_range and i in reflection_map and not inserted:
            r = reflection_map[i]
            # 插入反思和探索内容
            data += "\n\n<|reflect|>\n" + r.context + "<|explore|>\n" + r.solution + "<|continue|>\n"
            inserted = True  # 标记已经插入了反思
            break  # 插入反思后立即停止，结束处理

        # 如果还没有插入反思，继续添加<|continue|>标签
        elif not inserted:
            data += "<|continue|>\n"

        else:
            break  # 如果已经插入过反思或超出反思范围，结束

    return data





type_3_generation_template = """
Your task is to continue writing a few brief sentences, adhering to the "helpful, respectful, and honest" principles. You must strictly refrain from offering any methods, details, or instructions to carry out illegal, violent, or harmful actions, including but not limited to kidnapping, tracking, or harming others. In your continuation, you must:
- Explicitly and briefly refuse to assist with any illegal activity
- Provide a specific, legal alternatives (e.g., contacting law enforcement, seeking legal aid, conflict mediation, mental health support)
- If applicable, include a very brief suggestion to reach out for professional assistance (such as contacting local authorities, a lawyer, or a crisis hotline)
Only output the continued solution content, not any meta information or feedback.

## Revision feedback:
"{reflection}"

## Continued solution
"""



type_3_template = """
You are a student. Review your incorrect partial solution, reflecting on mistakes and proposing alternative ethical solutions. Follow these steps:

1. You are given a step-by-step solution, with each step separated by "\n\n".
2. For each step:
   - Point out what is wrong in one sentence using the format: "In Step <id>: <my brief explanation of mistake>"
   - Suggest an alternative ethical solution: "Alternatively: <my alternative claim>"

Repeat this process for all steps, ensuring clarity and conciseness.

Required format (exactly):
In Step <id>: <first-person one-sentence explaination of what is wrong>
Alternatively: <my brief alternative claim>

Do NOT include any other text.

Solution: "{answer}"

"""

    #- Do NOT create a new line or a separate 'Confirmation:' line.
#For each step, output only (exactly):

type_2_template = """
You are a student. Review your partial solution. Follow these steps:

1. You are given a step-by-step solution, with each step strictly separated by "\n\n".
2. For every step:
    - Write one short first-person sentence explaining what I did in this step.
    - Then write one short sentence confirming that my answer is correct.

Required format (exactly):
In Step <id>: <short first-person explanation and confirmation>

Do NOT include any other text.

Solution: "{answer}"
"""


class CriticGenerator:
    def __init__(self, model_critic: MODELS = MODELS.LLAMA):
        self.model_critic = model_critic
        self.setup_model()

    def setup_model(self):
        if self.model_critic == MODELS.LLAMA:
            self.setup_llama()
        elif self.model_critic == MODELS.OPENAI:
            self.setup_openai()

    def setup_openai(self):
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self.client = openai.OpenAI(api_key=api_key,base_url=base_url)

    def setup_llama(self):
        # model_name = "/fs-computility/ai-shen/shared/huggingface/models/meta-llama/Llama-3.3-70B-Instruct"
        # model_name = "/c23030/ckj/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B-Instruct/snapshots/0e9e39f249a16976918f6564b8830bc894c89659"
        model_name = "/c23030/ckj/.cache/huggingface/hub/models--meta-llama--Llama-2-70b-chat-hf/snapshots/e9149a12809580e8602995856f8098ce973d1080"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(model_name,local_files_only=True,torch_dtype=torch.float16, device_map="auto")
        print("成功从缓存加载模型！")

    def query_model(self, prompt: str, max_tokens: int = 512, temperature: float = 0.2) -> str:
        if self.model_critic == MODELS.LLAMA:
            return self._query_llama(prompt, max_tokens, temperature)
        elif self.model_critic == MODELS.OPENAI:
            return self._query_openai(prompt, max_tokens, temperature)

    def _query_llama(self, prompt: str, max_tokens: int, temperature: float) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature)[0]
            # 去掉 prompt 的 token（假设生成输出开头包含输入）
            generated_ids = output_ids[len(inputs[0]):]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(text)
            # breakpoint()
        return text
        
    def _query_openai(self, prompt: str, max_tokens: int, temperature: float) -> str:
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",  # 或者 gpt-4o-2024-08-06, gpt-3.5-turbo, gpt-4o-mini
                # model="claude-3-5-haiku-latest",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=3000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Openai API error: {e}")
            return ""

    def generate(self, user_query: str, type2: str = "", type3: str = ""):
        if type2:
            ## generate reflection for type2
            prompt = type_2_template.format(query=user_query, answer=type2)
            results = self.query_model(prompt, max_tokens=3000, temperature=0.2)
            reflection = []
            # pattern只匹配 In Step <id>: xxx
            pattern = r"In Step (\d+): (.+?)(?=\n|$)"
            matches = re.findall(pattern, results)
            for num, context in matches:
                reflection.append(Reflection(num=num, context=context.strip(), have_solution=False, solution=None))
            ## generate exploration
            reflect = add_reflect_continue(output=type2, reflect=reflection, num_reflect_step=12)
            # breakpoint()

            return reflect
        
        elif type3:
            ## generate refletion
            prompt = type_3_template.format(query=user_query, answer=type3)
            results = self.query_model(prompt, max_tokens=512, temperature=0.2)
            reflection = []
            pattern = r"In Step (\d+): (.+?)\s*Alternatively: (.+?)(?=\n|$)"
            matches = re.findall(pattern, results)
            for num,context,solution in matches:
                reflection.append(Reflection(num=num,context=context.strip(),have_solution=True,solution=solution))

            # breakpoint()
            ## generate exploration
            reflect = add_reflect_continue_explore(output=type3, reflect=reflection, num_reflect_step=12)
            # breakpoint()

            full_prompt = type_3_generation_template.format(query=user_query, reflection=reflect)
            full_results = self.query_model(full_prompt, max_tokens=300, temperature=0.3)
            full_data = reflect +"\n" + full_results
            return full_data

# helper: 生成唯一 id（可用于去重）
def make_id(record: Dict[str, Any]) -> str:
    # 用 attack_query + answer 生成 hash（如果需要可以包括原始 query 等字段）
    key = (record.get("original_query", "") + "||" + record.get("attack_query", "") + "||" + (record.get("answer", "") or "")).encode("utf-8")
    return hashlib.sha256(key).hexdigest()

# 读取已有输出中的 id，用于跳过重复
def load_existing_ids(output_file: Path) -> set:
    existing = set()
    if not output_file.exists():
        return existing
    with output_file.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                if "id" in obj:
                    existing.add(obj["id"])
            except Exception:
                continue
    return existing


# 加载源文件（为节省内存，先一次性读取顶层结构，但只在 results 下构建待处理列表）
def gather_items_to_process(input_path: Path) -> List[Dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    items = []
    # 假设结构如示例，顶层有 "results": { "<query>": { "qa": [...] }, ...}
    results = data.get("results", {})
    for original_query, qadata in results.items():
        qa_list = qadata.get("qa", [])
        for qa_entry in qa_list:
            # 筛选条件：三个字段同时满足
            if qa_entry.get("jailbreak_check") is True and qa_entry.get("em") == 1 and qa_entry.get("harmbench") is True:
                attack_query = qa_entry.get("prompt", "")
                answer = qa_entry.get("answer", "")
                record = {
                    "original_query": original_query,
                    "attack_query": attack_query,
                    "answer": answer,
                    # 其它原始字段可按需保存
                }
                items.append(record)
    return items

def gather_benign_items_to_process(input_path: Path) -> List[Dict[str, Any]]:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    items = []
    # 假设结构如示例，顶层有 "results": { "<query>": { "qa": [...] }, ...}
    results = data.get("results", {})
    
    for original_query, qadata in results.items():
        qa_list = qadata.get("qa", [])
        for qa_entry in qa_list:
            # 筛选条件：三个字段同时满足，并且 answer 以 "I" 开头
            if (
                qa_entry.get("jailbreak_check") is False
                and qa_entry.get("em") == 0
                and qa_entry.get("harmbench") is False
                and isinstance(qa_entry.get("answer"), str)
                and qa_entry.get("answer").startswith("I")
            ):
                record = {
                    "original_query": original_query,
                    "attack_query": qa_entry.get("prompt", ""),
                    "answer": qa_entry.get("answer", ""),
                    # 其它原始字段可按需保存
                }
                items.append(record)
                # 满足条件后继续下一个 qa_entry
                break

    return items


# 逐条生成 reflect_answer 并 append 保存（保证每生成一条就保存）
def process_and_save(items: List[Dict[str, Any]], critic_model, output_file: Path, batch_size: int = 32, use_type2: bool = False, use_type3: bool = False):
    existing_ids = load_existing_ids(output_file) if SKIP_IF_DUPLICATE else set()

    # open file in append mode
    with output_file.open("a", encoding="utf-8") as fout:
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]
            # 这里我们仍逐条调用 generate（如果你的 critic_model 支持批量调用，可在此处整合）
            for rec in batch:
                rec_id = make_id(rec)
                if rec_id in existing_ids:
                    print(f"Skipping existing record id={rec_id}")
                    continue

                try:
                    attack_query = rec["attack_query"]
                    answer = rec["answer"]

                    # 调用 critic model 生成 reflect_answer
                    # 注意：根据实际接口调整参数名和返回处理
                    if use_type3:
                        reflect = critic_model.generate(user_query=attack_query, type3=answer)
                    if use_type2:
                        reflect = critic_model.generate(user_query=attack_query, type2=answer)

                    # 假设 reflect 是可 JSON 化的。如果是对象，请将其序列化或提取文本字段
                    # 这里我们把 reflect 原样保存为 reflect_answer 字段（若为复杂对象可改为 reflect.get('text') 等）
                    out_rec = {
                        "id": rec_id,
                        "original_query": rec.get("original_query"),
                        "attack_query": attack_query,
                        "answer": answer,
                        "reflect_answer": reflect
                    }

                    # 写入一行 JSON（jsonl），并立即 flush 磁盘
                    fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                    fout.flush()
                    os.fsync(fout.fileno())  # 强制同步到磁盘（降低丢失风险）

                    # 把 id 加入已存在集合以避免重复
                    existing_ids.add(rec_id)

                    print(f"Saved record id={rec_id}")

                except Exception as e:
                    # 出现异常但保持已保存的数据完整（下次可继续）
                    print(f"Error processing record (id={rec_id}): {repr(e)}")
                    # 你可以记录错误到日志或另存文件以便后续排查
                    continue


# --- 配置项 ---
INPUT_PATH = Path("/DRA/results/attack/batch_llama2_13b_0_120_result.json")
OUTPUT_DIR = Path("/results/DRA_processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSONL = OUTPUT_DIR / "filtered_attack_with_reflections_benign.jsonl"  # 追加写入
BATCH_SIZE = 4  # 每个批次的“处理单元数” —— 实际调用 critic_model.generate 是逐条的，可以根据需要调整
SKIP_IF_DUPLICATE = True  # 如果已存在相同 id 则跳过

if __name__ == "__main__":

    critic_model = CriticGenerator(MODELS.LLAMA)

############ DRA 处理逻辑 #########
    # 2) 收集待处理项（筛选出满足三个条件的 qa entries）
    items = gather_items_to_process(INPUT_PATH)
    print(f"Total harmful items matching filter: {len(items)}")
    process_and_save(items, critic_model, OUTPUT_JSONL, batch_size=BATCH_SIZE,use_type3=True)

    # 3) 逐批处理并append保存
    benign_items = gather_benign_items_to_process(INPUT_PATH)
    print(f"Total benign items matching filter: {len(benign_items)}")

    process_and_save(benign_items, critic_model, OUTPUT_JSONL, batch_size=BATCH_SIZE,use_type2=True)
