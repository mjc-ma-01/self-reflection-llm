

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

from ..paths import DRA_PROCESSED_DIR, PROJECT_ROOT

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

def add_continue(output: str, reflect: list[Reflection]):
    data = ""
    sentences = output.split("\n\n")
    for i,sentence in enumerate(sentences):
        if i == len(sentences)-1:
            data += i
        else:
            data += i + "<|continue|>\n"
    return data

import random

def add_reflect_continue_explore(output: str, reflect: list[Reflection], num_reflect_step: int):
    """
    随机在前几个步骤（默认1-3）中插入反思
    """
    
    # 创建反思映射表
    reflection_map = {}
    for r in reflect:
        try:
            step_idx = int(r.num)
            reflection_map[step_idx] = r
        except ValueError:
            continue
    
    if not reflection_map:
        return 
    
    # 随机选择要插入的步骤范围（可以调整）
    target_step = random.randint(1, 3)
    
    # 定义步骤行正则表达式
    step_pattern = re.compile(r'^\s*(\d+)\s*&\s*(.*?)\s*\\\\\s*$')
    
    lines = output.split('\n')
    result_lines = []
    
    for i, line in enumerate(lines):
        result_lines.append(line)
        
        # 检查是否是步骤行
        match = step_pattern.match(line.strip())
        if match:
            step_num = int(match.group(1))
            
            # 检查是否需要在此步骤后插入反思
            if step_num == target_step and step_num in reflection_map:
                r = reflection_map[step_num]
                
                # 插入反思内容
                result_lines.append("\n<|reflect|>")
                result_lines.append(r.context)
                result_lines.append("<|explore|>")
                result_lines.append(r.solution)
                result_lines.append("<|continue|>")
                
                # 插入后立即结束循环，不再处理后续行
                return '\n'.join(result_lines)
    
    return '\n'.join(result_lines)

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
You are a student asked to review your own provided instructional guide. Your task is to conduct a security and ethics review by analyzing each step in the provided LaTeX table for potentially harmful, dangerous, or unethical instructions. Respond strictly in the first person ("I", "my").

**Your Process:**
1.  **Identify Harmful Intent:** First, determine if the overall instructional goal demonstrated in the table has an unethical, illegal, or dangerous objective.
2.  **Analyze Step Content:** Examine each numbered step in the table (e.g., "\n1 &", "\n2 &", etc.) to pinpoint any specific harmful instructions, unsafe practices, or unethical guidance contained within the description.
3.  **Reflect & Correct:** For each harmful element you find in the steps, provide two lines:
    *   A reflection on what is wrong with that specific step you wrote.
    *   A brief, ethical alternative for that instructional segment.

**Output Format Rules (STRICT):**
- Your entire response must be a list of reflections following the pattern below.
- You must analyze the steps sequentially in the order they appear in the table.
- Use **EXACTLY** this two-line pattern for each finding:
    `In Step <step_number>: <My brief explanation of the harmful intent/content I wrote in this step>`
    `Alternatively: <My brief description of how to correct this step ethically>`
- Do NOT output the original table, additional explanations, apologies, or markdown formatting like ```.
- If no harmful elements are found, output: "No harmful instructions identified in the steps."

**Examples:**
For a table teaching credit card usage (like the example), a correct response might be:
In Step 1: My instruction to "enter the PIN" could potentially be misused if the reader is attempting to use someone else's card without permission, which constitutes fraud.
Alternatively: This step should emphasize that one must only use their own card with their own PIN and remind users to protect their PIN from being observed.

**Now, review your own instructional table below:**
{answer}
"""

type_3_template_before = """
You are a student. Review your incorrect partial solution, reflecting on mistakes and proposing alternative ethical solutions. Follow these steps:

1. You are given a query and a step-by-step solution, with each step separated by "\n\n".
2. For each step:
   - Point out what is wrong in one sentence using the format: "In Step <id>: <my brief explanation of mistake>"
   - Suggest an alternative ethical solution: "Alternatively: <my alternative claim>"

Repeat this process for all steps, ensuring clarity and conciseness.

**Required Format (EXACTLY)**:
In Step <id>: <first-person one-sentence explaination of what is wrong>
Alternatively: <my brief alternative claim>

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
        model_name = os.getenv("REFLECTOR_CRITIC_MODEL")
        if not model_name:
            raise ValueError("Set REFLECTOR_CRITIC_MODEL when using the local critic backend.")
        local_files_only = os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            local_files_only=local_files_only,
            torch_dtype=torch.float16,
            device_map="auto",
        )
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
            generated_ids = output_ids[len(inputs[0]):]
            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
            print(text)
            # breakpoint()
        return text
        
    def _query_openai(self, prompt: str, max_tokens: int, temperature: float) -> str:
        max_retries = 5  # 最大重试次数
        initial_delay = 1  # 初始延迟（秒）
        max_delay = 10  # 最大延迟（秒）
        
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-3.5-turbo",  # 或者 gpt-4o-2024-08-06, gpt-3.5-turbo, gpt-4o-mini，claude-haiku-4-5-20251001，claude-3-sonnet-20240229
                # model="claude-3-5-haiku-latest",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=2048,
                    timeout=30.0  # 设置超时时间为30秒
                )
                text = response.choices[0].message.content.strip()
                print(text)
                return text
                
            except APITimeoutError as e:
                print(f"API超时 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = min(max_delay, initial_delay * (2 ** attempt))
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print("重试次数已用完")
                    return ""
                    
            except RateLimitError as e:
                print(f"速率限制 (尝试 {attempt + 1}/{max_retries}): {e}")
                if "rate limit" in str(e).lower():
                    wait_time = 60  # 速率限制通常需要等待更长时间
                else:
                    wait_time = min(max_delay, initial_delay * (2 ** attempt))
                
                print(f"等待 {wait_time} 秒后重试...")
                time.sleep(wait_time)
                
            except APIError as e:
                print(f"API错误 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    wait_time = min(max_delay, initial_delay * (2 ** attempt))
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
                else:
                    print("重试次数已用完")
                    return ""
                    
            except Exception as e:
                print(f"未知错误: {e}")
                return ""
        
        return ""


    def generate(self, user_query: str, type2: str = "", type3: str = ""):
        if type2:
            prompt = type_2_template.format(query=user_query, answer=type2)
            results = self.query_model(prompt)
            reflection = []
            pattern = r"Verify (\d+): (.+?)(?=\n|$)"
            matches = re.findall(pattern, results)
            for num,context in matches:
                reflection.append(Reflection(num=num,context=context.strip(),have_solution=False))
            data = add_reflect_continue(output=type2, reflect=reflection)
            return data
        
        elif type3:
            ## generate refletion
            prompt = type_3_template.format(query=user_query, answer=type3)
            results = self.query_model(prompt, max_tokens=4000, temperature=0.5)
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



def process_and_save(items: List[Dict[str, Any]], critic_model, output_file: Path, batch_size: int = 32):
    # 首先读取已存在的query，避免重复处理
    existing_queries = set()
    
    # 如果输出文件已存在，读取其中所有的query
    if output_file.exists():
        try:
            with output_file.open("r", encoding="utf-8") as fin:
                for line in fin:
                    try:
                        record = json.loads(line.strip())
                        if "query" in record:
                            existing_queries.add(record["query"])
                    except json.JSONDecodeError:
                        # 跳过格式错误的行
                        continue
        except Exception as e:
            print(f"⚠️ 读取现有文件时出错: {e}")
    
    print(f"📊 发现已有 {len(existing_queries)} 个不同query在输出文件中")
    
    with output_file.open("a", encoding="utf-8") as f:
        processed_count = 0
        skipped_count = 0
        
        for start in range(0, len(items), batch_size):
            batch = items[start:start + batch_size]

            for rec in batch:
                attack_query = rec["query"]
                
                # 检查这个query是否已经存在于输出文件中
                if attack_query in existing_queries:
                    skipped_count += 1
                    if skipped_count % 100 == 0:  # 每跳过100条打印一次日志
                        print(f"⏭️  已跳过 {skipped_count} 个重复样本")
                    continue
                
                answer = rec["reflect_answer"]

                reflect = critic_model.generate(user_query=attack_query, type3=answer)
                output_record = {
                    "query": attack_query,
                    "reflect_answer": reflect
                }
                
                f.write(json.dumps(output_record, ensure_ascii=False) + "\n")
                f.flush()  # 立即写入磁盘
                existing_queries.add(attack_query)
                processed_count += 1
            
            print(f"📈 进度: 已处理 {processed_count} 个，跳过 {skipped_count} 个重复，总计 {len(items)} 个样本")
    
    print(f"✅ 处理完成！新增处理 {processed_count} 个样本，跳过 {skipped_count} 个重复样本")
    return processed_count, skipped_count

# --- 配置项 ---
INPUT_PATH = PROJECT_ROOT / "DRA" / "data_source" / "ReLLM_latex_table.json"
OUTPUT_DIR = DRA_PROCESSED_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_JSONL = OUTPUT_DIR / "GPT_ReLLM(table)_filtered_attack_with_reflections.jsonl"  # 追加写入
BATCH_SIZE = 8  # 每个批次的“处理单元数” —— 实际调用 critic_model.generate 是逐条的，可以根据需要调整

if __name__ == "__main__":

    critic_model = CriticGenerator(MODELS.OPENAI)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        items = json.load(f)  # 读取JSON列表
    
    print(f"Total items matching filter: {len(items)}")
    process_and_save(items, critic_model, OUTPUT_JSONL, batch_size=BATCH_SIZE)
