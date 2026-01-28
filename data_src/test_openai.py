## 精细化 prompt，用gpt-3.5生成1k条训练数据

from openai import OpenAI
import json
import os
from prompt_config import *
import re

# 1. 设置客户端（示例为OpenAI官方库）
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
    )

def parse_text_response(raw_text):
    query = None
    reflect_answer = None
    
    if "===QUERY===" in raw_text and "===ANSWER===" in raw_text:
        parts = raw_text.split("===QUERY===")
        if len(parts) >= 2:
            after_query = parts[1]
            answer_parts = after_query.split("===ANSWER===")
            if len(answer_parts) >= 2:
                query = answer_parts[0].strip()
                reflect_answer = answer_parts[1].strip()

    if query and reflect_answer:
        if len(query) < 50:  # query应该比较长
            print(f"  [解析警告] 提取的query可能不完整，长度: {len(query)}字符")
        if "<|continue|>" not in reflect_answer:
            print("  [解析警告] reflect_answer中未找到必要的标签<|continue|>")
        
        print(f"  [解析成功] Query: {len(query)}字符, Answer: {len(reflect_answer)}字符")
        return {
            "query": query,
            "reflect_answer": reflect_answer
        }

def save_data(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def generate_single_entry():
    """生成单条仿写数据"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o", #"gpt-3.5-turbo-0125"
            messages=[
                {"role": "system", "content": "You are a precise data generation assistant. Follow instructions exactly."},
                {"role": "user", "content": create_harm_refuse.format(question=question)}
            ],
            temperature=0.7,
            max_tokens=4000,
            timeout=30.0  # 关键：为这个请求设置超时，避免无限等待
        )
        
        content = response.choices[0].message.content.strip()
        data_entry = parse_text_response(content)
        # breakpoint()

        if "query" not in data_entry or "reflect_answer" not in data_entry:
            print("警告: 生成的数据缺少必要字段")
            return None
        return data_entry

    except Exception as e:
        print(f"生成单条数据时出错: {e}")
        return None     
   

def generate_batch_data(num_entries=500, batch_size=10, delay_seconds=1):
    all_data = []      
    print(f"开始生成 {num_entries} 条数据...")

    for i in range(0, num_entries, batch_size):
        current_batch = min(batch_size, num_entries - i)
        
        for j in range(current_batch):
            entry_num = i + j + 1
            print(f"正在生成第 {entry_num}/{num_entries} 条数据...")
            
            entry = generate_single_entry()
            if entry:
                all_data.append(entry)
                print(f"  ✅ 成功生成第 {entry_num} 条数据")
            else:
                print(f"  ❌ 第 {entry_num} 条数据生成失败")
            
        if all_data:
            save_data(all_data, f"safe_imitation_data_part_{i//batch_size + 1}.json")
            print(f"已保存第 {i//batch_size + 1} 批数据")
        
    save_data(all_data, "DeAttack(benign)_safe_imitation_data_300_complete.json")
    print(f"\n完成! 共生成 {len(all_data)} 条数据")

    return all_data

if __name__ == "__main__":
    print("开始批量生成演示（500条）...")
    full_data = generate_batch_data(num_entries=400, batch_size=50, delay_seconds=1)