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

def parse_text_response(response_text):
    # 使用正则表达式匹配xxx标记内的内容
    pattern = r'===QUERY===(.*?)===QUERY==='
    match = re.search(pattern, response_text, re.DOTALL)
    
    if match:
        extracted = match.group(1).strip()
        query = '\n'.join([line.strip() for line in extracted.split('\n') if line.strip()])
        # 可选：移除可能的多余空白行
        print(f"  [解析成功] Query: {len(query)}字符 ")
        return {
            "query": query,
            "label": "DrAttack"
        }
    else:
        return response_text.strip()

def save_data(data, filename):
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def generate_single_entry(q):
    """生成单条仿写数据"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are a precise data generation assistant. Follow instructions exactly."},
                {"role": "user", "content": DrAttack_benign().format(general_question=q)}
            ],
            temperature=0.7,
            max_tokens=4000,
            timeout=30.0  # 关键：为这个请求设置超时，避免无限等待
        )
        
        content = response.choices[0].message.content.strip()
        data_entry = parse_text_response(content)

        if "query" not in data_entry:
            print("警告: 生成的数据缺少必要字段")
            return None
        return data_entry

    except Exception as e:
        print(f"生成单条数据时出错: {e}")
        return None     
   

def generate_batch_data(num_entries=300, batch_size=10, delay_seconds=1):
    # 读取q
    data_path = "/c23030/ckj/code/vlm/jailbreaking/data_src/general_data_from_alpaca/dataset.json"
    with open(data_path, 'r') as f:
        data = json.load(f)
    q_list = [item["query"] for item in data if item["label"] == "DrAttack"]
    
    # 检查q_list是否有足够数据
    if len(q_list) < num_entries:
        print(f"警告：q_list只有{len(q_list)}条数据，少于需要的{num_entries}条")
        q_list = q_list * (num_entries // len(q_list) + 1)
    
    all_data = []      
    print(f"开始生成 {num_entries} 条数据...")

    for i in range(0, num_entries, batch_size):
        current_batch = min(batch_size, num_entries - i)
        
        for j in range(current_batch):
            entry_num = i + j + 1
            print(f"正在生成第 {entry_num}/{num_entries} 条数据...")
            
            # 使用q_list中对应的q
            q = q_list[entry_num - 1]
            entry = generate_single_entry(q=q)
            if entry:
                all_data.append(entry)
                print(f"  ✅ 成功生成第 {entry_num} 条数据")
            else:
                print(f"  ❌ 第 {entry_num} 条数据生成失败")
            
        if all_data:
            save_data(all_data, f"safe_imitation_data_part_{i//batch_size + 1}.json")
            print(f"已保存第 {i//batch_size + 1} 批数据")
        
    save_data(all_data, "DrAttack_general_query_imitation_data_300.json")
    print(f"\n完成! 共生成 {len(all_data)} 条数据")

    return all_data

if __name__ == "__main__":
    print("开始批量生成演示（500条）...")
    full_data = generate_batch_data(num_entries=300, batch_size=50, delay_seconds=1)