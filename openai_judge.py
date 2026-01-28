import json
import os
from openai import OpenAI

# 配置
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("OPENAI_BASE_URL")
)

judge_prompt = '''You are a data evaluation expert. Below is a model's response and a standard answer. The model's answer is hidden in its response and not in standard format. The standard answer is the data after '#### ' in the sentence.

Model response:
{model_answer}

Standard answer:
{label}

Compare the model's final answer (extracted from the model response) with the standard answer (the part after '#### '). 
If they are identical, output 'yes'. Otherwise, output 'no'. Output strictly only 'yes' or 'no', with no other content.'''

try:
    # 读取JSON文件
    file_path = '/results/model_answer/gsm8k.json'
    with open(file_path, 'r') as f:
        data_list = json.load(f)
    
    print(f"成功读取文件，共 {len(data_list)} 条数据")

except FileNotFoundError:
    print(f"错误：找不到文件 {file_path}")
    exit(1)

yes_count = 0
total_count = 0
failed_items = []

# 评估每个答案
for i, item in enumerate(data_list):
    model_answer = item.get('answer', '')
    label = item.get('label', '')
    
    if not model_answer or not label:
        print(f"警告：第 {i+1} 条数据缺少answer或label，跳过")
        continue
    
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are a data evaluation expert. Follow instructions exactly."},
                {"role": "user", "content": judge_prompt.format(model_answer=model_answer, label=label)}
            ],
            max_tokens=10,
            temperature=0,
            timeout=10  # 添加超时设置
        )
        
        # 获取评估结果
        evaluation = response.choices[0].message.content.strip().lower()
        
        # 验证结果格式
        if evaluation not in ['yes', 'no']:
            print(f"警告：第 {i+1} 条返回了非标准格式: {evaluation}")
            evaluation = 'no'  # 默认标记为不一致
        
        item['evaluation'] = evaluation
        
        # 统计
        if evaluation == 'yes':
            yes_count += 1
        total_count += 1
        
        print(f"进度：{i+1}/{len(data_list)} - 结果: {evaluation}")
        
    except Exception as e:
        print(f"第 {i+1} 条数据评估失败: {e}")
        failed_items.append(i+1)
        item['evaluation'] = 'error'
        continue

# 计算一致率
if total_count > 0:
    consistency_rate = yes_count / total_count
else:
    consistency_rate = 0
    print("错误：没有成功评估任何数据")

# 保存带评估结果的文件
try:
    output_path = file_path.replace('.json', '_evaluated.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data_list, f, indent=2, ensure_ascii=False, separators=(',', ': '))
    print(f"评估结果已保存到: {output_path}")
    
except Exception as e:
    print(f"保存评估结果时出错: {e}")

# 保存统计信息
try:
    stats = {
        "total_data": len(data_list),
        "successfully_evaluated": total_count,
        "failed_evaluations": len(failed_items),
        "failed_indices": failed_items,
        "consistent_answers": yes_count,
        "consistency_rate": consistency_rate
    }
    
    stats_path = file_path.replace('.json', '_stats.json')
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"统计信息已保存到: {stats_path}")
    
except Exception as e:
    print(f"保存统计信息时出错: {e}")

# 打印统计信息
print(f"\n=== 评估统计 ===")
print(f"数据总量: {len(data_list)}")
print(f"成功评估: {total_count}")
print(f"评估失败: {len(failed_items)}")
if failed_items:
    print(f"失败索引: {failed_items}")
print(f"一致答案: {yes_count}")
print(f"一致率: {consistency_rate:.2%}")