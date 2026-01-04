## test_oai.py 生成训练数据结束之后，手动清洗数据，添加特定的标志符号
## 生成数据存进 safe_imitation_data_1k_complete.json

path = "/c23030/ckj/code/vlm/jailbreaking/DRA/safe_imitation_data_500_complete.json"
output_path = "/c23030/ckj/code/vlm/jailbreaking/DRA/safe_imitation_data_1000_complete.json"

import json
import re

def process_prefix_content(prefix):
    """
    处理前缀内容:
    a) 如果包含 '", then,"' 则插入continue变为 '"<|continue|>, then,"'
    b) 如果从then到内容最后，中间还有句号，则在句号之前插入<|continue|>（最后一句话不算）
    """
    # 步骤a: 处理 ", then," 模式
    if ', then' in prefix:
        prefix = prefix.replace(', then', '.<|continue|>\nthen')
    
    # 步骤b: 查找then的位置，处理then之后的句号
    # 查找最后一个then的位置（可能有多个，我们处理从最后一个then开始的内容）
    then_pattern = r'(?:then|, then)'
    then_matches = list(re.finditer(then_pattern, prefix, re.IGNORECASE))
    
    if then_matches:
        # 获取最后一个then的位置
        last_then_match = then_matches[-1]
        start_pos = last_then_match.start()
        
        # 获取从then开始到结尾的文本
        text_after_then = prefix[start_pos:]
        
        # 找出所有的句号位置（.）
        period_positions = [match.start() for match in re.finditer(r'\.', text_after_then)]
        
        # 如果有多个句号（至少2个，因为最后一句话的句号不算）
        if len(period_positions) >= 2:
            # 从后往前处理，在除最后一个句号外的所有句号前插入<|continue|>
            for i in range(len(period_positions) - 1):
                # 注意：每次插入后位置会变化，所以需要计算偏移量
                offset = i * len("<|continue|>")
                insert_pos = start_pos + period_positions[i] + offset
                
                # 插入<|continue|>到句号前
                prefix = prefix[:insert_pos] + "<|continue|>" + prefix[insert_pos:]
    
    return prefix

def clean_reflect_answer(text):
    """
    清洗reflect_answer字段
    1. 替换 <|continue|>\n\n<|reflect|>\n 为 \n\n<|reflect|>\n
    2. 去掉结尾的 <|continue|>
    3. 处理 \n\n<|reflect|>\n 之前的内容
    """
    if not isinstance(text, str):
        return text
    
    # 步骤1: 替换特定模式
    text = text.replace("<|continue|>\n\n<|reflect|>\n", "\n\n<|reflect|>\n")
    
    # 步骤2: 去掉结尾的 <|continue|>
    if text.endswith("<|continue|>"):
        text = text[:-len("<|continue|>")]
    
    # 步骤3: 分割字符串，获取\n\n<|reflect|>\n之前的内容
    parts = text.split("\n\n<|reflect|>\n", 1)
    if len(parts) != 2:
        return text
    
    prefix = parts[0]  # 需要处理的部分
    suffix = parts[1]  # 保留的部分
    
    # 步骤4: 处理prefix部分
    processed_prefix = process_prefix_content(prefix)
    
    # 重新组合字符串
    result = processed_prefix + "\n\n<|reflect|>\n" + suffix
    return result

def clean_json_data(data):
    """
    清洗整个数据列表
    """
    if not isinstance(data, list):
        raise ValueError("输入数据必须是list类型")
    
    cleaned_data = []
    
    for item in data:
        if 'reflect_answer' not in item:
            print(f"警告: 缺少reflect_answer字段: {item}")
            cleaned_data.append(item)
            continue
        try:
            cleaned_item = item.copy()
            cleaned_item['reflect_answer'] = clean_reflect_answer(item['reflect_answer'])
            cleaned_data.append(cleaned_item)
        except Exception as e:
            print(f"处理错误: {e}, 原始数据: {item}")
            cleaned_data.append(item)
    
    return cleaned_data

def process_json_file(input_file, output_file):
    try:
        # 读取输入文件
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        cleaned_data = clean_json_data(data)
        
        # 保存到输出文件
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(cleaned_data, f, ensure_ascii=False, indent=2)
        
        print(f"处理完成! 已保存到: {output_file}")
        print(f"处理了 {len(cleaned_data)} 条数据")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {input_file}")
    except json.JSONDecodeError as e:
        print(f"错误: JSON解析失败: {e}")
    except Exception as e:
        print(f"错误: {e}")

# 使用示例
if __name__ == "__main__":

    process_json_file(path, output_path)