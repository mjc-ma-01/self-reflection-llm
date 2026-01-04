import json
import os

def simple_process_jsonl(input_file):
    """简洁版本，只进行简单的字符串替换，并过滤不含<|reflect|>的数据"""
    
    output_file = input_file.replace('.jsonl', '_modified_filtered.jsonl')
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8') as f_out:
        
        processed_count = 0
        filtered_count = 0
        total_count = 0
        
        for line in f_in:
            total_count += 1
            try:
                data = json.loads(line.strip())
                
                # 检查reflect_answer字段是否存在且为字符串
                if 'reflect_answer' in data and isinstance(data['reflect_answer'], str):
                    
                    # 检查是否包含<|reflect|>字段
                    if '<|reflect|>' not in data['reflect_answer']:
                        filtered_count += 1
                        continue  # 跳过不包含<|reflect|>的数据
                    
                    # 进行简单的字符串替换
                    old_str = '} \\\\\n\\hline\n'
                    new_str = '<|continue|>\n} \\\\\n\\hline\n'
                    
                    if old_str in data['reflect_answer']:
                        data['reflect_answer'] = data['reflect_answer'].replace(old_str, new_str)
                        processed_count += 1
                    else:
                        # 即使没有替换，也保留数据（因为包含<|reflect|>）
                        processed_count += 1
                
                # 写回文件（只包含有<|reflect|>的数据）
                f_out.write(json.dumps(data, ensure_ascii=False) + '\n')
                
            except Exception as e:
                print(f"处理行时出错: {e}")
                # 出错时不保留原行（因为可能格式有问题）
                continue
        
        print(f"处理完成！")
        print(f"总数据行数: {total_count}")
        print(f"过滤掉的行数: {filtered_count} (不包含<|reflect|>字段)")
        print(f"保留的行数: {processed_count}")
        print(f"输出文件: {output_file}")

# 使用示例
if __name__ == "__main__":
    file_path = "/c23030/ckj/code/vlm/jailbreaking/results/DRA_processed/GPT_ReLLM(table)_filtered_attack_with_reflections.jsonl"
    
    simple_process_jsonl(file_path)