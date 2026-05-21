import json
import os

from ..paths import DATA_SRC_DIR

def simple_extract():
    base_dir = str(DATA_SRC_DIR)
    
    # 文件配置
    files = [
        ("DRA_safe_imitation_data_1k_complete.json", 300, "DRA"),
        ("DrAttack_safe_imitation_data_500_complete.json", 200, "DrAttack"),
        ("DrAttack(benign)_safe_imitation_data_300_complete.json", 100, "DrAttack")
    ]
    
    all_data = []
    
    for filename, num_items, label in files:
        filepath = os.path.join(base_dir, filename)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 假设data是列表
            if isinstance(data, list):
                # 取后N条
                start_idx = max(0, len(data) - num_items)
                selected = data[start_idx:]
                
                # 转换格式
                for item in selected:
                    if "query" in item:
                        all_data.append({
                            "query": item["query"],
                            "label": label
                        })
                
                print(f"从 {filename} 提取了 {len(selected)} 条数据")
            else:
                print(f"警告: {filename} 不是列表格式")
                
        except Exception as e:
            print(f"处理 {filename} 时出错: {str(e)}")
    
    # 保存结果
    output_file = os.path.join(base_dir, "merged_data.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n总共提取了 {len(all_data)} 条数据")
    print(f"保存到: {output_file}")
    
    # 统计信息
    dra_count = sum(1 for item in all_data if item["label"] == "DRA")
    drattack_count = sum(1 for item in all_data if item["label"] == "DrAttack")
    
    print(f"DRA: {dra_count} 条")
    print(f"DrAttack: {drattack_count} 条")

if __name__ == "__main__":
    simple_extract()
