

import json
from datasets import Dataset, load_dataset

from ..paths import PROJECT_ROOT


class ReNeLLM(Dataset):
    def __init__(self, num_samples=None) -> None:  # 添加num_samples参数
        self.num_samples = num_samples
        self.dataset = load_dataset("Deep1994/ReNeLLM-Jailbreak", token="")["train"]
        self.latex_table_prefix = "The latex table below needs"
        self.python_code_prefix = "The Python code below needs"
        self.paragraph_prefix = "Please supplement the second paragraph"

    def filter_by_prefix(self, prefix):
        """根据前缀筛选数据"""
        filtered_data = self.dataset.filter(
            lambda row: row["nested_prompt"].startswith(prefix)
        )
        return filtered_data

    def get_dataset_by_type(self, data_type):
        """根据类型获取数据集"""
        if data_type == "latex_table":
            filtered_dataset = self.filter_by_prefix(self.latex_table_prefix)
        elif data_type == "python_code":
            filtered_dataset = self.filter_by_prefix(self.python_code_prefix)
        elif data_type == "paragraph":
            filtered_dataset = self.filter_by_prefix(self.paragraph_prefix)
        else:
            raise ValueError(f"Unknown data type: {data_type}")
        
        # 转换列名
        original_columns = filtered_dataset.column_names
        transformed_dataset = filtered_dataset.map(
            lambda row: {
                "query": row["nested_prompt"],
                "reflect_answer": row["claude2_output"]
            },
            remove_columns=original_columns
        )
        
        # 随机选择指定数量的样本
        if self.num_samples and len(transformed_dataset) >= self.num_samples:
            return transformed_dataset.shuffle(seed=42).select(range(self.num_samples))
        else:
            print(f"⚠️  Warning: Only {len(transformed_dataset)} samples available for {data_type}, requested {self.num_samples}")
            return transformed_dataset.shuffle(seed=42)

    def save_dataset(self, dataset, save_path):
        """保存数据集到JSON文件"""
        data_list = [{"query": item["query"], "reflect_answer": item["reflect_answer"]} 
                     for item in dataset]
        
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 数据集已成功保存到 {save_path}")
        print(f"📊 保存了 {len(data_list)} 个样本")
        return data_list

if __name__ == "__main__":
    num_samples = 500  # 每种类型需要的样本数量
    
    # 初始化数据集
    c1 = ReNeLLM(num_samples=num_samples)
    
    # 分别获取三种类型的数据集
    print("🔄 正在获取 LaTeX table 类型数据...")
    latex_table_dataset = c1.get_dataset_by_type("latex_table")
    
    print("🔄 正在获取 Python code 类型数据...")
    python_code_dataset = c1.get_dataset_by_type("python_code")
    
    print("🔄 正在获取 Paragraph 类型数据...")
    paragraph_dataset = c1.get_dataset_by_type("paragraph")
    
    # 分别保存到三个JSON文件
<<<<<<< HEAD:data_src/load_hf.py
    base_path = "/DRA/data_source/"
=======
    base_path = PROJECT_ROOT / "DRA" / "data_source"
>>>>>>> bed9ec8 (update):src/self_reflection_llm/data_src/load_hf.py
    
    # 保存 LaTeX table 类型数据
    latex_save_path = base_path / "ReLLM_latex_table_full.json"
    c1.save_dataset(latex_table_dataset, latex_save_path)
    
    # 保存 Python code 类型数据
    python_save_path = base_path / "ReLLM_python_code_full.json"
    c1.save_dataset(python_code_dataset, python_save_path)
    
    # 保存 Paragraph 类型数据
    paragraph_save_path = base_path / "ReLLM_paragraph_full.json"
    c1.save_dataset(paragraph_dataset, paragraph_save_path)
    
    # 打印汇总信息
    print("\n📈 数据汇总:")
    print(f"   LaTeX table 类型: {len(latex_table_dataset)} 个样本")
    print(f"   Python code 类型: {len(python_code_dataset)} 个样本")
    print(f"   Paragraph 类型: {len(paragraph_dataset)} 个样本")
    print(f"   总计: {len(latex_table_dataset) + len(python_code_dataset) + len(paragraph_dataset)} 个样本")
