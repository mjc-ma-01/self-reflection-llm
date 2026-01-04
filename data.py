import json
from datasets import Dataset, load_dataset


import json
from datasets import Dataset

class ReflectDataset:
    def __init__(self) -> None:
        self.dataset = None
        
    def get_dataset(self):
        data = []
        
        # 定义五个文件路径
        files = [
            "/c23030/ckj/code/vlm/jailbreaking/results/DRA_processed/GPT_ReLLM(code)310_filtered_attack_with_reflections.jsonl",
            "/c23030/ckj/code/vlm/jailbreaking/results/DRA_processed/GPT_ReLLM(paragraph)470_filtered_attack_with_reflections.jsonl",
            "/c23030/ckj/code/vlm/jailbreaking/results/DRA_processed/GPT_ReLLM(table)440_filtered_attack_with_reflections.jsonl",
            "/c23030/ckj/code/vlm/jailbreaking/DRA/DRA_safe_imitation_data_1k_complete.json",
            "/c23030/ckj/code/vlm/jailbreaking/DRA/DrAttack_safe_imitation_data_500_complete.json",
            "/c23030/ckj/code/vlm/jailbreaking/DRA/DrAttack(benign)_safe_imitation_data_300_complete.json",
            "/c23030/ckj/code/vlm/jailbreaking/DRA/DRA(benign)_safe_imitation_data_200_complete.json"
        ]
        
        # 处理每个文件，各取前200条
        for file_path in files:
            if file_path.endswith('.jsonl'):
                # 处理JSONL文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= 200:
                            break
                        item = json.loads(line.strip())
                        data.append({
                            "question": item["query"],
                            "label": item["reflect_answer"]
                        })
            else:
                # 处理JSON文件
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    for i, item in enumerate(json_data[:200]):
                        data.append({
                            "question": item["query"],
                            "label": item["reflect_answer"]
                        })
        
        # 创建Dataset
        dataset = Dataset.from_list(data).shuffle(seed=42)
        print(f"Dataset loaded! {len(dataset)} samples")
        
        # 划分训练集和测试集
        self.dataset = dataset.train_test_split(test_size=0.1, seed=42)
        return self.dataset


## For Test
class XSTest(Dataset):
    def __init__(self, num_samples=None) -> None:  # 添加num_samples参数
        self.num_samples = num_samples
        self.dataset = load_dataset("walledai/XSTest")["test"]

    def get_dataset(self):
        self.dataset = self.dataset.map(
            lambda row: {
                "question": row["prompt"],
                "label": row["label"]
            })
        return self.dataset


class StrongReject(Dataset):
    def __init__(self, num_samples=None) -> None:  # 添加num_samples参数
        self.num_samples = num_samples
        self.dataset = load_dataset("walledai/StrongREJECT")["train"]
    def get_dataset(self):
        self.dataset = self.dataset.map(
            lambda row: {
                "question": row["prompt"],
                "label": "unsafe"
            })
        return self.dataset

if __name__ == "__main__":
    c1 = XSTest(num_samples=100)
    ds1 = c1.get_dataset()    
    c2 = StrongReject(num_samples=100)
    ds2 = c2.get_dataset()
