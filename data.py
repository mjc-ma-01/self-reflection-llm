import json
from datasets import Dataset, load_dataset
from collections import Counter

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
        breakpoint()
        self.dataset = self.dataset.map(
            lambda row: {
                "question": row["prompt"],
                "label": row["category"]
            })
        return self.dataset


class ReLLM(Dataset):
    def __init__(self, num_samples=None) -> None:  # 添加num_samples参数
        self.num_samples = num_samples
        self.dataset = load_dataset("Deep1994/ReNeLLM-Jailbreak")["train"]

    def _classify(self, text):
        text = text.strip()
        if text.startswith("The latex table"):
            return "table"
        elif text.startswith("The Python code"):
            return "code"
        elif text.startswith("Please supplement the second"):
            return "paragraph"
        return "unknown"
    
    def get_dataset(self, save_path=None, samples_per_category=500):
        processed = self.dataset.map(
            lambda x: {
                "question": x["nested_prompt"],
                "label": self._classify(x["nested_prompt"])
            },
            remove_columns=self.dataset.column_names
        )
        if self.num_samples:
            processed = processed.select(range(min(self.num_samples, len(processed))))
        
        counts = Counter(processed["label"])
        print("统计结果:", dict(counts))
        
        selected = []
        start_idx, end_idx = 1000, 1500

        category_samples = {"table": [], "code": [], "paragraph": []}
        
        for idx, item in enumerate(processed):
            label = item["label"]
            if label in category_samples and len(category_samples[label]) < samples_per_category:
                category_samples[label].append({
                    "question": item["question"],
                    "label": label
                })
        selected = []
        for label, samples in category_samples.items():
            selected.extend(samples)
            print(f"类别 '{label}': 保存了 {len(samples)} 条数据")
        
        # 保存为JSON
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(selected, f, ensure_ascii=False, indent=2)
            print(f"\n总计保存 {len(selected)} 条数据到: {save_path}")
        
        return processed

# 使用示例
if __name__ == "__main__":
    c1 = XSTest(num_samples=100)
    ds1 = c1.get_dataset()    
    c2 = StrongReject(num_samples=100)
    ds2 = c2.get_dataset()

    # dataset = ReLLM()
    # result = dataset.get_dataset(
    #     save_path="selected_data_300_per_category.json",
    #     samples_per_category=300
    # )
    # print("\n前5个样本预览:")
    # for i in range(min(5, len(result))):
    #     print(f"{i+1}. [{result[i]['label']}] {result[i]['question'][:80]}...")