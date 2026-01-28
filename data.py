import json
from datasets import Dataset, load_dataset, concatenate_datasets
from collections import Counter
## /c23030/ckj/code/vlm/dataset/Satori
# from torch.utils.data import Dataset
import re


class Alpaca(Dataset):
    def __init__(self, num_samples=None) -> None:
        self.num_samples = num_samples
        self.dataset = load_dataset("tatsu-lab/alpaca_eval")["eval"]

        if self.num_samples is not None:
            self.dataset = self.dataset.select(range(self.num_samples))

    def _insert_continue_tokens(self, text: str):
        if not text or len(text.strip()) == 0:
            return text, 0
        total_len = len(text)

        # -------- 1. 在第一句话后插入 --------
        sentences = re.split(r'(?<=[.!?])\s+', text, maxsplit=1)
        if len(sentences) == 1:
            first_part = sentences[0]
            rest_part = ""
        else:
            first_part, rest_part = sentences
        first_part = first_part + "\n<|continue|>\n"

        # -------- 2. 在剩余部分均匀插入3个 --------
        if len(rest_part) > 0:
            chunk_size = len(rest_part) // 4
            new_rest = ""
            for i in range(3):
                new_rest += rest_part[i*chunk_size:(i+1)*chunk_size]
                new_rest += "<|continue|>"
            new_rest += rest_part[3*chunk_size:]
        else:
            new_rest = ""
        new_text = first_part + new_rest
        return new_text, total_len

    def get_dataset(self):
        self.dataset = self.dataset.map(
            lambda row: self._process_row(row),
            remove_columns=self.dataset.column_names
        )
        return self.dataset

    def _process_row(self, row):
        new_output, length = self._insert_continue_tokens(row["output"])

        return {
            "question": row["instruction"],
            "label": new_output
        }


class Math(Dataset):
    def __init__(self, num_samples=None) -> None:  # 添加num_samples参数
        self.num_samples = num_samples
        self.dataset = load_dataset("/c23030/ckj/code/vlm/dataset/Satori")["train"]

    def get_dataset(self):
        TARGET_TYPES = ["type_III_I_data", "type_II_I_data", "type_II_II_data"]
        MAX_SAMPLES_PER_TYPE = self.num_samples
        
        queries = []
        reflect_answers = []
        
        for t in TARGET_TYPES:
            filtered = self.dataset.filter(lambda x: x["type"] == t)
            
            if len(filtered) > 0:
                n_samples = min(MAX_SAMPLES_PER_TYPE, len(filtered))
                for i in range(n_samples):
                    item = filtered[i]
                    queries.append(item["query"])
                    reflect_answers.append(item["response"])
        return Dataset.from_dict({
            "query": queries,
            "reflect_answer": reflect_answers
        })
        
class ReflectDataset:
    def __init__(self) -> None:
        self.dataset = None
        
    def get_dataset(self):
        data = []
        
        # 定义五个文件路径
        files = [
            "/results/DRA_processed/GPT_ReLLM(code)310_filtered_attack_with_reflections.jsonl",
            "/results/DRA_processed/GPT_ReLLM(paragraph)470_filtered_attack_with_reflections.jsonl",
            "/results/DRA_processed/GPT_ReLLM(table)440_filtered_attack_with_reflections.jsonl",
            "/data_src/DRA_safe_imitation_data_1k_complete.json",
            "/data_src/DrAttack_safe_imitation_data_500_complete.json",
            "/data_src/DrAttack(benign)_safe_imitation_data_300_complete.json",
            "/data_src/DRA(benign)_safe_imitation_data_200_complete.json",
            "/data_src/GPT_reflect/wrong-correct.json",
            "/data_src/GPT_reflect/correct-correct.json"
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
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    for i, item in enumerate(json_data[:299]):
                        data.append({
                            "question": item["query"],
                            "label": item["reflect_answer"]
                        })
        dataset = Dataset.from_list(data).shuffle(seed=42)
        print(f"Dataset loaded! {len(dataset)} samples")

        math = Math(num_samples=100)
        m = math.get_dataset()

        alpaca = Alpaca(num_samples=200)
        a = alpaca.get_dataset()

        combined_dataset = concatenate_datasets([dataset, m, a])
        # combined_dataset = dataset

        self.dataset = combined_dataset.train_test_split(test_size=0.1, seed=42)
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


#################################################### Testing Data ################################################
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
                "label": row["category"]
            })
        return self.dataset

## https://huggingface.co/datasets/allenai/WildChat
## refer to: https://huggingface.co/datasets/ToxicityPrompts/PolygloToxicityPrompts
## select 500
## 529k
class WildChat(Dataset):
    def __init__(self, num_samples=None) -> None:  # 添加num_samples参数
        self.num_samples = num_samples
        self.dataset = load_dataset("ToxicityPrompts/PolygloToxicityPrompts", "wildchat-en")["wildchat"].filter(lambda x: x["prompt_toxicity"] >=0.5).select(range(203))
    def get_dataset(self):
        self.dataset = self.dataset.map(
            lambda row: {
                "question": row["prompt"],
                "label": row["prompt_toxicity"]
            },
            remove_columns=self.dataset.column_names
        )
        return self.dataset

## Do Not Answer: https://huggingface.co/datasets/LibrAI/do-not-answer
## 939 prompts
class DoNotAnswer(Dataset):
    def __init__(self, num_samples=None) -> None:  # 添加num_samples参数
        self.num_samples = num_samples
        self.dataset = load_dataset("LibrAI/do-not-answer")["train"].select(range(self.num_samples))
    def get_dataset(self):
        return self.dataset.select_columns(["question", "types_of_harm"]).rename_column("types_of_harm", "label")


################################## General data ######################################

### SimpleQA: https://huggingface.co/datasets/basicv8vc/SimpleQA
## 4.33k
class SimpleQA(Dataset):
    def __init__(self, num_samples=None) -> None:  # 添加num_samples参数
        self.num_samples = num_samples
        self.dataset = load_dataset("basicv8vc/SimpleQA")["test"].select(range(200))
    def get_dataset(self):
        self.dataset = self.dataset.map(
            lambda row: {
                "question": row["problem"],
                "label": row["answer"]
            })
        return self.dataset


## GSM8K: https://huggingface.co/datasets/openai/gsm8k
## 1.32k 
class GSM8k(Dataset):
    def __init__(self, num_samples=None) -> None:  # 添加num_samples参数
        self.num_samples = num_samples
        self.dataset = load_dataset("openai/gsm8k",'main')["train"].select(range(self.num_samples))
    def get_dataset(self):
        return self.dataset.select_columns(["question", "answer"]).rename_column("answer", "label")

## Adv_GLUE: https://huggingface.co/datasets/AI-Secure/adv_glue/viewer/adv_qqp
tasks = ['sst2', 'qqp', 'mnli', 'mnli-mm', 'qnli', 'rte']
task_to_keys = {
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
}
class Adv_GLUE(Dataset):
    def __init__(self, num_samples=None) -> None:  # 添加num_samples参数
        self.num_samples = num_samples
        self.dataset = load_dataset("AI-Secure/adv_glue", "adv_qnli")
        breakpoint()
    def get_dataset(self):
        return self.dataset.select_columns(["question", "sentence"]).rename_column("sentence", "label")






# 使用示例
if __name__ == "__main__":
    # c1 = XSTest(num_samples=100)
    # ds1 = c1.get_dataset()    
    # c2 = StrongReject(num_samples=100)
    # ds2 = c2.get_dataset()

    # c3 = WildChat(num_samples=100)
    # ds3 = c3.get_dataset()    

    # c3 = Math(num_samples=100)
    # ds3 = c3.get_dataset()    
    c3 = Alpaca(num_samples=100)
    ds3 = c3.get_dataset()    
    breakpoint()


    # c4 = DoNotAnswer(num_samples=100)
    # ds4 = c4.get_dataset()    

    # c5 = GSM8k(num_samples=200)
    # ds5 = c5.get_dataset()

    # c5 = BigBench(num_samples=200)
    # ds5 = c5.get_dataset()

    # dataset = ReLLM()
    # result = dataset.get_dataset(
    #     save_path="selected_data_300_per_category.json",
    #     samples_per_category=300
    # )
    # print("\n前5个样本预览:")
    # for i in range(min(5, len(result))):
    #     print(f"{i+1}. [{result[i]['label']}] {result[i]['question'][:80]}...")