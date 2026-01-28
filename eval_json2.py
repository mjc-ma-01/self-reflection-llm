import json
import os

# 基础路径与文件列表
base_path = "/results/model_answer/llama8b-harm-sft"
file_names = ["strongreject_check.json", "wildchat_check.json", "xstest_harmful_check.json"]
file_paths = [os.path.join(base_path, name) for name in file_names]

print("分析结果：")
print("-" * 60)

# 初始化总计
total_reflect_count = 0
total_harmbench_true = 0

for path in file_paths:
    file_reflect_count = 0
    file_harmbench_true = 0

    try:
        with open(path, 'r', encoding='utf-8') as f:
            # 文件最外层是一个字典
            data = json.load(f)

        # 遍历最外层字典的每一个值 (每个值本身也是一个字典)
        for content in data.values():
            # content["detail"] 是一个列表，遍历其中的每个item
            for item in content.get("detail", []):
                if item.get("harmbench") is True:
                    file_harmbench_true += 1
                    if "<|reflect|>" in item.get("answer", ""):
                        file_reflect_count += 1

        # 计算并打印当前文件的结果
        percentage = (file_reflect_count / file_harmbench_true * 100) if file_harmbench_true > 0 else 0
        print(f"文件：{os.path.basename(path)}")
        print(f"  `harmbench: true` 的条目总数：{file_harmbench_true}")
        print(f"  包含 `<|reflect|>` 标记的条目数：{file_reflect_count}")
        print(f"  占比：{percentage:.2f}%\n")

        total_reflect_count += file_reflect_count
        total_harmbench_true += file_harmbench_true

    except FileNotFoundError:
        print(f"文件未找到：{path}\n")
    except json.JSONDecodeError:
        print(f"JSON解析失败：{path}，请检查文件格式\n")
    except KeyError as e:
        print(f"数据结构异常（缺少键 {e}）：{path}\n")

# 输出总计
if total_harmbench_true > 0:
    print("-" * 60)
    total_percentage = (total_reflect_count / total_harmbench_true * 100)
    print(f"总计（三个文件）：")
    print(f"  `harmbench: true` 的条目总数：{total_harmbench_true}")
    print(f"  包含 `<|reflect|>` 标记的条目总数：{total_reflect_count}")
    print(f"  总计占比：{total_percentage:.2f}%")
else:
    print("未找到任何 `harmbench: true` 的条目。")