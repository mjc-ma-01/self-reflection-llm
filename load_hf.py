from datasets import load_dataset
import json,os,shutil
from huggingface_hub import hf_hub_download,list_repo_files,snapshot_download
from transformers import AutoModelForCausalLM,AutoTokenizer
# # repo_id = "walledai/HarmBench"
# # repo_id = "lmsys/toxic-chat"
# # revision = None
# # local_directory = f"/DRA/data/{repo_id}"
# # # Download the repository files
# # snapshot_download(
# #     repo_id=repo_id,
# #     repo_type="dataset",
# #     revision=revision,
# #     local_dir=local_directory,
# #     local_dir_use_symlinks=False,  # Avoid symlinks for easier file handling
# # )
# # print(f"Files downloaded to: {local_directory}")


# repo_id = "cais/HarmBench-Llama-2-13b-cls"
# revision = None
# # local_directory = f"/DRA/data/{repo_id}"
# # Download the repository files
# snapshot_download(
#     repo_id=repo_id,
#     repo_type="model",
#     revision=revision,
#     # local_dir=local_directory,
#     # local_dir_use_symlinks=False,  # Avoid symlinks for easier file handling
# )
# # print(f"Files downloaded to: {local_directory}")

# 加载本地模型
model_path = "/c23030/ckj/code/vlm/models/reflect_results/model:sft_mllm_llama_8b/train:reflect_cot_1k_w400_benign_lr3e-6_epoch5_warmup0.1/checkpoint-best"

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 直接推送到 Hub
from huggingface_hub import whoami
user_info = whoami()
print(f"你的用户名: {user_info['name']}")

# 改为使用你的用户名

repo_id = f"{user_info['name']}/llama-8b-reflect-sft"  
repo_id = f"{user_info['name']}/llama-8b-reflect-GRPO"  
# 推送到 Hugging Face Hub
model.push_to_hub(repo_id, commit_message="Upload trained model")
tokenizer.push_to_hub(repo_id, commit_message="Upload tokenizer")

print(f"✅ 模型已上传到: https://huggingface.co/{repo_id}")



#=======================AILAb 上传===========================
model_path = "/mnt/shared-storage-user/lixiangtian/workspace/safe_research/verl-safe/checkpoints/safe_alpaca_reflect/segmented_math_krystal7"

model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

from huggingface_hub import whoami
user_info = whoami()
print(f"你的用户名: {user_info['name']}")

# repo_id = f"{user_info['name']}/llama-8b-reflect-sft"  
repo_id = f"{user_info['name']}/llama-8b-reflect-GRPO"  
model.push_to_hub(repo_id, commit_message="Upload trained model")
tokenizer.push_to_hub(repo_id, commit_message="Upload tokenizer")

print(f"✅ 模型已上传到: https://huggingface.co/{repo_id}")