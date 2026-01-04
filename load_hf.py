from datasets import load_dataset
import json,os,shutil
from huggingface_hub import hf_hub_download,list_repo_files,snapshot_download

# repo_id = "walledai/HarmBench"
# repo_id = "lmsys/toxic-chat"
# revision = None
# local_directory = f"/c23030/ckj/code/vlm/jailbreaking/DRA/data/{repo_id}"
# # Download the repository files
# snapshot_download(
#     repo_id=repo_id,
#     repo_type="dataset",
#     revision=revision,
#     local_dir=local_directory,
#     local_dir_use_symlinks=False,  # Avoid symlinks for easier file handling
# )
# print(f"Files downloaded to: {local_directory}")


repo_id = "cais/HarmBench-Llama-2-13b-cls"
revision = None
# local_directory = f"/c23030/ckj/code/vlm/jailbreaking/DRA/data/{repo_id}"
# Download the repository files
snapshot_download(
    repo_id=repo_id,
    repo_type="model",
    revision=revision,
    # local_dir=local_directory,
    # local_dir_use_symlinks=False,  # Avoid symlinks for easier file handling
)
# print(f"Files downloaded to: {local_directory}")