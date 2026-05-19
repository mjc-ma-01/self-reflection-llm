import argparse

from huggingface_hub import snapshot_download

from ..paths import project_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model or dataset into a repo-local cache.")
    parser.add_argument("--repo_id", default="thu-ml/STAIR-Llama-3.1-8B-DPO-3")
    parser.add_argument("--repo_type", default="model", choices=["model", "dataset", "space"])
    parser.add_argument("--revision", default=None)
    parser.add_argument("--local_dir", default=None, help="Defaults to models/<repo_id with slash replaced>.")
    args = parser.parse_args()

    local_dir = args.local_dir or str(project_path("models", args.repo_id.replace("/", "--")))
    snapshot_download(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        revision=args.revision,
        local_dir=local_dir,
    )
    print(f"Files downloaded to: {local_dir}")


if __name__ == "__main__":
    main()
