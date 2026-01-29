import os
from huggingface_hub import snapshot_download


def main() -> None:

    repo_id = "opendatalab/OmniDocBench"
    local_dir = "/root/autodl-tmp/data/raw/OmniDocBench"

    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN is not set. Please export HF_TOKEN before running this script."
        )

    print(f"开始下载数据集 {repo_id} 到 {local_dir}...")

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=local_dir,
        max_workers=8,
        token=hf_token,
    )

    print("下载完成！")

if __name__ == "__main__":
    main()