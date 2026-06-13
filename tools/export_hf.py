import os
from huggingface_hub import HfApi, create_repo

def push_to_hub(repo_id="username/coolm-42M", token=None):
    if not token:
        token = os.environ.get("HF_TOKEN")
    if not token:
        print("Please set HF_TOKEN environment variable.")
        return

    api = HfApi(token=token)
    
    print(f"Creating repository {repo_id}...")
    try:
        create_repo(repo_id, exist_ok=True, token=token)
    except Exception as e:
        print(f"Error creating repo: {e}")
        return

    print("Uploading model weights...")
    if os.path.exists("checkpoints/best_model.pt"):
        api.upload_file(
            path_or_fileobj="checkpoints/best_model.pt",
            path_in_repo="best_model.pt",
            repo_id=repo_id
        )
    
    print("Uploading tokenizer...")
    if os.path.exists("data/tokenizer.json"):
        api.upload_file(
            path_or_fileobj="data/tokenizer.json",
            path_in_repo="tokenizer.json",
            repo_id=repo_id
        )

    print("Uploading ONNX model...")
    if os.path.exists("docs/model.onnx"):
        api.upload_file(
            path_or_fileobj="docs/model.onnx",
            path_in_repo="model.onnx",
            repo_id=repo_id
        )

    print("Uploading README.md (Model Card)...")
    if os.path.exists("README.md"):
        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=repo_id
        )

    print("Uploading assets...")
    if os.path.exists("assets"):
        for f in os.listdir("assets"):
            file_path = os.path.join("assets", f)
            if os.path.isfile(file_path):
                print(f"Uploading asset: {f}...")
                api.upload_file(
                    path_or_fileobj=file_path,
                    path_in_repo=f"assets/{f}",
                    repo_id=repo_id
                )

    print("Done! Model pushed to HuggingFace.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="your_username/coolm-42M", help="HuggingFace repo ID")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace token")
    args = parser.parse_args()
    push_to_hub(repo_id=args.repo_id, token=args.token)
