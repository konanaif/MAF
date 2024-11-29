import os, sys
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    prompt_download_path = os.path.join(
        "models--esyoon--INTapt-HuBERT-large-coraal-prompt-generator"
    )
    if os.path.exists(prompt_download_path):
        print("Prompt generator model already exists")
    snapshot_download(
        "facebook/hubert-large-ls960-ft", repo_type="model", cache_dir="."
    )
    snapshot_download(
        "esyoon/INTapt-HuBERT-large-coraal-prompt-generator",
        repo_type="model",
        cache_dir=".",
    )
