import os
from pathlib import Path

import modal

APP_NAME = "medgemma-mentalist-gguf-http-upload"

TARGET_REPO = "hllzmz/medgemma-mentalist-gguf"
QUANTS = ["q4_k_m", "q5_k_m", "q8_0"]

VOLUME = modal.Volume.from_name("medgemma-gguf-cache", create_if_missing=True)

image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub>=0.23.0",
)

app = modal.App(APP_NAME, image=image)


@app.function(
    timeout=60 * 60 * 6,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/vol": VOLUME},
)
def push_only_http():
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set.")

    api = HfApi(token=token)

    gguf_dir = Path("/vol/work/gguf_out")
    files = [gguf_dir / f"medgemma-mentalist-{q}.gguf" for q in QUANTS]

    missing = [str(p) for p in files if not p.exists()]
    if missing:
        raise RuntimeError("Missing GGUF files:\n" + "\n".join(missing))

    # Ensure repo exists
    try:
        api.repo_info(repo_id=TARGET_REPO, repo_type="model")
    except Exception:
        api.create_repo(repo_id=TARGET_REPO, repo_type="model", exist_ok=True)

    # Upload
    for p in files:
        print(f"Uploading {p.name} from {p} ...")
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=p.name,
            repo_id=TARGET_REPO,
            repo_type="model",
            commit_message=f"Add {p.name}",
        )

    print("Done. Uploaded to:", f"https://huggingface.co/{TARGET_REPO}")


@app.local_entrypoint()
def main():
    push_only_http.remote()