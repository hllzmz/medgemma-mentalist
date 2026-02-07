import os
import subprocess
from pathlib import Path

import modal

APP_NAME = "medgemma-mentalist-gguf-build-app"

BASE_MODEL = "unsloth/medgemma-27b-text-it"
LORA_MODEL = "hllzmz/medgemma-mentalist"
TARGET_REPO = "hllzmz/medgemma-mentalist-gguf"
QUANTS = ["q4_k_m", "q5_k_m", "q8_0"]

VOLUME = modal.Volume.from_name("medgemma-gguf-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install(
        "git",
        "git-lfs",
        "cmake",
        "build-essential",
        "python3-dev",
        "pkg-config",
        "libcurl4-openssl-dev",
        "ca-certificates",
    )
    .pip_install(
        "torch",
        "transformers>=4.41.0",
        "accelerate",
        "peft",
        "safetensors",
        "huggingface_hub>=0.23.0",
        "hf_transfer",
        "sentencepiece",
        "protobuf",
    )
)

app = modal.App(APP_NAME, image=image)


def run(cmd, cwd=None, env=None):
    print(">>", " ".join(map(str, cmd)))
    subprocess.check_call(list(map(str, cmd)), cwd=cwd, env=env or os.environ.copy())


def ensure_clean_repo(repo_dir: Path):
    status = (
        subprocess.check_output(["git", "status", "--porcelain"], cwd=str(repo_dir))
        .decode("utf-8")
        .strip()
    )
    if status:
        raise RuntimeError(
            "Local repo has uncommitted changes; aborting to avoid data loss. "
            "Please clean or clone a fresh repo directory."
        )


@app.function(
    gpu="H200",
    timeout=60 * 60 * 6,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    volumes={"/vol": VOLUME},
)
def build_and_push():
    work = Path("/vol/work")
    work.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(Path("/vol/hf_home"))
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(Path("/vol/hf_cache"))
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

    merged_dir = work / "merged_hf"
    gguf_dir = work / "gguf_out"
    llama_cpp_dir = work / "llama.cpp"
    repo_dir = work / "repo"

    if merged_dir.exists():
        run(["rm", "-rf", str(merged_dir)])
    if gguf_dir.exists():
        run(["rm", "-rf", str(gguf_dir)])

    merged_dir.mkdir(exist_ok=True)
    gguf_dir.mkdir(exist_ok=True)

    if not llama_cpp_dir.exists():
        run(
            [
                "git",
                "clone",
                "https://github.com/ggerganov/llama.cpp",
                str(llama_cpp_dir),
            ]
        )

    run(["git", "fetch", "--all"], cwd=str(llama_cpp_dir))
    run(["git", "checkout", "master"], cwd=str(llama_cpp_dir))
    run(["git", "reset", "--hard", "origin/master"], cwd=str(llama_cpp_dir))
    run(["git", "rev-parse", "HEAD"], cwd=str(llama_cpp_dir))

    run(["cmake", "-S", ".", "-B", "build", "-DLLAMA_CURL=OFF"], cwd=str(llama_cpp_dir))
    run(["cmake", "--build", "build", "-j"], cwd=str(llama_cpp_dir))

    merge_script = r"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = os.environ["BASE_MODEL"]
lora = os.environ["LORA_MODEL"]
out = os.environ["MERGED_DIR"]

tok = AutoTokenizer.from_pretrained(base, use_fast=True)

model = AutoModelForCausalLM.from_pretrained(
    base,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

model = PeftModel.from_pretrained(model, lora)
model = model.merge_and_unload()


current_embedding_size = model.get_input_embeddings().weight.shape[0]
tokenizer_size = len(tok)

print(f"DEBUG: Tokenizer Size: {tokenizer_size}, Model Embedding Size: {current_embedding_size}")

if current_embedding_size != tokenizer_size:
    model.resize_token_embeddings(tokenizer_size)

model.config.vocab_size = tokenizer_size

tok.save_pretrained(out)
model.save_pretrained(out, safe_serialization=True)
print("Merged saved to", out)
"""
    env = os.environ.copy()
    env["BASE_MODEL"] = BASE_MODEL
    env["LORA_MODEL"] = LORA_MODEL
    env["MERGED_DIR"] = str(merged_dir)
    run(["python", "-c", merge_script], env=env)

    # Update converter mappings for new/changed tokenizers
    update_py = llama_cpp_dir / "convert_hf_to_gguf_update.py"
    if update_py.exists():
        run(["python", str(update_py), str(merged_dir)], cwd=str(llama_cpp_dir))
    else:
        print("WARNING: convert_hf_to_gguf_update.py not found; skipping.")

    # Convert
    f16_gguf = gguf_dir / "medgemma-mentalist-f16.gguf"
    convert_py = llama_cpp_dir / "convert_hf_to_gguf.py"
    run(
        [
            "python",
            str(convert_py),
            str(merged_dir),
            "--outfile",
            str(f16_gguf),
            "--outtype",
            "f16",
        ],
        cwd=str(llama_cpp_dir),
    )

    print("Conversion OK; continue with quantize + push...")

    # Quantize
    candidates = [
        llama_cpp_dir / "build" / "bin" / "llama-quantize",
        llama_cpp_dir / "build" / "bin" / "quantize",
        llama_cpp_dir / "build" / "llama-quantize",
        llama_cpp_dir / "build" / "quantize",
    ]
    quant_bin = next((p for p in candidates if p.exists()), None)
    if quant_bin is None:
        raise RuntimeError(
            "Could not find llama.cpp quantize binary. Checked:\n"
            + "\n".join(str(p) for p in candidates)
        )

    quant_files = []
    for q in QUANTS:
        out = gguf_dir / f"medgemma-mentalist-{q}.gguf"
        quant_files.append(out)
        run([str(quant_bin), str(f16_gguf), str(out), q])

    # Clone/create HF model repo and push ONLY quant files
    push_script = r"""
import os
from huggingface_hub import HfApi, Repository

repo_id = os.environ["TARGET_REPO"]
local_dir = os.environ["REPO_DIR"]
token = os.environ["HF_TOKEN"]

api = HfApi(token=token)

try:
    api.repo_info(repo_id=repo_id, repo_type="model")
except Exception:
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)

repo = Repository(local_dir=local_dir, clone_from=repo_id, repo_type="model", token=token)
print("Repo ready:", repo_id)
"""
    env2 = os.environ.copy()
    env2["TARGET_REPO"] = TARGET_REPO
    env2["REPO_DIR"] = str(repo_dir)

    index_lock = repo_dir / ".git" / "index.lock"
    if index_lock.exists():
        index_lock.unlink()

    if not (repo_dir / ".git").exists():
        run(["python", "-c", push_script], env=env2)

    run(["git", "lfs", "install"], cwd=str(repo_dir))
    ensure_clean_repo(repo_dir)
    run(["git", "pull", "--rebase"], cwd=str(repo_dir))

    for p in quant_files:
        dest = repo_dir / p.name
        if dest.exists():
            dest.unlink()
        run(["cp", str(p), str(dest)])

    readme = repo_dir / "README.md"
    readme.write_text(
        f"""# {TARGET_REPO}

GGUF quantizations for:

- Base model: `{BASE_MODEL}`
- LoRA adapter: `{LORA_MODEL}` (merged into base)

## Quantizations
- `medgemma-mentalist-q4_k_m.gguf`
- `medgemma-mentalist-q5_k_m.gguf`
- `medgemma-mentalist-q8_0.gguf`
""",
        encoding="utf-8",
    )

    run(["git", "add", "."], cwd=str(repo_dir))
    run(["git", "status"], cwd=str(repo_dir))
    try:
        run(
            ["git", "commit", "-m", "Add GGUF quantizations (q4_k_m, q5_k_m, q8_0)"],
            cwd=str(repo_dir),
        )
    except subprocess.CalledProcessError:
        print("No changes to commit.")
    run(["git", "push"], cwd=str(repo_dir))

    print("Done. Pushed quant GGUF files to", TARGET_REPO)


@app.local_entrypoint()
def main():
    build_and_push.remote()
