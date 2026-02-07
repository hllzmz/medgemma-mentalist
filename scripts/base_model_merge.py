import modal


image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "huggingface_hub",
        "bitsandbytes"
    )
)

app = modal.App("medgemma-overwrite-merger", image=image)

BASE_MODEL_ID = "unsloth/medgemma-27b-text-it"
TARGET_REPO_ID = "hllzmz/medgemma-mentalist"

@app.function(
    gpu="H200",
    timeout=7200,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def merge_and_overwrite():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel


    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    # LoRA Adapter
    model = PeftModel.from_pretrained(base_model, TARGET_REPO_ID)

    # Merge adapter into base model
    model = model.merge_and_unload()
    
    model.push_to_hub(
        TARGET_REPO_ID, 
        safe_serialization=True, 
        max_shard_size="5GB",
        commit_message="Merge LoRA adapter into base model"
    )
    
    tokenizer.push_to_hub(TARGET_REPO_ID)


@app.local_entrypoint()
def main():
    merge_and_overwrite.remote()