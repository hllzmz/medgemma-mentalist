import os, json
import modal


CONFIG = {
    "model_name": "unsloth/medgemma-27b-text-it",
    "train_path": "data/train.jsonl",
    "val_path": "data/val.jsonl",
    "output_dir": "/data/outputs/medgemma-mentalist-lora",
    "max_seq_len": 4096,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0,
    "learning_rate": 2e-4,
    "warmup_steps": 10,
    "weight_decay": 0.01,
    "num_epochs": 2,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 8,
    "logging_steps": 10,
    "eval_steps": 200,
    "save_steps": 100,
    "max_steps": -1,
    "seed": 42,
    "bf16": True,
    "gradient_checkpointing": True,
}

# Modal setup 
image = modal.Image.debian_slim().pip_install(
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "trl",
    "accelerate",
    "datasets",
    "unsloth",
    "bitsandbytes",
    "jsonlines",
    "wandb",
)

app = modal.App("unsloth-medgemma-mentalist-lora")

volume = modal.Volume.from_name("data", create_if_missing=False)


@app.function(
    image=image,
    gpu="H200",
    timeout=60 * 60 * 8,
    volumes={"/data": volume},
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("huggingface-secret"),
    ],
)
def run():
    import random
    import torch
    import wandb
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from transformers import TrainingArguments
    from trl import SFTTrainer

    cfg = CONFIG
    os.makedirs(cfg["output_dir"], exist_ok=True)

    # Repro 
    torch.manual_seed(cfg["seed"])
    random.seed(cfg["seed"])

    # Track with wandb
    wandb.init(project="unsloth-medgemma-lora", config=cfg)

    # Load base model & tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["model_name"],
        max_seq_length=cfg["max_seq_len"],
        load_in_4bit=False,
        dtype=torch.bfloat16 if cfg["bf16"] else torch.float16,
    )

    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"

    # Apply LoRA
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora_r"],
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=cfg["lora_alpha"],
        lora_dropout=cfg["lora_dropout"],
        bias="none",
        use_rslora=False,
    )
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable:  {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    # Data loading
    def read_jsonl(path):
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
        return rows

    train_rows = read_jsonl(cfg["train_path"])
    if os.path.exists(cfg["val_path"]):
        val_rows = read_jsonl(cfg["val_path"])
    else:
        split = int(0.9 * len(train_rows))
        val_rows = train_rows[split:]
        train_rows = train_rows[:split]

    # Formatting to text
    def format_example(row):
        convs = row["conversations"]

        # Ensure at least one assistant turn
        if not any(c["role"] == "assistant" for c in convs):
            return None
        # Build a prompt-completion pair using chat template
        text = tokenizer.apply_chat_template(
            convs, tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    train_texts = [r for r in (format_example(x) for x in train_rows) if r]
    val_texts = [r for r in (format_example(x) for x in val_rows) if r]

    train_ds = Dataset.from_list(train_texts)
    val_ds = Dataset.from_list(val_texts)

    # Training args
    total_train_steps = None
    if cfg["max_steps"] > 0:
        total_train_steps = cfg["max_steps"]

    training_args = TrainingArguments(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=cfg["gradient_accumulation_steps"],
        num_train_epochs=cfg["num_epochs"],
        learning_rate=cfg["learning_rate"],
        warmup_steps=cfg["warmup_steps"],
        weight_decay=cfg["weight_decay"],
        logging_steps=cfg["logging_steps"],
        eval_steps=cfg["eval_steps"],
        evaluation_strategy="steps",
        save_steps=cfg["save_steps"],
        save_total_limit=2,
        bf16=cfg["bf16"],
        gradient_checkpointing=cfg["gradient_checkpointing"],
        max_steps=cfg["max_steps"],
        report_to="wandb",
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        dataset_text_field="text",
        max_seq_length=cfg["max_seq_len"],
        packing=True,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        seed=3407,
        args=training_args,
    )

    trainer.train()

    # Save adapter
    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

    # Finish wandb run
    wandb.finish()

    # Quick sanity eval
    samples = [
        "I have been feeling very down lately and can't seem to find joy in anything. What should I do?",
        "How could I encourage someone to seek professional help without making a diagnosis?",
    ]
    model.eval()
    FastLanguageModel.for_inference(model)
    for prompt in samples:
        chat = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = tokenizer([text], return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=200,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        print("\n--- SAMPLE ---")
        print(decoded)


if __name__ == "__main__":
    run.remote()
