import os

# =========================
# FIX WINDOWS UTF-8
# =========================

os.environ["PYTHONUTF8"] = "1"

# =========================
# IMPORTS
# =========================

import torch

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments
)

from peft import (
    LoraConfig
)

from trl import SFTTrainer

# =========================
# MODEL NAME
# =========================

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# =========================
# LOAD TOKENIZER
# =========================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded!")

# =========================
# LOAD MODEL
# =========================

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    model_name,

    trust_remote_code=True,

    device_map="auto",

    torch_dtype=torch.float16
)

print("Model loaded!")

# =========================
# LOAD DATASET
# =========================

print("Loading dataset...")

dataset = load_dataset(
    "json",
    data_files="dataset/finetune_dataset.json"
)

print(dataset)

# =========================
# FORMAT FUNCTION
# =========================

def formatting_func(example):

    text = f"""
### Instruction:
{example['instruction']}

### Question:
{example['input']}

### Response:
{example['output']}
"""

    # IMPORTANT:
    # MUST RETURN LIST
    return [text]

# =========================
# LoRA CONFIG
# =========================

print("Setting LoRA config...")

peft_config = LoraConfig(

    r=16,

    lora_alpha=32,

    lora_dropout=0.05,

    bias="none",

    task_type="CAUSAL_LM"
)

# =========================
# TRAINING ARGUMENTS
# =========================

print("Setting training arguments...")

training_args = TrainingArguments(

    output_dir="./lora_output",

    # training
    num_train_epochs=1,

    per_device_train_batch_size=1,

    gradient_accumulation_steps=4,

    learning_rate=2e-4,

    # mixed precision
    fp16=True,

    # optimizer
    optim="paged_adamw_8bit",

    # logging
    logging_steps=10,

    save_steps=100,

    save_total_limit=2,

    # reporting
    report_to="none"
)

# =========================
# TRAINER
# =========================

print("Preparing trainer...")

trainer = SFTTrainer(

    model=model,

    train_dataset=dataset["train"],

    peft_config=peft_config,

    formatting_func=formatting_func,

    max_seq_length=512,

    args=training_args
)

print("Trainer ready!")

# =========================
# TRAIN
# =========================

print("Starting fine-tuning...")

trainer.train()

print("Training completed!")

# =========================
# SAVE MODEL
# =========================

save_path = "medical_lora_adapter"

print(f"Saving model to {save_path}...")

trainer.model.save_pretrained(save_path)

tokenizer.save_pretrained(save_path)

print("LoRA adapter saved successfully!")