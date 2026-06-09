import os
os.environ["PYTHONUTF8"] = "1"

import torch
import pandas as pd

from datasets import load_dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    EarlyStoppingCallback
)

from peft import LoraConfig
from trl import SFTTrainer

# =====================================================
# MODEL
# =====================================================

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# =====================================================
# TOKENIZER
# =====================================================

print("=" * 50)
print("Loading tokenizer...")
print("=" * 50)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token

print("Tokenizer loaded!")

# =====================================================
# MODEL
# =====================================================

print("=" * 50)
print("Loading model...")
print("=" * 50)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map="auto"
)

model.config.use_cache = False

print("Model loaded!")

# =====================================================
# LOAD DATASET
# =====================================================

print("=" * 50)
print("Loading dataset...")
print("=" * 50)

dataset = load_dataset(
    "json",
    data_files="dataset/finetune_dataset.json"
)["train"]

print(f"Total Dataset: {len(dataset)}")

# =====================================================
# CREATE PROMPT COLUMN
# =====================================================

def create_prompt(example):
    return {
        "text": f"""### Instruction:
{example['instruction']}

### Question:
{example['input']}

### Response:
{example['output']}
"""
    }

dataset = dataset.map(create_prompt)

# =====================================================
# TRAIN / VALID / TEST SPLIT
# =====================================================

print("=" * 50)
print("Splitting dataset...")
print("=" * 50)

split_1 = dataset.train_test_split(
    test_size=0.2,
    seed=42
)

train_dataset = split_1["train"]
temp_dataset = split_1["test"]

split_2 = temp_dataset.train_test_split(
    test_size=0.5,
    seed=42
)

valid_dataset = split_2["train"]
test_dataset = split_2["test"]

print(f"Train      : {len(train_dataset)}")
print(f"Validation : {len(valid_dataset)}")
print(f"Test       : {len(test_dataset)}")

# =====================================================
# LORA CONFIG
# =====================================================

print("=" * 50)
print("Configuring LoRA...")
print("=" * 50)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# =====================================================
# TRAINING ARGUMENTS
# =====================================================

print("=" * 50)
print("Configuring Training Arguments...")
print("=" * 50)

training_args = TrainingArguments(
    output_dir="./lora_output",

    num_train_epochs=3,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,

    learning_rate=2e-4,

    fp16=True,

    optim="paged_adamw_8bit",

    logging_steps=100,

    evaluation_strategy="steps",
    eval_steps=500,

    save_strategy="steps",
    save_steps=500,

    save_total_limit=2,

    load_best_model_at_end=True,

    metric_for_best_model="eval_loss",
    greater_is_better=False,

    report_to="none"
)

# =====================================================
# TRAINER
# =====================================================

print("=" * 50)
print("Preparing Trainer...")
print("=" * 50)

trainer = SFTTrainer(
    model=model,

    train_dataset=train_dataset,
    eval_dataset=valid_dataset,

    dataset_text_field="text",

    peft_config=peft_config,

    max_seq_length=512,

    args=training_args,

    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=3
        )
    ]
)

print("Trainer Ready!")

# =====================================================
# TRAIN
# =====================================================

print("=" * 50)
print("STARTING TRAINING")
print("=" * 50)

trainer.train(
    resume_from_checkpoint="./lora_output/checkpoint-18000"
)

print("=" * 50)
print("TRAINING FINISHED")
print("=" * 50)

# =====================================================
# SAVE TRAINING LOGS
# =====================================================

print("=" * 50)
print("Saving logs...")
print("=" * 50)

logs_df = pd.DataFrame(
    trainer.state.log_history
)

logs_df.to_csv(
    "training_logs.csv",
    index=False
)

print("training_logs.csv saved!")

# =====================================================
# EVALUATE VALIDATION
# =====================================================

print("=" * 50)
print("VALIDATION EVALUATION")
print("=" * 50)

val_metrics = trainer.evaluate()

for k, v in val_metrics.items():
    print(f"{k}: {v}")

# =====================================================
# SAVE MODEL
# =====================================================

SAVE_PATH = "medical_lora_adapter"

print("=" * 50)
print("Saving LoRA Adapter...")
print("=" * 50)

trainer.model.save_pretrained(
    SAVE_PATH
)

tokenizer.save_pretrained(
    SAVE_PATH
)

print("LoRA Adapter Saved!")

# =====================================================
# SAVE TEST SET
# =====================================================

test_dataset.to_json(
    "test_dataset.json"
)

print("test_dataset.json saved!")

# =====================================================
# FINISHED
# =====================================================

print("=" * 50)
print("PROCESS FINISHED")
print("=" * 50)