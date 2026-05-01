import pandas as pd
import numpy as np
import os
from transformers import AlbertTokenizer

# CONFIG (SUDAH DIPERBAIKI)
DATA_PATH = "RAG_Medical_chatBot/dataset/clean_dataset.csv"
SAVE_DIR = "RAG_Medical_chatBot/albert/"
MODEL_NAME = "albert-base-v2"
MAX_LEN = 128

os.makedirs(SAVE_DIR, exist_ok=True)

# 1. LOAD DATA
print("Loading clean dataset...")
df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print(df.head())

texts = df['text'].astype(str).tolist()

# =========================
# 2. LOAD TOKENIZER
# =========================
print("Loading ALBERT tokenizer...")
tokenizer = AlbertTokenizer.from_pretrained(MODEL_NAME)

# =========================
# 3. TOKENIZATION
# =========================
print("Tokenizing...")

tokens = tokenizer(
    texts,
    padding='max_length',
    truncation=True,
    max_length=MAX_LEN,
    return_tensors='np'
)

input_ids = tokens['input_ids']
attention_masks = tokens['attention_mask']

print("Done")
print("Shape:", input_ids.shape)

# =========================
# 4. SAVE
# =========================
np.save(f"{SAVE_DIR}/albert_input_ids.npy", input_ids)
np.save(f"{SAVE_DIR}/albert_attention_masks.npy", attention_masks)

print("Saved to:", SAVE_DIR)