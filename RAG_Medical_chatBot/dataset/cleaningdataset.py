import pandas as pd
import re
import os

# CONFIG
DATA_PATH = "RAG_Medical_chatBot/dataset/fix_dataset.csv"
SAVE_DIR = "RAG_Medical_chatBot/dataset/"

os.makedirs(SAVE_DIR, exist_ok=True)

# 1. LOAD DATA
print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Original shape:", df.shape)
print(df.head())

# 2. CLEANING FUNCTION
def clean_text(text):
    text = str(text).lower()

    # remove encoding artifacts
    text = re.sub(r'[•â€¢]', '', text)

    # remove weird symbols (keep numbers & letters)
    text = re.sub(r'[^a-z0-9\s]', '', text)

    # normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# apply cleaning
df['question'] = df['question'].apply(clean_text)
df['answer'] = df['answer'].apply(clean_text)

print("Cleaning done")

# 3. HANDLE NULL & DUPLICATE
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
df.reset_index(drop=True, inplace=True)

print("After cleaning:", df.shape)

# 4. FEATURE ENGINEERING (UNIVERSAL)

# combine text (untuk retrieval / QA)
df['text'] = df['question'] + " [SEP] " + df['answer']

# context untuk RAG
df['context'] = df['answer']

# optional features (buat analisis / eksperimen)
df['q_len'] = df['question'].apply(len)
df['a_len'] = df['answer'].apply(len)

# 5. SAVE OUTPUT
OUTPUT_PATH = f"{SAVE_DIR}/clean_dataset.csv"
df.to_csv(OUTPUT_PATH, index=False)

print(f"Clean dataset saved at: {OUTPUT_PATH}")
print("Final shape:", df.shape)

print("Before cleaning:", df.shape)

# after cleaning text
df['question'] = df['question'].apply(clean_text)
df['answer'] = df['answer'].apply(clean_text)

print("After text cleaning:", df.shape)

# cek null
print("Null rows:", df.isnull().sum())

# drop null
df.dropna(inplace=True)
print("After dropna:", df.shape)

# cek duplicate
duplicates = df.duplicated().sum()
print("Total duplicates:", duplicates)

df.drop_duplicates(inplace=True)
print("After drop duplicates:", df.shape)