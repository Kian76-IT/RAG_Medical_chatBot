import pandas as pd
import re
import os


# =========================
# CONFIG
# =========================

DATA_PATH = "RAG_Medical_chatBot/dataset/fix_dataset.csv"
SAVE_DIR = "RAG_Medical_chatBot/dataset/"

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================
# LOAD DATA
# =========================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH)

print("Original shape:", df.shape)

# =========================
# CLEANING FUNCTION
# =========================

def clean_text(text):

    text = str(text).lower()

    # remove weird encoding artifacts
    text = re.sub(r'[•â€¢]', '', text)

    # keep punctuation penting
    text = re.sub(r"[^a-z0-9\s.,!?'\-]", '', text)

    # normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text

# =========================
# TEXT CLEANING
# =========================

df['question'] = df['question'].apply(clean_text)
df['answer'] = df['answer'].apply(clean_text)

print("Text cleaning done")

# =========================
# HANDLE NULL
# =========================

print("Null values:")
print(df.isnull().sum())

df.dropna(inplace=True)

# =========================
# HANDLE DUPLICATES
# =========================

duplicates = df.duplicated().sum()

print("Total duplicates:", duplicates)

df.drop_duplicates(inplace=True)

# reset index
df.reset_index(drop=True, inplace=True)

print("After cleaning:", df.shape)

# =========================
# FEATURE ENGINEERING
# =========================

# retrieval text
df['text'] = df['question'] + " " + df['answer']

# context for RAG
df['context'] = df['answer']

# optional statistics
df['q_len'] = df['question'].apply(len)
df['a_len'] = df['answer'].apply(len)

# =========================
# SAVE OUTPUT
# =========================

OUTPUT_PATH = f"{SAVE_DIR}/clean_dataset.csv"

df.to_csv(OUTPUT_PATH, index=False)

print(f"Clean dataset saved at: {OUTPUT_PATH}")
print("Final shape:", df.shape)