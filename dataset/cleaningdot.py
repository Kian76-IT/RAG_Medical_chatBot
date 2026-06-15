import pandas as pd

df = pd.read_csv("dataset/diabetes_merged_dataset.csv")

def clean_text(text):
    text = str(text)
    text = text.replace("â€¢", "")
    text = text.replace("•", "")
    return text

df['question'] = df['question'].apply(clean_text)
df['answer'] = df['answer'].apply(clean_text)

df.to_csv("dataset/fix_dataset_fix.csv", index=False)

print("Bullet removed!")