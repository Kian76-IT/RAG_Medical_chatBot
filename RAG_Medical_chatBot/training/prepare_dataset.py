import pandas as pd
import json


# LOAD DATASET
df = pd.read_csv("dataset/clean_dataset.csv")
data = []

# CONVERT FORMAT
for _, row in df.iterrows():
    sample = {
        "instruction": "Answer the medical question about diabetes.",
        "input": row["question"],
        "output": row["answer"]
    }
    data.append(sample)

# SAVE JSON
with open("dataset/finetune_dataset.json", "w") as f:
    json.dump(data, f, indent=4)

print("Fine-tuning dataset created!")
print("Total samples:", len(data))