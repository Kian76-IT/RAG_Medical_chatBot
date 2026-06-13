import pandas as pd
import numpy as np
from setting import DATA_PATH, TOP_K
from primary.load import load_data
# models
from models.bert import BertModel
from models.roberta import RobertaModel
from models.distilbert import DistilBertModel
from models.albert import AlbertModel
# rag
from rag.retriever import Retriever
# metrics
from evaluation.eval_metrics import (
    precision_at_k,
    recall_at_k,
    f1_score,
    accuracy_at_k
)


# LOAD DATA
print("Loading dataset...")
df, texts = load_data(DATA_PATH)

print("Dataset loaded:", df.shape)


# GROUND TRUTH (SIMPLE)
# asumsi: setiap query cocok dengan dirinya sendiri
ground_truth = [[i] for i in range(len(texts))]


# MODELS
models = {
    "BERT": BertModel(),
    "RoBERTa": RobertaModel(),
    "DistilBERT": DistilBertModel(),
    "ALBERT": AlbertModel()
}

results = []


# EXPERIMENT LOOP 
for name, model in models.items():
    print(f"\nRunning model: {name}")
    # 1. EMBEDDING
    embeddings = model.encode(texts)
    # 2. BUILD RETRIEVER
    retriever = Retriever(embeddings)
    total_p, total_r, total_f1, total_acc = 0, 0, 0, 0
    # batasi sample biar cepat (ubah kalau mau full)
    SAMPLE_SIZE = min(200, len(texts))
    for i in range(SAMPLE_SIZE):
        query = texts[i]
        # 3. QUERY EMBEDDING
        query_vec = model.encode([query])
        # 4. SEARCH
        indices = retriever.search(query_vec, TOP_K)[0]
        # 5. EVALUATION
        p = precision_at_k(ground_truth[i], indices)
        r = recall_at_k(ground_truth[i], indices)
        f1 = f1_score(p, r)
        acc = accuracy_at_k(ground_truth[i], indices)

        total_p += p
        total_r += r
        total_f1 += f1
        total_acc += acc

    # AVERAGE RESULT
    n = SAMPLE_SIZE

    avg_p = total_p / n
    avg_r = total_r / n
    avg_f1 = total_f1 / n
    avg_acc = total_acc / n

    results.append([
        name,
        avg_acc,
        avg_p,
        avg_r,
        avg_f1
    ])

    print(f"{name} DONE")
    print(f"Accuracy: {avg_acc:.4f} | Precision: {avg_p:.4f} | Recall: {avg_r:.4f} | F1: {avg_f1:.4f}")


# SAVE RESULT
result_df = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "Precision", "Recall", "F1"]
)

result_df.to_csv("results/results.csv", index=False)

print("\nFinal Results:")
print(result_df)
print("\nSaved to results/results.csv")