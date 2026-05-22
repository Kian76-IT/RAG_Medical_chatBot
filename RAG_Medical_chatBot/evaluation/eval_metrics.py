import sys
import os

# =========================
# FIX IMPORT PATH
# =========================

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

# =========================
# IMPORTS
# =========================

import numpy as np
import pandas as pd

from setting import DATA_PATH, MODELS
from primary.load import load_data
from models.base import BaseEmbeddingModel
from rag.retriever import Retriever



# =========================
# LOAD DATA
# =========================

print("Loading dataset...")

df, texts = load_data(DATA_PATH)

print("Dataset loaded!")

# =========================
# EVALUATION QUERIES
# =========================

# gunakan seluruh question sebagai query

queries = df["question"].tolist()

# ground truth:
# document asli dari dataset

ground_truth = list(range(len(df)))

# =========================
# METRICS
# =========================

def precision_at_k(retrieved, relevant, k):

    retrieved_k = retrieved[:k]

    relevant_count = sum(
        [1 for doc in retrieved_k if doc == relevant]
    )

    return relevant_count / k


def recall_at_k(retrieved, relevant, k):

    retrieved_k = retrieved[:k]

    relevant_count = sum(
        [1 for doc in retrieved_k if doc == relevant]
    )

    return relevant_count


def reciprocal_rank(retrieved, relevant):

    for rank, doc_id in enumerate(retrieved, start=1):

        if doc_id == relevant:

            return 1 / rank

    return 0


def hit_rate(retrieved, relevant):

    return 1 if relevant in retrieved else 0

# =========================
# RESULT STORAGE
# =========================

all_results = []

# =========================
# LOOP ALL MODELS
# =========================

for model_name, model_path in MODELS.items():

    print("\n=========================")
    print(f"Evaluating {model_name}")
    print("=========================")

    # =========================
    # LOAD MODEL
    # =========================

    embedding_model = BaseEmbeddingModel(
        model_path
    )

    # =========================
    # CREATE EMBEDDINGS
    # =========================

    print("Encoding documents...")

    embeddings = embedding_model.encode(texts)

    # =========================
    # CREATE RETRIEVER
    # =========================

    retriever = Retriever(embeddings)

    # =========================
    # METRIC STORAGE
    # =========================

    precision_scores = []
    recall_scores = []
    mrr_scores = []
    hit_scores = []

    # =========================
    # QUERY LOOP
    # =========================

    for query, relevant_doc in zip(
        queries,
        ground_truth
    ):

        # =========================
        # QUERY EMBEDDING
        # =========================

        query_vec = embedding_model.encode(
            [query]
        )

        # =========================
        # RETRIEVE
        # =========================

        indices = retriever.search(
            query_vec,
            k=3
        )[0]

        # =========================
        # EVALUATION
        # =========================

        precision_scores.append(
            precision_at_k(
                indices,
                relevant_doc,
                3
            )
        )

        recall_scores.append(
            recall_at_k(
                indices,
                relevant_doc,
                3
            )
        )

        mrr_scores.append(
            reciprocal_rank(
                indices,
                relevant_doc
            )
        )

        hit_scores.append(
            hit_rate(
                indices,
                relevant_doc
            )
        )

    # =========================
    # AVERAGE SCORES
    # =========================

    avg_precision = np.mean(
        precision_scores
    )

    avg_recall = np.mean(
        recall_scores
    )

    avg_mrr = np.mean(
        mrr_scores
    )

    avg_hit = np.mean(
        hit_scores
    )

    # =========================
    # PRINT RESULT
    # =========================

    print(f"\nResults for {model_name}:")

    print(
        f"Precision@3: {avg_precision:.4f}"
    )

    print(
        f"Recall@3: {avg_recall:.4f}"
    )

    print(
        f"MRR: {avg_mrr:.4f}"
    )

    print(
        f"Hit Rate: {avg_hit:.4f}"
    )

    # =========================
    # SAVE RESULTS
    # =========================

    all_results.append({

        "Model": model_name,

        "Precision@3": avg_precision,

        "Recall@3": avg_recall,

        "MRR": avg_mrr,

        "Hit Rate": avg_hit
    })

# =========================
# FINAL RESULT TABLE
# =========================

results_df = pd.DataFrame(
    all_results
)

print("\n=========================")
print("FINAL RESULTS")
print("=========================")

print(results_df)

# =========================
# SAVE CSV
# =========================

results_df.to_csv(
    "evaluation/retrieval_results.csv",
    index=False
)

print(
    "\nResults saved to evaluation/retrieval_results.csv"
)