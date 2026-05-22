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

import pandas as pd
import numpy as np

from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from bert_score import score as bertscore

from setting import DATA_PATH

from primary.load import load_data
from models.base import BaseEmbeddingModel
from rag.retriever import Retriever
from rag.pipeline import run_rag
from llm.generator import LLMGenerator

# =========================
# LOAD DATA
# =========================

print("Loading dataset...")

df, texts = load_data(DATA_PATH)

print("Dataset loaded!")

# =========================
# LOAD EMBEDDING MODEL
# =========================

print("Loading embedding model...")

embedding_model = BaseEmbeddingModel(
    "sentence-transformers/paraphrase-albert-small-v2"
)

# =========================
# CREATE EMBEDDINGS
# =========================

print("Encoding documents...")

embeddings = embedding_model.encode(texts)

# =========================
# RETRIEVER
# =========================

retriever = Retriever(embeddings)

# =========================
# LOAD LLM
# =========================

print("Loading LLM...")

llm = LLMGenerator(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)

# =========================
# METRIC STORAGE
# =========================

bleu_scores = []

rouge1_scores = []
rougeL_scores = []

predictions = []
references = []

# =========================
# ROUGE SCORER
# =========================

scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rougeL'],
    use_stemmer=True
)

# =========================
# LIMIT EVALUATION
# =========================

# jangan semua 32k dulu
# nanti lama sekali

EVAL_SIZE = 100

eval_df = df.sample(
    EVAL_SIZE,
    random_state=42
)

# =========================
# EVALUATION LOOP
# =========================

for idx, row in eval_df.iterrows():

    query = row["question"]

    reference = row["answer"]

    # =========================
    # RAG RETRIEVAL
    # =========================

    results = run_rag(
        query,
        embedding_model,
        retriever,
        df,
        k=3
    )

    context = "\n".join(
        results["context"].tolist()
    )

    # =========================
    # GENERATE RESPONSE
    # =========================

    prediction = llm.generate(
        query,
        context
    )

    # =========================
    # STORE
    # =========================

    predictions.append(prediction)
    references.append(reference)

    # =========================
    # BLEU
    # =========================

    bleu = sentence_bleu(
        [reference.split()],
        prediction.split()
    )

    bleu_scores.append(bleu)

    # =========================
    # ROUGE
    # =========================

    rouge_scores = scorer.score(
        reference,
        prediction
    )

    rouge1_scores.append(
        rouge_scores['rouge1'].fmeasure
    )

    rougeL_scores.append(
        rouge_scores['rougeL'].fmeasure
    )

    print(f"Done: {len(predictions)}/{EVAL_SIZE}")

# =========================
# BERTSCORE
# =========================

print("\nCalculating BERTScore...")

P, R, F1 = bertscore(
    predictions,
    references,
    lang="en",
    verbose=True
)

# =========================
# FINAL RESULTS
# =========================

results = {

    "BLEU": np.mean(bleu_scores),

    "ROUGE-1": np.mean(rouge1_scores),

    "ROUGE-L": np.mean(rougeL_scores),

    "BERTScore-F1": F1.mean().item()
}

# =========================
# PRINT RESULTS
# =========================

print("\n=========================")
print("GENERATOR EVALUATION")
print("=========================")

for metric, value in results.items():

    print(f"{metric}: {value:.4f}")

# =========================
# SAVE RESULTS
# =========================

results_df = pd.DataFrame([results])

results_df.to_csv(
    "evaluation/generator_results.csv",
    index=False
)

print(
    "\nResults saved to evaluation/generator_results.csv"
)