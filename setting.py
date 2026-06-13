import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset",
    "clean_dataset.csv"
)

TOP_K = 3

MODELS = {
    "BERT": "sentence-transformers/all-MiniLM-L6-v2",
    "RoBERTa": "sentence-transformers/all-distilroberta-v1",
    "DistilBERT": "sentence-transformers/msmarco-distilbert-base-v4",
    "ALBERT": "sentence-transformers/paraphrase-albert-small-v2"
}

LLM_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

MAX_NEW_TOKENS = 256
