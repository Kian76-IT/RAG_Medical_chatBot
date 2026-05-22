DATA_PATH = "dataset/clean_dataset.csv"

MODELS = {

    # MiniLM
    "BERT": "sentence-transformers/all-MiniLM-L6-v2",

    # RoBERTa
    "RoBERTa": "sentence-transformers/all-distilroberta-v1",

    # DistilBERT
    "DistilBERT": "sentence-transformers/msmarco-distilbert-base-v4",

    # ALBERT
    "ALBERT": "sentence-transformers/paraphrase-albert-small-v2"
}

TOP_K = 3