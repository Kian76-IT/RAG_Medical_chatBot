DATA_PATH = "RAG_Medical_chatBot/dataset/clean_dataset.csv"

MODELS = {
    "BERT": "all-MiniLM-L6-v2",
    "RoBERTa": "all-roberta-base-v1",
    "DistilBERT": "all-distilroberta-v1",
    "ALBERT": "sentence-transformers/paraphrase-albert-small-v2"
}

TOP_K = 3