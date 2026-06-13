from .base import BaseEmbeddingModel

class BertModel(BaseEmbeddingModel):
    def __init__(self):
        super().__init__("all-MiniLM-L6-v2")  # representasi BERT-family