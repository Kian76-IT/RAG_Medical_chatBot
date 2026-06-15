from .base import BaseEmbeddingModel

class DistilBertModel(BaseEmbeddingModel):
    def __init__(self):
        super().__init__("all-distilroberta-v1")