from .base import BaseEmbeddingModel

class AlbertModel(BaseEmbeddingModel):
    def __init__(self):
        super().__init__("sentence-transformers/paraphrase-albert-small-v2")