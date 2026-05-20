from .base import BaseEmbeddingModel

class RobertaModel(BaseEmbeddingModel):
    def __init__(self):
        super().__init__("all-roberta-base-v1")