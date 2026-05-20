from sentence_transformers import SentenceTransformer

class BaseEmbeddingModel:
    def __init__(self, model_name):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(texts, convert_to_numpy=True)