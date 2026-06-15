import faiss
import numpy as np

class Retriever:
    def __init__(self, embeddings):
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)

    def search(self, query_embedding, k=3):
        distances, indices = self.index.search(query_embedding, k)
        return indices