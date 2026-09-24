import faiss
import numpy as np

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field


# ============================================================
# ORIGINAL FAISS RETRIEVER
# ============================================================

class Retriever:

    def __init__(self, embeddings):

        self.dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(
            self.dimension
        )

        self.index.add(
            embeddings
        )

    def search(self, query_embedding, k=3):

        distances, indices = self.index.search(
            query_embedding,
            k
        )

        return indices


# ============================================================
# LANGCHAIN RETRIEVER
# ============================================================

class AlbertFaissRetriever(BaseRetriever):

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    embedding_model: object = Field(
        exclude=True
    )

    faiss_retriever: Retriever = Field(
        exclude=True
    )

    dataframe: object = Field(
        exclude=True
    )

    k: int = 5

    def _get_relevant_documents(
        self,
        query,
        *,
        run_manager=None
    ):

        # ----------------------------------------------------
        # 1. Encode user query with ALBERT
        # ----------------------------------------------------

        query_embedding = self.embedding_model.encode(
            [query]
        )

        # ----------------------------------------------------
        # 2. FAISS search
        # ----------------------------------------------------

        indices = self.faiss_retriever.search(
            query_embedding,
            k=self.k
        )[0]

        # ----------------------------------------------------
        # 3. Convert dataframe rows -> LangChain Documents
        # ----------------------------------------------------

        documents = []

        for idx in indices:

            row = self.dataframe.iloc[int(idx)]

            document = Document(
                page_content=str(row["text"]),
                metadata={
                    "index": int(idx)
                }
            )

            documents.append(
                document
            )

        return documents