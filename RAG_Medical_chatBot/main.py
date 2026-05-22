from setting import DATA_PATH

from primary.load import load_data
from models.base import BaseEmbeddingModel

from rag.retriever import Retriever
from rag.pipeline import run_rag

from llm.generator import LLMGenerator


# LOAD DATA
print("Loading dataset...")

df, texts = load_data(DATA_PATH)

print("Dataset loaded!")

# EMBEDDING MODEL
print("Loading embedding model...")

embedding_model = BaseEmbeddingModel(
    "all-MiniLM-L6-v2"
)

print("Creating embeddings...")
embeddings = embedding_model.encode(texts)
print("Embeddings created!")


# RETRIEVER
retriever = Retriever(embeddings)
print("Retriever ready!")


# LLM
print("Loading LLM...")
llm = LLMGenerator(
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
)
print("LLM loaded!")


# CHAT LOOP
while True:

    query = input("\nYou: ")

    if query.lower() == "exit":
        print("Chat ended.")
        break

    # RAG RETRIEVAL
    results = run_rag(
        query,
        embedding_model,
        retriever,
        df,
        k=3
    )

    # combine retrieved context
    context = "\n".join(
        results["context"].tolist()
    )

    # GENERATE RESPONSE
    response = llm.generate(
        query,
        context
    )

    # OUTPUT
    print("\nBot:")
    print(response)