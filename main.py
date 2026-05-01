from setting import DATA_PATH
from primary.load import load_data
from models.base import BaseEmbeddingModel
from rag.retriever import Retriever

df, texts = load_data(DATA_PATH)

model = BaseEmbeddingModel("all-MiniLM-L6-v2")
embeddings = model.encode(texts)

retriever = Retriever(embeddings)

while True:
    query = input("You: ")
    if query == "exit":
        break

    query_vec = model.encode([query])
    indices = retriever.search(query_vec, 3)[0]

    print("Bot:")
    for idx in indices:
        print("-", df.iloc[idx]['answer'])