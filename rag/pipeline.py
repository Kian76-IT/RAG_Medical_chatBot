def run_rag(query,embedding_model,retriever,df,k=3):
    # QUERY EMBEDDING
    query_embedding = embedding_model.encode(
        [query]
    )

    # FAISS SEARCH
    indices = retriever.search(
        query_embedding,
        k
    )[0]

    # RETRIEVED DOCUMENTS
    retrieved_docs = df.iloc[
        indices
    ].copy()

    print("\n========== RETRIEVED CONTEXT ==========")
    for i, text in enumerate(
        retrieved_docs["text"].tolist(),
        start=1
    ):
        print(f"\n[{i}]")
        print(text[:300])
    print("\n=======================================")
    return {
        "indices": indices,
        "contexts": retrieved_docs["text"].tolist()
    }