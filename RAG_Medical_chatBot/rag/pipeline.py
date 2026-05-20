def run_rag(query, model, retriever, df, k=3):
    query_vec = model.encode([query])
    indices = retriever.search(query_vec, k)
    results = df.iloc[indices[0]]
    return results