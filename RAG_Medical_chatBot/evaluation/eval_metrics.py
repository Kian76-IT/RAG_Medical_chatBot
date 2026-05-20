def precision_at_k(relevant, retrieved):
    return len(set(relevant) & set(retrieved)) / len(retrieved)

def recall_at_k(relevant, retrieved):
    return len(set(relevant) & set(retrieved)) / len(relevant)

def f1_score(p, r):
    return 2 * (p * r) / (p + r + 1e-9)

def accuracy_at_k(relevant, retrieved):
    return 1 if len(set(relevant) & set(retrieved)) > 0 else 0