def precision_at_k(recommended, relevant, k=5):
    return len(set(recommended[:k]) & set(relevant)) / k
