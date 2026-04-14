def precision_recall(recommended, relevant):
    recommended = set(recommended)
    relevant = set(relevant)

    true_positive = len(recommended & relevant)

    precision = true_positive / len(recommended) if recommended else 0
    recall = true_positive / len(relevant) if relevant else 0

    return precision, recall


# Example usage
if __name__ == "__main__":
    recommended = ["Interstellar", "Matrix", "Avengers"]
    relevant = ["Interstellar", "Matrix"]

    p, r = precision_recall(recommended, relevant)
    print("Precision:", p)
    print("Recall:", r)
