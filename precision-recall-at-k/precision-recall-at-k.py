def precision_recall_at_k(recommended, relevant, k):
    """
    Compute precision@k and recall@k for a recommendation list.
    """
    # Write code here
    if k <= 0:
        return [0.0, 0.0]

    top_k = recommended[: k]
    relevant_set = set(relevant)

    match = 0

    for item in top_k: 
        if item in relevant_set:
            match += 1

    return [match / k, match / len(relevant)]
    