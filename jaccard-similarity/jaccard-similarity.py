def jaccard_similarity(set_a, set_b):
    """
    Compute the Jaccard similarity between two item sets.
    """
    # Write code here
    set_a = set(set_a)
    set_b = set(set_b)
    
    set_ab = set_a & set_b

    denom = len(set_a) + len(set_b) - len(set_ab)

    if denom == 0:
        return 0.0

    
    return len(set_ab) / denom