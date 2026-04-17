import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    if not x:
        return None
        
    x = sorted(x)
    n = len(x)
    
    # 1. Tính Mean
    mean = sum(x) / n
    
    # 2. Tính Median
    if n % 2 == 1:
        median = x[n // 2]
    else:
        median = (x[n // 2 - 1] + x[n // 2]) / 2.0
        
    # 3. Tính Mode
    counts = Counter(x)
    mode = max(counts, key=counts.get)
    

    return mean, median, mode