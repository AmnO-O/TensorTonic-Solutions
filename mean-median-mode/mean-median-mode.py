import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    # Write code here
    x = sorted(x)
    n = len(x)
    
    if n == 0:
        return None 
        
    mean = 0 
    mode = 0 
    median = 0 
    counts = {}

    for val in x: 
        mean = mean + val
        counts[val] = counts.get(val, 0) + 1

    if n % 2 == 1:
        median = x[n // 2]
    else:
        median = (x[n // 2 - 1] + x[n // 2]) / 2
        
    mode = max(counts, key = counts.get)    
    mean = mean / n 

    
    return mean, median, mode 