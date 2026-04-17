import numpy as np

def geometric_pmf_mean(k, p):
    """
    Compute Geometric PMF and Mean.
    """
    # Write code here
    
    mean = 1 / p 
    k = [((1 - p) ** (x - 1)) * p for x in k]
    k = np.array(k)
    
    return k, mean 