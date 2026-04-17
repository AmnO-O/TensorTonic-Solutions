import numpy as np

def geometric_pmf_mean(k, p):
    """
    Compute Geometric PMF and Mean.
    """
    # Write code here
    
    mean = 1 / p 
    k_array = np.array(k)
    
    pmf = ((1 - p) ** (k_array - 1)) * p
    
    return pmf, mean 