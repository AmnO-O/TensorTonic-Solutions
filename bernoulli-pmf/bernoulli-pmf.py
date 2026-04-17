import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    # Write code here

    x = np.array(x)
    x = np.where(x == 0, 1 - p, p)
    
    var = p * (1 - p)

    return x, p, var 