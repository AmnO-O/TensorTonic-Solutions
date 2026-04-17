import numpy as np
from scipy.special import comb

def binomial_pmf_cdf(n, p, k):
    """
    Compute Binomial PMF and CDF.
    """
    # Write code here
    k = np.array(k)
    pmf = comb(n, k) * (p ** k) * ((1 - p) ** (n - k))

    
    i = np.arange(0, k + 1)
    cdf = np.sum(comb(n, i) * (p ** i) * ((1 - p) ** (n - i))) 

    return pmf, cdf 