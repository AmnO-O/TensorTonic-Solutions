import numpy as np

def sample_var_std(x):
    """
    Compute sample variance and standard deviation.
    """
    # Write code here
    n = len(x)
    x = np.array(x) 
    mean = np.mean(x)
    var = (1 / (n - 1)) * np.sum((x - mean) ** 2)
    std = np.sqrt(var)

    return var, std  