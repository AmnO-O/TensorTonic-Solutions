import numpy as np

def make_diagonal(v):
    """
    Returns: (n, n) NumPy array with v on the main diagonal
    """
    # Write code here
    n = len(v)
    D = np.zeros((n, n))

    index = np.arange(n)

    D[index, index] = v

    return D 
