import numpy as np

def sigmoid(x):
    """
    Vectorized sigmoid function.
    """
    # Write code here
    x = np.array(x)

    return np.power(np.exp(-x) + 1, -1).tolist()