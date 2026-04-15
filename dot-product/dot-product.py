import numpy as np

def dot_product(x, y):
    """
    Compute the dot product of two 1D arrays x and y.
    Must return a float.
    """
    # Write code here

    dot = 0

    if len(x) != len(y):
        raise ValueError

    for i in range(len(x)):
        dot += x[i] * y[i]
    return dot 