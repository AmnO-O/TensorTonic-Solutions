import numpy as np

def max_pooling_2d(X, pool_size):
    """
    Apply 2D max pooling with non-overlapping windows.
    """
    x = np.array(X)
    H, W = x.shape 

    H_out = H // pool_size 
    W_out = W // pool_size 

    x = x.reshape(H_out, pool_size, W_out, pool_size)

    return np.max(x, axis = (1, 3)).tolist() 