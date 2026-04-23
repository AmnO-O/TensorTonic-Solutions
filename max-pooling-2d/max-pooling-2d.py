import numpy as np

def max_pooling_2d(X, pool_size):
    """
    Apply 2D max pooling with non-overlapping windows.
    """
    x = np.array(X)
    H, W = x.shape 

    H_out = H // pool_size 
    W_out = W // pool_size 

    out = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            # Cắt vùng cửa sổ từ (i*s) đến ((i+1)*s)
            window = x[i*pool_size : (i+1)*pool_size, 
                       j*pool_size : (j+1)*pool_size]
            out[i, j] = np.max(window) 

    return out.tolist()