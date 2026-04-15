import numpy as np

def ReLU(x):
    x = np.array(x)
    return np.maximum(x, 0)

def identity_block(x, W1, W2):
    """
    Returns: np.ndarray of shape (batch, channels) with identity residual block output
    """
    # YOUR CODE HERE
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)

    fx = ReLU(np.dot(x, W1.T))
    y = ReLU(np.dot(fx, W2.T) + x)

    return y
