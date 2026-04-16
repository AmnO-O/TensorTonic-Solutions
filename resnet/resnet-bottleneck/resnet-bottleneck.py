import numpy as np

def ReLU(x):
    return np.maximum(x, 0)

def bottleneck_block(x, W1, W2, W3, Ws):
    """
    Returns: np.ndarray with bottleneck residual block output (compress, process, expand + skip)
    """
    # YOUR CODE HER

    x = np.asarray(x)
    W1 = np.asarray(W1)
    W2 = np.asarray(W2)
    W3 = np.asarray(W3)

    shortcut = x

    if Ws is not None:
        shortcut = np.dot(x, Ws)

    out = ReLU(np.dot(x, W1))
    out = ReLU(np.dot(out, W2))
    out = np.dot(out, W3)


    return ReLU(out + shortcut)