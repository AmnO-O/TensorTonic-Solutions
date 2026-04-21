import numpy as np

def max_pool2d(x: np.ndarray, kernel_size: int = 3, stride: int = 2) -> np.ndarray:
    """
    Apply 2D max pooling (shape simulation).
    """
    # YOUR CODE HERE
    B, H, W, C = x.shape 
    H_out = (H - kernel_size) // stride + 1

    return np.zeros((B, H_out, H_out, C))