import numpy as np

def alexnet_conv1(image: np.ndarray) -> np.ndarray:
    """
    AlexNet first conv layer: 11x11, stride 4, 96 filters (shape simulation).
    """
    # YOUR CODE HERE
    B, H, _, C = image.shape 
    p = 2 
    s = 4
    k = 11 
    
    H_out = (H + 2 * p - k) // s + 1 
    return np.zeros((B, H_out, H_out, 96))
    