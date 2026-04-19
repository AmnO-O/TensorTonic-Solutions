import numpy as np

def maxpool_2x2(x):
    B, H, W, C = x.shape
    return x.reshape(B, H//2, 2, W//2, 2, C).max(axis=(2, 4))

def vgg_features(x: np.ndarray, config: list, conv_weights: list, conv_biases: list) -> np.ndarray:
    """
    Returns: np.ndarray feature tensor after applying conv layers and max pooling
    """
    # Your implementation here
    conv_idx = 0 
    
    for layer in config: 
        if layer == 'M':
            x = maxpool_2x2(x)
        else:
            W = conv_weights[conv_idx]
            b = conv_biases[conv_idx] 

            x = x @ W + b
            x = np.maximum(x, 0) 
            conv_idx += 1 

    return x 