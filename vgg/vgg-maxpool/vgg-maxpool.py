import numpy as np

def vgg_maxpool(x: np.ndarray) -> np.ndarray:
    """
    Implement VGG-style max pooling (2x2, stride 2).
    """
    # Your implementation here
    N, H, W, C = x.shape 
    out = np.zeros((N, H // 2, W // 2, C))
    
    for i in range(0, H, 2): 
        for j in range(0, W, 2): 
            window = x[: , i : i + 2, j : j + 2, :]
            out[: , i // 2, j // 2, :] = np.max(window, axis = (1, 2)) 

    return out 
    