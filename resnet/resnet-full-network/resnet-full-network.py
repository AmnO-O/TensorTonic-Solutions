import numpy as np

def ReLU(x):
    return np.maximum(x, 0)

def resnet_forward(x, conv1, W1_b1, W2_b1, W1_b2, W2_b2, Ws_b2, fc):
    """
    Returns: np.ndarray of shape (batch, num_classes) with classification logits
    """
    x = np.array(x)
    conv1 = np.array(conv1)
    W1_b1 = np.array(W1_b1)
    W2_b1 = np.array(W2_b1)
    W1_b2 = np.array(W1_b2)
    W2_b2 = np.array(W2_b2)
    Ws_b2 = np.array(Ws_b2)
    fc = np.array(fc)
        

    x = x @ conv1 
    x = ReLU(x) 
    shortcut = x 
    x = x @ W1_b1
    x = ReLU(x)
    fx = x @ W2_b1
    x = fx + shortcut 
    x = ReLU(x)

    shortcut = x 
    x = x @ W1_b2 
    x = ReLU(x)
    fx = x @ W2_b2 
    x = fx + shortcut @ Ws_b2
    x = ReLU(x) 

    x = x @ fc
    return x 