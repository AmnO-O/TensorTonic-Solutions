import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    # Write code here

    x = np.array(x)

    x = np.maximum(x, 0)
    
    return np.array(x)