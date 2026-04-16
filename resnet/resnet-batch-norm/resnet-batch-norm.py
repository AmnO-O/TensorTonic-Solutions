import numpy as np

def ReLU(x):
    return np.maximum(x, 0)

def batch_norm_block(x, W1, W2, gamma1, beta1, gamma2, beta2, mode):
    """
    Returns: np.ndarray of same shape as input with batch-normalized and skip-connected output
    """
    x = np.array(x)
    W1 = np.array(W1)
    W2 = np.array(W2)
    
    # YOUR CODE HERE
    if mode == "post":
        shortcut = x
        x = x @ W1
        
        mean = np.mean(x, axis = 0)
        var = np.var(x, axis = 0)
        x_hat = (x - mean) / np.sqrt(var + 1e-5)
        x = x_hat * gamma1 + beta1

        x = ReLU(x)
        x = x @ W2

        mean = np.mean(x, axis = 0)
        var = np.var(x, axis = 0)
        x_hat = (x - mean) / np.sqrt(var + 1e-5)
        x = x_hat * gamma2 + beta2 

        x = x + shortcut 
        x = ReLU(x)

    elif mode == 'pre':
        shortcut = x
        
        mean = np.mean(x, axis = 0)
        var = np.var(x, axis = 0)
        x_hat = (x - mean) / np.sqrt(var + 1e-5)
        x = x_hat * gamma1 + beta1

        x = ReLU(x) 

        x = x @ W1
        
        mean = np.mean(x, axis = 0)
        var = np.var(x, axis = 0)
        x_hat = (x - mean) / np.sqrt(var + 1e-5)
        x = x_hat * gamma2 + beta2 

        x = ReLU(x)

        x = x @ W2
        
        x = x + shortcut

    return {'output' : x, 'mode': mode}