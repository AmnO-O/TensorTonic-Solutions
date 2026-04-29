import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    x = np.array(x) 
    gamma = np.array(gamma)
    beta = np.array(beta) 
    ndim = x.ndim
    
    if ndim == 4:
        # Lấy số lượng channels (C) từ shape (N, C, H, W)
        C = x.shape[1]
        # Reshape gamma và beta về (1, C, 1, 1) để broadcast
        gamma = gamma.reshape(1, C, 1, 1)
        beta = beta.reshape(1, C, 1, 1)
        axes = (0, 2, 3)
    else:
        # Mặc định cho (N, D)
        axes = (0,)

    mean = np.mean(x, axis=axes, keepdims=True)
    var = np.var(x, axis=axes, keepdims=True)
    
    x_hat = (x - mean) / np.sqrt(var + eps)
    
    return gamma * x_hat + beta
