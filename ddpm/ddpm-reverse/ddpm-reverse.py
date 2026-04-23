import numpy as np

def reverse_step(x_t, t, epsilon_pred, betas, z=None):
    """
    Returns: np.ndarray x_{t-1} after one reverse diffusion step
    """
    # YOUR CODE HERE
    alphas = 1.0 - np.array(betas)
    alpha_bar = np.cumprod(alphas)
    x_t = np.array(x_t)

    sigma = np.sqrt(betas[t - 1])
    epsilon_pred = np.array(epsilon_pred)

    if t == 1:
        z_noise = 0 
    elif z == None:
        z_noise = np.random.standard_normal(size=x_t.shape)
    else:
        z_noise = np.array(z)
    
    x = (1.0 / np.sqrt(1 - betas[t - 1])) * (x_t - (betas[t - 1] / np.sqrt(1.0 - alpha_bar[t - 1])) * epsilon_pred) + sigma * z_noise

    return x 