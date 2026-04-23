import numpy as np

def get_alpha_bar(betas):
    """
    Compute cumulative product of (1 - beta).
    Returns list of floats rounded to 6 decimals.
    """
    # YOUR CODE HERE

    alphas = 1 - np.array(betas)
    alpha_bar = np.cumprod(alphas)
    
    return [round(float(a), 6) for a in alpha_bar]


def forward_diffusion(x_0, t, betas, epsilon):
    """
    Returns: tuple of (np.ndarray x_t, np.ndarray epsilon) with same shape as x_0
    """
    # YOUR CODE HERE

    alpha_bar = get_alpha_bar(betas)  

    epsilon = np.array(epsilon) 
    x_0 = np.array(x_0)
    x_t = np.sqrt(alpha_bar[t-1]) * x_0 + np.sqrt(1 - alpha_bar[t-1]) * epsilon
    return x_t.tolist()