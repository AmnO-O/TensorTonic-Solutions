import numpy as np

def compute_ddpm_loss(x_0, betas, t_values, epsilon, epsilon_pred):
    """
    Returns: float scalar MSE loss between true noise and predicted noise
    """
    # YOUR CODE HERE

    epsilon = np.array(epsilon)
    epsilon_pred = np.array(epsilon_pred) 

    return np.mean(np.square(epsilon - epsilon_pred))