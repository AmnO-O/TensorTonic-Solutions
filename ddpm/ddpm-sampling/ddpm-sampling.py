import numpy as np

def ddpm_sample(x_T, betas, epsilon_preds, z_values):
    """
    Returns: np.ndarray of the final denoised sample
    """
    betas = np.array(betas)
    alphas = 1.0 - betas 
    alpha_bar = np.cumprod(alphas)
    
    epsilon_preds = np.array(epsilon_preds)
    z_values = np.array(z_values) 

    x_t = np.array(x_T)
    T = len(betas)

    for t in reversed(range(T)):
        # Calculate the execution step: 
        # When t=2 -> step 0. When t=1 -> step 1. When t=0 -> step 2.
        step_idx = T - 1 - t
        
        if t > 0:
            # Consume z_values in forward order
            noise = z_values[step_idx] * np.sqrt(betas[t])
        else: 
            noise = 0 
            
        # Consume epsilon_preds in forward order
        eps_theta = epsilon_preds[step_idx]
        
        # DDPM Math
        coeff = (1.0 - alphas[t]) / np.sqrt(1.0 - alpha_bar[t])
        x_t = (1.0 / np.sqrt(alphas[t])) * (x_t - coeff * eps_theta) + noise

    return x_t