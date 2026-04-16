import numpy as np

def compute_gradient_with_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITH skip connections.
    Gradient at layer l = sum of paths through network
    """
    # YOUR CODE HERE
        
    gradients_F = np.array(gradients_F)
    x = np.expand_dims(x, 0)

    for L in range(len(gradients_F)):
        x = x @ (gradients_F[L] + np.eye(gradients_F[L].shape[0]))    
    return x.flatten()
    

def compute_gradient_without_skip(gradients_F: list, x: np.ndarray) -> np.ndarray:
    """
    Compute gradient flow through L layers WITHOUT skip connections.
    """
    # YOUR CODE HERE
    
    gradients_F = np.array(gradients_F)
    x = np.expand_dims(x, 0)

    for L in range(len(gradients_F)):
        x = np.dot(x, gradients_F[L]) 
    
    return x.flatten()
    

    
