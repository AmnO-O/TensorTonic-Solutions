import numpy as np

def vector_norm_3d(v):
    """
    Compute the Euclidean norm of 3D vector(s).
    """
    # Your code here
    v = np.array(v)
    is_single = v.ndim == 1 

    if v.ndim == 1: 
        v = np.expand_dims(v, 0)
  
    norm = np.sqrt(np.sum(v ** 2, axis = 1))
    
    return norm[0] if is_single else norm
