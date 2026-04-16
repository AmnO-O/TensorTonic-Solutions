import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    
    x = np.array(x) 
    
    # 1. Tạo bộ sinh số ngẫu nhiên (nếu có rng thì dùng, không thì dùng mặc định)
    if rng is None:
        random_values = np.random.rand(*x.shape)
    else:
        random_values = rng.random(x.shape)
        
    scale = 1 / (1 - p)
    
    dropout_mask = (random_values > p).astype(np.float32) * scale

    x = x * dropout_mask 

    return (x, dropout_mask) 