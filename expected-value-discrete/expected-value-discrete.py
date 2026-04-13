import numpy as np

def expected_value_discrete(x, p):
    x = np.array(x, dtype=np.float64)
    p = np.array(p, dtype=np.float64)

    # Kiểm tra tổng xác suất
    if not np.isclose(np.sum(p), 1.0):
        raise ValueError("Total probability must be equal to 1.")
    
    return x.dot(p)