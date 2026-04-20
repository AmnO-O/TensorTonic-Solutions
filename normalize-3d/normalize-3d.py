import numpy as np


def normalize_3d(v):
    v = np.array(v, dtype=float)
    # Tính norm cho từng vectơ (theo hàng)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    
    # Tránh chia cho 0: nếu norm bằng 0 thì giữ nguyên, ngược lại thì chia
    return np.divide(v, norm, out=np.zeros_like(v), where=norm!=0)
