import numpy as np

def ReLU(x):
    return np.maximum(x, 0)

def conv_block(x, W1, W2, Ws):
    x = np.asarray(x)
    
    # 1. Nhánh chính (Main path)
    # Bước 1: ReLU sau lớp W1
    h = ReLU(np.dot(x, W1))
    
    # Bước 2: Nhân với W2 (KHÔNG dùng ReLU ở đây)
    z = np.dot(h, W2)

    # 2. Nhánh tắt (Shortcut path)
    s = np.dot(x, Ws)

    # 3. Cộng lại và dùng ReLU cuối cùng
    # Đây là điểm mấu chốt để ra 0.0 thay vì số âm
    y = ReLU(z + s)

    return y