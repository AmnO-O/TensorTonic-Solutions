import numpy as np

def local_response_normalization(x: np.ndarray, k: float = 2, n: int = 5,
                                 alpha: float = 1e-4, beta: float = 0.75) -> np.ndarray:
    """
    Apply Local Response Normalization across channels for input shape (B, H, W, C).
    """
    # Lấy thông tin shape, C nằm ở cuối (axis=3)
    B, H, W, C = x.shape
    half_n = n // 2
    
    # Bước 1: Tính bình phương các phần tử
    square_x = np.square(x)
    
    # Bước 2: Tạo mảng chứa tổng bình phương lân cận
    # Chúng ta sẽ tính tổng trên trục cuối cùng (axis=-1)
    sum_sq = np.zeros_like(square_x)
    
    for i in range(C):
        # Xác định phạm vi cửa sổ trượt n cho channel i
        start = max(0, i - half_n)
        end = min(C, i + half_n + 1)
        
        # Tính tổng các bình phương của các channel lân cận
        # x[:, :, :, start:end] -> (B, H, W, window_size)
        sum_sq[..., i] = np.sum(square_x[..., start:end], axis=-1)
    
    # Bước 3: Áp dụng công thức chuẩn hóa
    # b = a / (k + alpha * sum_sq)^beta
    denom = np.power(k + alpha * sum_sq, beta)
    return x / denom