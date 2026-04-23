import numpy as np

def linear_beta_schedule(T, beta_1=0.0001, beta_T=0.02):
    """
    Linear noise schedule from beta_1 to beta_T.
    Returns list of floats rounded to 6 decimals.
    """
    # YOUR CODE HERE
    return np.linspace(beta_1, beta_T, T)

def cosine_alpha_bar_schedule(T, s=0.008):
    """
    Cosine schedule for alpha_bar (cumulative signal retention).
    Returns list of floats rounded to 6 decimals, clipped to [0.0001, 0.9999].
    """
    # YOUR CODE HERE
    steps = np.arange(T + 1)
    f_t = np.cos(((steps / T + s) / (1 + s)) * np.pi / 2) ** 2 
    alpha_bars = f_t / f_t[0]

    alpha_bars = alpha_bars[1:]

    alpha_bars = np.clip(alpha_bars, 0.0001, 0.9999)
    return alpha_bars 

def alpha_bar_to_betas(alpha_bars):
    """
    Convert alpha_bar schedule to beta schedule.
    Returns list of floats rounded to 6 decimals, clipped to [0.0001, 0.9999].
    """

    alpha_bars = np.array(alpha_bars)
    
    # 1. Tạo mảng alpha_bar_{t-1}
    # Thêm 1.0 vào đầu và bỏ phần tử cuối cùng để có mảng dịch chuyển
    # [1.0, alpha_bar[0], alpha_bar[1], ..., alpha_bar[T-2]]
    alpha_bars_prev = np.insert(alpha_bars[:-1], 0, 1.0)
    
    # 2. Tính betas theo công thức: beta_t = 1 - (alpha_bar_t / alpha_bar_{t-1})
    betas = 1 - (alpha_bars / alpha_bars_prev)
    
    # 3. Clip các giá trị vào khoảng an toàn [0.0001, 0.9999]
    betas_clipped = np.clip(betas, 0.0001, 0.9999)
    
    # 4. Trả về list đã làm tròn 6 chữ số
    return [round(float(b), 6) for b in betas_clipped]