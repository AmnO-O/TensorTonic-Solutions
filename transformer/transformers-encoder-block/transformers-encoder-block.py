import numpy as np

def softmax(x, axis=-1):
    """Provided: Softmax function."""
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def layer_norm(x: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    x_norm = (x - mean) / np.sqrt(var + eps)
    return gamma * x_norm + beta 

def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    batch_size, seq_len, d_model = Q.shape
    d_k = d_model // num_heads
    
    # 1. Chiếu (Project) và chia đầu (Split heads)
    # Kết quả reshape: (B, T, h, d_k) -> transpose sang (B, h, T, d_k)
    def transform(x, W):
        return np.dot(x, W).reshape(batch_size, seq_len, num_heads, d_k).transpose(0, 2, 1, 3)

    q = transform(Q, W_q)
    k = transform(K, W_k)
    v = transform(V, W_v)

    # 2. Scaled Dot-Product Attention
    # Nhân (B, h, T, d_k) với (B, h, d_k, T) -> (B, h, T, T)
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    attn_weights = softmax(scores)
    
    # 3. Nhân với V và gộp đầu (Concat)
    # (B, h, T, T) @ (B, h, T, d_k) -> (B, h, T, d_k)
    context = np.matmul(attn_weights, v)
    context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    
    # 4. Chiếu đầu ra (Output projection)
    return np.dot(context, W_o)

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    # Linear 1 -> ReLU -> Linear 2
    h = np.maximum(0, np.dot(x, W1) + b1)
    return np.dot(h, W2) + b2

def encoder_block(x: np.ndarray, W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                  W_o: np.ndarray, W1: np.ndarray, b1: np.ndarray, W2: np.ndarray,
                  b2: np.ndarray, gamma1: np.ndarray, beta1: np.ndarray,
                  gamma2: np.ndarray, beta2: np.ndarray, num_heads: int) -> np.ndarray:
    
    # Sub-layer 1: Multi-head Attention + Residual + LayerNorm
    attn_out = multi_head_attention(x, x, x, W_q, W_k, W_v, W_o, num_heads)
    x_post_attn = layer_norm(x + attn_out, gamma1, beta1)
    
    # Sub-layer 2: Feed Forward + Residual + LayerNorm
    ffn_out = feed_forward(x_post_attn, W1, b1, W2, b2)
    output = layer_norm(x_post_attn + ffn_out, gamma2, beta2)
    
    return output