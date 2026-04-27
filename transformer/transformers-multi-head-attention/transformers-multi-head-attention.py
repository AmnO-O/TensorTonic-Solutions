import numpy as np

def softmax(x, axis=-1):
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e_x / np.sum(e_x, axis=axis, keepdims=True)

def split_into_heads(x, num_heads):
    batch_size, seq_len, d_model = x.shape 
    x = x.reshape(batch_size, seq_len, num_heads, d_model // num_heads)
    x = x.transpose(0, 2, 1, 3)
    return x 
    
def multi_head_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                         W_q: np.ndarray, W_k: np.ndarray, W_v: np.ndarray,
                         W_o: np.ndarray, num_heads: int) -> np.ndarray:
    """
    Compute multi-head attention.
    """
    # Your code here
    Q = np.matmul(Q, W_q)
    K = np.matmul(K, W_k)
    V = np.matmul(V, W_v)

    
    Q = split_into_heads(Q, num_heads)
    K = split_into_heads(K, num_heads)
    V = split_into_heads(V, num_heads)

    batch_size, num_heads, seq_len, d_k = Q.shape
    
    scores = np.matmul(Q, K.transpose(0, 1, 3, 2)) / np.sqrt(d_k)
    attention = softmax(scores)
    output = np.matmul(attention, V)


    output = output.transpose(0, 2, 1, 3)
    output = output.reshape(batch_size, seq_len, d_k * num_heads)

    return np.matmul(output, W_o)