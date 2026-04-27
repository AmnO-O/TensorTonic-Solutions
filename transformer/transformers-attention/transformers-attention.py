import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
    """
    Compute scaled dot-product attention.
    """
    # Your code here
    batch, seq_len_q, d_k = Q.shape 

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    weight = torch.softmax(scores, dim = -1)
    output = torch.matmul(weight, V) 
    
    return output  