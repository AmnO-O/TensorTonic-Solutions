import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Generate sinusoidal positional encodings.
    """
    # Your code here

    PE = np.zeros((seq_length, d_model))
    position = np.arange(0, seq_length)[:, np.newaxis]

    i = np.arange(0, d_model, 2)
    div_term = np.exp(- np.log(10000) * i / d_model)
    
    PE[: , 0 : : 2] = np.sin(position * div_term)
    PE[: , 1 : : 2] = np.cos(position * div_term)

    return PE 
    
    