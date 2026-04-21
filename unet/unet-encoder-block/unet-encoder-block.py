import numpy as np

def unet_encoder_block(x: np.ndarray, out_channels: int) -> tuple:
    """
    Returns (pool_out, skip_out) as zero arrays with correct shapes.
    """
    # Your implementation here

    B, H, W, C = x.shape 
 
    pool_out = np.zeros((B, (H - 4) // 2, (W - 4) // 2, out_channels))
    skip_out = np.zeros((B, H - 4, W - 4, out_channels))

    return (pool_out, skip_out)