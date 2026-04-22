import numpy as np


def encoder_block(x: np.ndarray, num_stages : int = 4):
    B, H, W, C = x.shape 

    for i in range(num_stages):
        if C == 1: 
            C = 64 
            H -= 4
            W -= 4
            W = W // 2
            H = H // 2
        else:
            H = H - 4
            W = W - 4
            C = C * 2
            W = W // 2
            H = H // 2 
            
    return np.zeros((B, H, W, C))

def decoder_block(x: np.ndarray, num_stages = 4):
    B, H, W, C = x.shape 

    W = W * 2
    H = H * 2 
    
    for i in range(num_stages - 1):
            # skip connection 
        C = C * 2
            
        C = C // 2
        W = W - 4
        H = H - 4

        C = C // 2 
        W = W * 2 
        H = H * 2 
            
    return np.zeros((B, H, W, C))

def bottleneck(x : np.ndarray):
    B, H, W, C = x.shape 
    return np.zeros((B, H - 4, W - 4, C * 2))



            
def unet(x: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """
    Complete U-Net: trace shape through 4 encoder blocks, bottleneck, 4 decoder blocks, output.
    Each block: two 3x3 unpadded convs (reduce by 4), encoder pools (halve), decoder upsamples (double).
    Returns zero array with correct output shape.
    """
    # Your implementation here

    x = encoder_block(x)
    x = bottleneck(x) 
    x = decoder_block(x)
    B, H, W, C = x.shape 

    return np.zeros((B, H - 4, W - 4, num_classes))
