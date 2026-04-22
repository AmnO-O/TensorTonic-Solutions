import numpy as np

def crop_and_concat(encoder_features: np.ndarray, decoder_features: np.ndarray) -> np.ndarray:
    """
    Crop encoder features to match decoder spatial dims, then concatenate along channels.
    """
    # Your implementation here
    B, H, W, C = encoder_features.shape
    Bd, Hd, Wd, Cd = decoder_features.shape

    diffH = (H - Hd) // 2
    diffW = (W - Wd) // 2

    x_crop = encoder_features[:, diffH : diffH + Hd, diffW : diffW + Wd, :]
    return np.concatenate([x_crop, decoder_features], axis = -1)