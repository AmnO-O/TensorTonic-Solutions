import numpy as np

def conv2d(x, W, b):
    """
    Simple 2D convolution layer forward pass.
    Valid padding, stride=1.
    """
    # Write code here

    x = np.array(x)
    W = np.array(W)
    b = np.array(b)

    nbatch, C_in, H, W_in = x.shape
    C_out, _ , KH, KW = W.shape

    H_out = H - KH + 1
    W_out = W_in - KW + 1 

    output = np.zeros((nbatch, C_out, H_out, W_out))
    
    for bat in range(nbatch):
        for c_out in range(C_out):
            output[bat, c_out] = b[c_out]

            for c_in in range(C_in):
                for i in range(KH):
                    for j in range(KW):
                        window = x[bat, c_in, i : i + H_out, j : j + W_out]
                        output[bat, c_out] += W[c_out, c_in, i, j] * window

    return output
            