import numpy as np

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Write code here
    total_tp = np.sum(y_true == y_pred)
    total_f = len(y_true) - total_tp

    return total_tp * 2 / (2 * total_tp + 2 * total_f)