import numpy as np

def f1_micro(y_true, y_pred) -> float:
    """
    Compute micro-averaged F1 for multi-class integer labels.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Write code here
    total_tp = np.sum(y_true == y_pred)
    total_fp = len(y_true) - total_tp

    ## for this problem total_fp = total_fn because this is multi-class 

    return total_tp * 2 / (2 * total_tp + 2 * total_fp)