import numpy as np
import math 

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """

    if len(y) == 0:
        return 0.0
        
    # Write code here
    counts = {}
    
    for label in y: 
        counts[label] = counts.get(label, 0) + 1 
        
    entropy = 0 
    
    for label in counts:
        p = counts[label] / len(y)
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy 