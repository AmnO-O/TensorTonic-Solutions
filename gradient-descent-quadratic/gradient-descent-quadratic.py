def derivative(a, b, x):
    return 2 * a * x + b 

def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here

    for step in range(steps):
        x0 = x0 - lr * derivative(a, b, x0)

    return x0 