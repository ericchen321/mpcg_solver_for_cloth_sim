# formulate the quadratic function for visualization

import numpy as np

def f_quad(A, b, c, x):
    r"""
    Compute f = 1/2x^TAx - bx + c
    :param A: (n, n) array
    :param b: (n, ) array
    :param c: scalar
    :param x: (n, 1) array

    Return:
    f (scalar)
    """
    b_2d = np.expand_dims(b, axis=1)
    f = 0.5*np.matmul(np.transpose(x), np.matmul(A, x)) - np.matmul(np.transpose(b_2d), x) + c
    return f