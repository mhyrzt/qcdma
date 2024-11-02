import numpy as np
from scipy.special import factorial


def coherent(N: int, alpha: complex) -> np.ndarray:
    """_summary_

    Args:
        N (int): Dimension of the Hilbert space
        alpha (complex): Coherent state Parameter

    Returns:
        np.ndarray: Coherent State Vector
    """
    n = np.arange(N)
    exp = np.exp(-np.abs(alpha) ** 2 / 2)
    state = np.pow(alpha, n) / np.sqrt(factorial(n))
    return exp * state

def fock(N: int, idx: int) -> np.ndarray:
    arr = np.zeros(N)
    arr[idx] = 1.0
    return arr