import numpy as np
from typing import Literal

StarCouplerType = Literal["phase", "gamma", "hadamard"]


def phase(M: int):
    B = np.ones((M, M)) / np.sqrt(M)
    phi_matrix = np.outer(np.arange(M), np.arange(M)) * (2 * np.pi / M)
    phase_factors = np.exp(1j * phi_matrix)
    return B * phase_factors


def gamma(M: int):
    idx = np.arange(M)
    gamma = np.exp(-2.0j * np.pi / M)
    return np.pow(gamma, np.outer(idx, idx)) / np.sqrt(M)


def H(n: int) -> np.ndarray:
    if n == 1:
        return np.array([[1]])
    h = H(n // 2)
    return np.block([[h, h], [h, -h]])


def hadamard(M: int) -> np.ndarray:
    return H(M) / np.sqrt(M)


def get_star_coupler(method: StarCouplerType, M: int) -> np.ndarray:
    methods = {
        "phase": phase,
        "gamma": gamma,
        "hadamard": hadamard,
    }
    assert method in methods, f"{method} is not one of {list(methods.keys())}"
    return np.real(methods[method](M))
