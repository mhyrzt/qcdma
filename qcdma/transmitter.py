import numpy as np
from qcdma.state import coherent


class QuantumTransmitter:
    def __init__(
        self,
        phase: float,
        N: int = 10,
        alpha: float = 1.0,
    ) -> None:
        self.N = N
        self.alpha = alpha
        self.phase = phase
        self.one_state = coherent(N, alpha)
        self.zero_state = coherent(N, 0)

    def send(self, data: int):
        phase = self.phase if data == 1 else 0.0
        state = self.one_state if data == 1 else self.zero_state
        return state * np.exp(1j * phase)
