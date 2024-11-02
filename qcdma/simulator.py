import numpy as np
from qcdma.star_coupler import get_star_coupler, StarCouplerType
from qcdma.transmitter import QuantumTransmitter
from typing import Iterable


class SimulatorQCDMA:
    def __init__(
        self,
        n_users: int,
        star_coupler_method: StarCouplerType,
        threshold: float = 0.2,
        N_transmitter: int = 10,
        alpha_transmitter: float = 1,
    ):
        assert n_users > 0
        self.B = get_star_coupler(star_coupler_method, M=n_users)
        self.n_users = n_users
        self.phases = np.random.uniform(0, 2 * np.pi, n_users)
        self.transmitter = [
            QuantumTransmitter(phase=phase, N=N_transmitter, alpha=alpha_transmitter)
            for phase in self.phases
        ]
        self.threshold = threshold

        self.expected_states_0 = np.array(
            [trans.zero_state for trans in self.transmitter]
        )
        self.expected_states_1 = np.array(
            [trans.one_state for trans in self.transmitter]
        )

    def psi_e(self, data: np.ndarray) -> np.ndarray:
        return np.vstack([self.transmitter[i].send(x) for i, x in enumerate(data)])

    def phi_d(self, phi_e: np.ndarray) -> np.ndarray:
        phases = self.phases[:, np.newaxis]
        return phi_e * np.exp(-1j * phases)

    def intensity(self, phi_d: np.ndarray) -> float:
        return np.abs(phi_d) ** 2

    def decode(self, phi_d: np.ndarray) -> np.ndarray:
        """ 
        Decodes by maximizing correlation.
        """
        correlations = np.zeros(self.n_users)
        
        # Compare received signals against expected 0 and 1 states for each user
        for i in range(self.n_users):
            # We can use inner product as a measure of correlation
            corr_1 = np.vdot(phi_d[i], self.expected_states_1[i])
            corr_0 = np.vdot(phi_d[i], self.expected_states_0[i])
            
            # Decision based on which correlation is larger
            correlations[i] = 1 if abs(corr_1) > abs(corr_0) else 0
            
        return correlations.astype(int)


    def simulate(self, data: Iterable):
        assert len(data) == self.n_users
        psi_e = self.psi_e(data)
        phi_e = self.B.T @ psi_e
        phi_d = self.phi_d(phi_e)
        intensity = self.intensity(phi_d)
        decoded_bits = self.decode(intensity)

        return {
            "data": np.array(data),
            "psi_e": psi_e,
            "phi_e": phi_e,
            "phi_d": phi_d,
            "intensity": intensity,
            "decoded_bits": decoded_bits,
        }
