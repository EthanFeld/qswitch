import numpy as np

def kraus_identity(d: int):
    K = [np.eye(d, dtype=complex)]
    return K

def kraus_pauli_channel(px: float, py: float, pz: float):
    """Qubit Pauli channel with probabilities on X Y Z.
    Total non identity weight p = px+py+pz ; identity weight 1-p."""
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    pI = max(0.0, 1.0 - (px + py + pz))
    K = [
        np.sqrt(pI) * I,
        np.sqrt(px) * X,
        np.sqrt(py) * Y,
        np.sqrt(pz) * Z,
    ]
    return K

def kraus_depolarizing_qubit(p: float):
    """Qubit depolarizing with parameter p: with prob p apply random Pauli."""
    px = py = pz = p / 3.0
    return kraus_pauli_channel(px, py, pz)

def kraus_phase_damping_qubit(lmbda: float):
    """Phase damping with parameter lambda in [0,1]."""
    K0 = np.array([[1, 0], [0, np.sqrt(1 - lmbda)]], dtype=complex)
    K1 = np.array([[0, 0], [0, np.sqrt(lmbda)]], dtype=complex)
    return [K0, K1]

def kraus_amplitude_damping_qubit(gamma: float):
    """Amplitude damping with gamma in [0,1]."""
    K0 = np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex)
    K1 = np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex)
    return [K0, K1]

def kraus_bb84_like_qubit(q: float):
    """BB84 style effective channel used in many analyses.
    N_q(rho) = (1-q)^2 rho + q(1-q) X rho X + q^2 Y rho Y + q(1-q) Z rho Z
    """
    pI = (1 - q) ** 2
    px = q * (1 - q)
    py = q ** 2
    pz = q * (1 - q)
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    return [np.sqrt(pI) * I, np.sqrt(px) * X, np.sqrt(py) * Y, np.sqrt(pz) * Z]

def validate_kraus(kraus):
    """Check CPTP: sum_i K_i^† K_i = I within tolerance."""
    d = kraus[0].shape[0]
    acc = np.zeros((d, d), dtype=complex)
    for K in kraus:
        acc += K.conj().T @ K
    return np.allclose(acc, np.eye(d), atol=1e-8)
