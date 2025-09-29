import numpy as np

def apply_kraus_map(kraus, rho):
    out = np.zeros_like(rho, dtype=complex)
    for K in kraus:
        out += K @ rho @ K.conj().T
    return out

def compose_kraus(kraus_B, kraus_A):
    """Return Kraus set for B∘A."""
    out = []
    for B in kraus_B:
        for A in kraus_A:
            out.append(B @ A)
    return out

def choi_from_kraus(kraus):
    """Choi matrix J = sum_i vec(K_i) vec(K_i)^† with column stacking."""
    d = kraus[0].shape[0]
    J = np.zeros((d * d, d * d), dtype=complex)
    for K in kraus:
        v = K.reshape(d * d, order="F")
        J += np.outer(v, v.conj())
    return J

def pauli_transfer_qubit(kraus):
    """Return 4x4 Pauli transfer matrix R where vec_P(rho) -> R vec_P(rho).
    Basis order [I, X, Y, Z] normalized with Tr(sigma_a sigma_b)=2 delta_ab.
    """
    I = np.array([[1, 0], [0, 1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    P = [I, X, Y, Z]
    R = np.zeros((4, 4), dtype=float)
    for a, A in enumerate(P):
        for b, B in enumerate(P):
            # element R_ab = Tr[ A E(B) ] / 2
            EB = np.zeros((2, 2), dtype=complex)
            for K in kraus:
                EB += K @ B @ K.conj().T
            R[a, b] = (np.trace(A.conj().T @ EB).real) / 2.0
    return R

def fit_pauli_diagonal_qubit(kraus):
    """Return probabilities (pI, pX, pY, pZ) of the closest Pauli channel
    in least squares sense on transfer eigenvalues for X Y Z rows."""
    R = pauli_transfer_qubit(kraus)
    # For a Pauli channel, R is diagonal in Pauli basis with entries [1, rX, rY, rZ]
    rX, rY, rZ = R[1,1], R[2,2], R[3,3]
    # Solve for pX pY pZ given eigenvalues of Pauli channel:
    # For Pauli channel with probs pI pX pY pZ:
    # rX = 1 - 2(pY + pZ), rY = 1 - 2(pX + pZ), rZ = 1 - 2(pX + pY)
    A = np.array([
        [0, -2, -2],
        [-2, 0, -2],
        [-2, -2, 0],
    ], dtype=float)
    b = np.array([rX - 1, rY - 1, rZ - 1], dtype=float)
    try:
        pX, pY, pZ = np.linalg.lstsq(A, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        pX = pY = pZ = 0.0
    pI = max(0.0, 1.0 - (pX + pY + pZ))
    # clip to valid range
    vec = np.clip(np.array([pI, pX, pY, pZ]), 0.0, 1.0)
    vec = vec / max(1e-12, vec.sum())
    return tuple(vec.tolist())

def channel_distance_process_infidelity(kraus_A, kraus_B):
    """One minus process fidelity between channels computed via Choi states."""
    JA = choi_from_kraus(kraus_A)
    JB = choi_from_kraus(kraus_B)
    d = kraus_A[0].shape[0]
    # normalized Choi states
    rhoA = JA / d
    rhoB = JB / d
    # Uhlmann fidelity via eigenvalue method on sqrt
    sA = sqrtm(rhoA)
    prod = sA @ rhoB @ sA
    vals = np.linalg.eigvalsh((prod + prod.conj().T) / 2.0)
    vals = np.clip(vals, 0.0, None)
    F = (np.sqrt(vals).sum()) ** 2
    return 1.0 - F

def sqrtm(A):
    w, V = np.linalg.eigh(A)
    w = np.clip(w, 0.0, None)
    return (V * np.sqrt(w)) @ V.conj().T
