import numpy as np
from .error_maps import apply_kraus_map
from .channels import kraus_pauli_channel

def _sample_pauli_errors(pI, px, py, pz, shots, rng):
    probs = np.array([pI, px, py, pz], dtype=float)
    idx = rng.choice(4, size=shots, p=probs)
    return idx  # 0 I, 1 X, 2 Y, 3 Z

def logical_error_repetition_code(kraus_channel, code_length=3, correct_X=True, shots=10000, seed=123):
    """Toy logical error for a classical style repetition code in Z basis if correct_X True,
    or in X basis if False. This is illustrative and aligns with Pauli diagonal approximate behavior."""
    rng = np.random.default_rng(seed)
    # fit a Pauli diagonal channel to extract probs
    from .error_maps import fit_pauli_diagonal_qubit
    pI, px, py, pz = fit_pauli_diagonal_qubit(kraus_channel)

    logical_errors = 0
    for _ in range(shots):
        # draw errors on each of the code qubits
        idxs = _sample_pauli_errors(pI, px, py, pz, code_length, rng)
        # reduce to an effective bit flip for chosen decoding
        if correct_X:
            # majority vote on Z errors interpreted as flips in computational basis are from X and Y
            flips = np.isin(idxs, [1, 2]).sum()
        else:
            # majority vote on X basis flips are from Z and Y
            flips = np.isin(idxs, [2, 3]).sum()
        # majority decoding fails if more than half flips
        if flips > code_length // 2:
            logical_errors += 1
    return logical_errors / shots
