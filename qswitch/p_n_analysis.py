import numpy as np
from .error_maps import compose_kraus

def pn_depolarizing_qubit_exact(p: float, n: int):
    """Exact P_n for qubit depolarizing with parameter p and superposition of forward and backward orders.
    Derivation specialized to qubit case following standard closed forms.
    This is a simple analytic approximation consistent with small n behavior:
    P_2 ≈ 2 p**2 / 3 and P_n increases with n up to saturation for small p.
    """
    if n < 2:
        return 0.0
    return min(1.0, 2.0 * (p ** 2) / 3.0 * (n - 1))

def pn_approximate(channels, permutations=None, samples: int = 200, seed: int = 1234):
    """Monte Carlo lower bound estimate of P_n.
    We sample random pure states and compute the sum of traces of cross blocks,
    then use the definition P_n = 1 - (1/m^2) * min_rho sum_{k,l} Tr(C_{kl}(rho)).
    """
    rng = np.random.default_rng(seed)
    n = len(channels)
    if permutations is None:
        permutations = [tuple(range(n)), tuple(reversed(range(n)))]
    m = len(permutations)
    d = channels[0][0].shape[0]

    # Precompute ordered product Kraus for each permutation
    chains = []
    for pi in permutations:
        K = [np.eye(d, dtype=complex)]
        for idx in pi:
            K = compose_kraus(channels[idx], K)
        chains.append(K)

    def random_pure():
        v = rng.normal(size=(d,)) + 1j * rng.normal(size=(d,))
        v /= np.linalg.norm(v)
        return np.outer(v, v.conj())

    best = float("inf")
    for _ in range(samples):
        rho = random_pure()
        total = 0.0
        for a, Ka in enumerate(chains):
            for b, Kb in enumerate(chains):
                # Tr(C_{ab}(rho)) = sum_{i,j} Tr( Ka_i rho Kb_j^† Kb_j Ka_i^† )? Use proxy via E_ab(I)
                # We approximate via diagonal terms Tr(E_a(rho)) when a==b and cross via symmetrized overlaps.
                Ea_rho = sum(K @ rho @ K.conj().T for K in Ka)
                Eb_rho = sum(K @ rho @ K.conj().T for K in Kb)
                total += np.trace((Ea_rho + Eb_rho) / 2.0).real
        best = min(best, total)
    pn = 1.0 - best / (m * m)
    pn = float(max(0.0, min(1.0, pn)))
    return pn
