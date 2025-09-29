import numpy as np
from .switch_core import quantum_switch, forward_backward_permutations
from .error_maps import pauli_transfer_qubit, fit_pauli_diagonal_qubit

def expected_holevo_qubit_pauli(kraus_plus, kraus_minus, p_plus, p_minus):
    """Approximate expected Holevo for qubit by using Pauli diagonal fit and
    the closed form capacity of a Pauli channel with optimal ensemble which equals
    1 - H2(pI + max(px,py,pz)) in a rough but useful proxy.
    This provides a consistent objective to compare n."""
    def capacity_proxy(kraus):
        pI, px, py, pz = fit_pauli_diagonal_qubit(kraus)
        p_nonid = px + py + pz
        # crude proxy: best of single axis signalling
        pmax = max(px, py, pz)
        # Shannon binary entropy
        def H2(x):
            x = min(max(x, 1e-12), 1-1e-12)
            return -x*np.log2(x) - (1-x)*np.log2(1-x)
        # proxy capacity in bits
        return 1.0 - H2(pI + pmax)
    Cplus = capacity_proxy(kraus_plus)
    Cminus = capacity_proxy(kraus_minus)
    return (1 - 0) * (p_plus * Cplus + p_minus * Cminus)

def find_optimal_n(channel_factory, factory_kwargs, n_min=2, n_max=12):
    """Search n for forward backward superposition that maximizes expected capacity proxy.
    channel_factory: function that returns a Kraus list for given parameters
    factory_kwargs: dict of kwargs to build each identical channel
    Returns dict with best_n and a table of scores.
    """
    scores = []
    for n in range(n_min, n_max + 1):
        channels = [channel_factory(**factory_kwargs) for _ in range(n)]
        res = quantum_switch(channels, permutations=forward_backward_permutations(n))
        kp = res["branches"]["plus"]
        km = res["branches"]["minus"]
        pp = res["probabilities"]["plus"]
        pm = res["probabilities"]["minus"]
        score = expected_holevo_qubit_pauli(kp, km, pp, pm)
        scores.append((n, score))
    best_n, best_score = max(scores, key=lambda t: t[1])
    return {"best_n": best_n, "scores": scores}
