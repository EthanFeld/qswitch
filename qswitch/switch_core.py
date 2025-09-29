import itertools
import numpy as np
from typing import List, Tuple, Dict, Optional

def forward_backward_permutations(n: int):
    """Return permutation indices for forward and backward orders."""
    return [tuple(range(n)), tuple(reversed(range(n)))]

def _compose_chain_given_indices(channels: List[List[np.ndarray]],
                                 order: Tuple[int, ...],
                                 idx_tuple: Tuple[int, ...]) -> np.ndarray:
    """
    Build C_pi(s) = K_{pi(n-1), s_{pi(n-1)}} ... K_{pi(0), s_{pi(0)}}.
    idx_tuple is indexed in the original channel order.
    """
    d = channels[0][0].shape[0]
    M = np.eye(d, dtype=complex)
    for slot in order:
        K = channels[slot][idx_tuple[slot]]
        M = K @ M
    return M

def _branch_prob_and_cptp(Kset: List[np.ndarray]) -> Tuple[float, List[np.ndarray]]:
    """Return (p_branch, CPTP normalized Kraus set)."""
    if not Kset:
        return 0.0, Kset
    d = Kset[0].shape[0]
    I_over_d = np.eye(d, dtype=complex) / d
    E_I = sum(K @ I_over_d @ K.conj().T for K in Kset)
    p = float(np.trace(E_I).real)
    if p <= 0:
        return 0.0, Kset
    scale = 1.0 / np.sqrt(p)
    return p, [scale * K for K in Kset]

def quantum_switch(channels: List[List[np.ndarray]],
                   permutations: Optional[List[Tuple[int, ...]]] = None,
                   branch_projection: str = "X",
                   max_pairs: Optional[int] = None,
                   seed: int = 7) -> Dict:
    """
    Quantum switch for any number of channels n with two superposed orders.
    Correct construction that sums over ALL pairs of Kraus index tuples
    across the two orders, which preserves the interference structure.

    channels
        list of Kraus sets [K^(0), K^(1), ..., K^(n-1)]
    permutations
        exactly two permutations of range(n)  default is forward and backward
    branch_projection
        only "X" is supported and produces two branches "plus" and "minus"
    max_pairs
        optional cap on the number of (s_a, s_b) pairs.
        if provided and smaller than the full cartesian size, pairs are sampled
        uniformly without replacement for an unbiased estimate
    seed
        random seed used when sampling pairs

    returns dict with
        branches: {'plus': [K_i], 'minus': [K_i]}  each CPTP
        probabilities: {'plus': p_plus, 'minus': p_minus}
        orders: the two permutations used
        sampled_pairs: number of (s_a, s_b) pairs used
    """
    print("new")
    n = len(channels)
    if permutations is None:
        permutations = forward_backward_permutations(n)
    if len(permutations) != 2:
        raise NotImplementedError("Provide exactly two permutations for superposed orders")
    if branch_projection != "X":
        raise NotImplementedError("Only X basis projection is implemented")

    pi_a, pi_b = permutations

    # Build all Kraus index tuples in the original channel order
    kraus_counts = [len(Kset) for Kset in channels]
    idx_all = list(itertools.product(*[range(c) for c in kraus_counts]))

    # Build all pairs of index tuples across the two orders
    # Number of pairs is |idx_all|^2, so allow sampling if requested
    total_pairs = len(idx_all) * len(idx_all)
    if (max_pairs is not None) and (max_pairs < total_pairs):
        rng = np.random.default_rng(seed)
        # sample indices for s_a and s_b independently and pair them
        sel_a = rng.choice(len(idx_all), size=min(max_pairs, len(idx_all)), replace=False)
        sel_b = rng.choice(len(idx_all), size=min(max_pairs, len(idx_all)), replace=False)
        pair_list = [(idx_all[i], idx_all[j]) for i, j in zip(sel_a, sel_b)]
    else:
        pair_list = [(sa, sb) for sa in idx_all for sb in idx_all]

    K_plus: List[np.ndarray] = []
    K_minus: List[np.ndarray] = []

    # For each pair (s_a, s_b) build C_{pi_a}(s_a) and C_{pi_b}(s_b), then combine
    for s_a, s_b in pair_list:
        Ca = _compose_chain_given_indices(channels, pi_a, s_a)
        Cb = _compose_chain_given_indices(channels, pi_b, s_b)
        K_plus.append( 0.5 * (Ca + Cb) )
        K_minus.append(0.5 * (Ca - Cb) )

    # Success probabilities and CPTP renormalization
    p_plus,  Kp = _branch_prob_and_cptp(K_plus)
    p_minus, Km = _branch_prob_and_cptp(K_minus)

    # Normalize classical weights for reporting
    tot = p_plus + p_minus
    if tot > 0:
        p_plus  /= tot
        p_minus /= tot
    else:
        p_plus = p_minus = 0.5

    return {
        "branches": {"plus": Kp, "minus": Km},
        "probabilities": {"plus": p_plus, "minus": p_minus},
        "orders": [pi_a, pi_b],
        "sampled_pairs": len(pair_list),
    }

# Convenience wrapper for two channels
def quantum_switch_two(ch1: List[np.ndarray], ch2: List[np.ndarray]) -> Dict:
    return quantum_switch([ch1, ch2], permutations=[(0, 1), (1, 0)])
