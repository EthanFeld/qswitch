import argparse
from .channels import (
    kraus_depolarizing_qubit,
    kraus_amplitude_damping_qubit,
    kraus_phase_damping_qubit,
    kraus_bb84_like_qubit,
)
from .switch_core import quantum_switch, forward_backward_permutations
from .error_maps import fit_pauli_diagonal_qubit
from .optimize import find_optimal_n
from .qec_simulation import logical_error_repetition_code

def demo():
    # two identical depolarizing channels with p=0.2
    p = 0.2
    channels = [kraus_depolarizing_qubit(p), kraus_depolarizing_qubit(p)]
    res = quantum_switch(channels, permutations=forward_backward_permutations(2))
    kp = res["branches"]["plus"]
    km = res["branches"]["minus"]
    pp = res["probabilities"]["plus"]
    pm = res["probabilities"]["minus"]

    pI0, px0, py0, pz0 = fit_pauli_diagonal_qubit(kraus_depolarizing_qubit(p))
    pIp, pxp, pyp, pzp = fit_pauli_diagonal_qubit(kp)

    print("original pauli probs:", (pI0, px0, py0, pz0))
    print("switched plus pauli probs:", (pIp, pxp, pyp, pzp))
    print("branch probabilities:", res["probabilities"])

    le_orig = logical_error_repetition_code(kraus_depolarizing_qubit(p), code_length=5)
    le_sw = logical_error_repetition_code(kp, code_length=5)
    print("logical error original L=5:", le_orig)
    print("logical error switched plus L=5:", le_sw)

def search_opt_n_dep(p=0.2):
    out = find_optimal_n(kraus_depolarizing_qubit, {"p": p}, n_min=2, n_max=10)
    print("best n:", out["best_n"])
    for n, score in out["scores"]:
        print(n, score)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["demo", "opt_n"])
    ap.add_argument("--p", type=float, default=0.2, help="depolarizing parameter for opt_n")
    args = ap.parse_args()
    if args.command == "demo":
        demo()
    elif args.command == "opt_n":
        search_opt_n_dep(p=args.p)

if __name__ == "__main__":
    main()
