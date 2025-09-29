import numpy as np
import matplotlib.pyplot as plt
from .error_maps import pauli_transfer_qubit, fit_pauli_diagonal_qubit

def plot_pauli_eigs(kraus, title="Pauli transfer diagonal"):
    R = pauli_transfer_qubit(kraus)
    vals = [R[1,1], R[2,2], R[3,3]]
    plt.figure()
    plt.bar(["X","Y","Z"], vals)
    plt.ylim(-1, 1)
    plt.title(title)
    plt.xlabel("Axis")
    plt.ylabel("Eigenvalue")
    plt.tight_layout()

def compare_logical_error_curve(kraus_A, kraus_B, lengths=(3,5,7)):
    errs_A = []
    errs_B = []
    from .qec_simulation import logical_error_repetition_code
    for L in lengths:
        errs_A.append(logical_error_repetition_code(kraus_A, code_length=L))
        errs_B.append(logical_error_repetition_code(kraus_B, code_length=L))
    x = list(map(str, lengths))
    plt.figure()
    plt.plot(x, errs_A, marker="o", label="original")
    plt.plot(x, errs_B, marker="o", label="switched plus branch")
    plt.xlabel("repetition length")
    plt.ylabel("logical error rate")
    plt.legend()
    plt.tight_layout()
