import numpy as np
from qswitch.channels import (
    kraus_depolarizing_qubit,
    kraus_pauli_channel,
    kraus_phase_damping_qubit,
    kraus_amplitude_damping_qubit,
    validate_kraus,
)

def test_kraus_cptp():
    for K in [
        kraus_depolarizing_qubit(0.3),
        kraus_pauli_channel(0.1, 0.05, 0.07),
        kraus_phase_damping_qubit(0.2),
        kraus_amplitude_damping_qubit(0.2),
    ]:
        assert validate_kraus(K)
