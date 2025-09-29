from qswitch.channels import kraus_depolarizing_qubit
from qswitch.switch_core import quantum_switch, forward_backward_permutations

def test_switch_runs():
    chans = [kraus_depolarizing_qubit(0.2), kraus_depolarizing_qubit(0.2)]
    res = quantum_switch(chans, permutations=forward_backward_permutations(2))
    assert "branches" in res and "probabilities" in res
    assert len(res["branches"]["plus"]) > 0
    assert len(res["branches"]["minus"]) > 0
