"""Checks that the fast batched quantum layer equals the full Qiskit circuit.

Run:  pytest tests/ -q      (from the hyqcgnn/ folder)
"""
import sys
from pathlib import Path

import numpy as np
from qiskit.quantum_info import Statevector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from models.quantum_layer import N_QUANTUM_PARAMS, QuantumLayer, amplitude_encode  # noqa: E402


def test_amplitude_encoding_is_normalised():
    x = np.random.default_rng(0).normal(size=(4, 25))
    amps = amplitude_encode(x)
    assert amps.shape == (4, config.AMP_DIM)
    assert np.allclose(np.linalg.norm(amps, axis=1), 1.0)


def test_fast_path_matches_full_circuit():
    rng = np.random.default_rng(1)
    layer = QuantumLayer()
    x = rng.normal(size=(3, 25))
    params = rng.uniform(0, 2 * np.pi, size=N_QUANTUM_PARAMS)

    fast = layer.forward(x, params)
    for k in range(3):
        circuit = layer.full_circuit(x[k], params)
        state = Statevector(circuit)
        slow = [state.expectation_value(_z(q)).real for q in range(config.N_QUBITS)]
        assert np.allclose(fast[k], slow, atol=1e-8)


def _z(qubit: int):
    from qiskit.quantum_info import SparsePauliOp

    label = ["I"] * config.N_QUBITS
    label[config.N_QUBITS - 1 - qubit] = "Z"  # Qiskit labels are little-endian
    return SparsePauliOp("".join(label))


def test_expectations_in_range():
    layer = QuantumLayer()
    out = layer.forward(np.random.default_rng(2).normal(size=(5, 25)), np.zeros(N_QUANTUM_PARAMS))
    assert out.shape == (5, config.N_QUBITS)
    assert np.all(np.abs(out) <= 1 + 1e-9)
