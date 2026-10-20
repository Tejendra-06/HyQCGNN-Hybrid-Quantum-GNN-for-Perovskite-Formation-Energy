"""Quantum layer (paper Section II.B.2, Fig. 2), built with Qiskit.

Pipeline for one input vector x (the flattened 5x5 matrix, 25 numbers):

  1. amplitude encoding  - x is zero-padded to 32 numbers and L2-normalised,
                           then used as the amplitudes of a 5-qubit state
                           (this is what Qiskit's StatePreparation does);
  2. trainable ansatz    - RX(theta_i), RY(phi_i) on every qubit, then a ring of
                           RZZ(eta_i) gates between neighbouring qubits
                           (q0-q1, q1-q2, q2-q3, q3-q4, q4-q0);
  3. trainable read-out  - RX, RY, RZ rotations on every qubit, i.e. the layer
                           "learns which observable to measure";
  4. measurement         - expectation values <Z_i> of the 5 qubits.

Speed trick: the trainable part (steps 2-3) does not depend on the input, so
for each set of parameters we build its 32x32 unitary U ONCE with Qiskit's
`Operator`, and apply it to every encoded input at the same time with a single
matrix product. The result is mathematically identical to running every
sample through the full circuit (checked in tests/test_quantum_layer.py), but
fast enough for 2000 gradient-free iterations over the whole training set.
Exact expectation values are used (no shot noise), like a statevector
simulation.
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Operator

import config

N = config.N_QUBITS
DIM = config.AMP_DIM
N_QUANTUM_PARAMS = 6 * N  # theta, phi, eta, read-out RX, RY, RZ  (5 each)


def build_trainable_circuit():
    """Ansatz + read-out. Returns (circuit, ordered parameter list)."""
    theta = ParameterVector("theta", N)
    phi = ParameterVector("phi", N)
    eta = ParameterVector("eta", N)
    ro_x = ParameterVector("ro_x", N)
    ro_y = ParameterVector("ro_y", N)
    ro_z = ParameterVector("ro_z", N)

    qc = QuantumCircuit(N, name="trainable")
    for q in range(N):          # ansatz: single-qubit rotations
        qc.rx(theta[q], q)
        qc.ry(phi[q], q)
    for q in range(N):          # ansatz: entangling ring of RZZ gates
        qc.rzz(eta[q], q, (q + 1) % N)
    for q in range(N):          # read-out layer
        qc.rx(ro_x[q], q)
        qc.ry(ro_y[q], q)
        qc.rz(ro_z[q], q)

    params = [*theta, *phi, *eta, *ro_x, *ro_y, *ro_z]
    return qc, params


def amplitude_encode(features: np.ndarray) -> np.ndarray:
    """(B, 25) -> (B, 32) unit-norm amplitude vectors (zero padded)."""
    features = np.asarray(features, dtype=np.float64)
    padded = np.zeros((features.shape[0], DIM))
    padded[:, : features.shape[1]] = features
    norms = np.linalg.norm(padded, axis=1, keepdims=True)
    empty = norms[:, 0] < 1e-12
    padded[empty, 0] = 1.0           # an all-zero input falls back to |00000>
    norms[empty] = 1.0
    return padded / norms


class QuantumLayer:
    """Maps (B, 25) classical features to (B, 5) expectation values <Z_i>."""

    def __init__(self):
        self.circuit, self.param_list = build_trainable_circuit()
        # sign table: <Z_i> = sum_b p(b) * (+1 if bit i of b is 0 else -1)
        # (Qiskit convention: qubit 0 is the least-significant bit)
        basis = np.arange(DIM)
        self.signs = np.stack([1 - 2 * ((basis >> q) & 1) for q in range(N)], axis=1)
        self._cached_params = None
        self._cached_unitary = None

    def unitary(self, params: np.ndarray) -> np.ndarray:
        # Training and validation passes of one optimiser iteration share the same
        # parameters, so cache the last U: the circuit is built once per iteration.
        params = np.asarray(params, dtype=np.float64)
        if self._cached_params is None or not np.array_equal(params, self._cached_params):
            bound = self.circuit.assign_parameters(dict(zip(self.param_list, map(float, params))))
            self._cached_unitary = Operator(bound).data
            self._cached_params = params.copy()
        return self._cached_unitary

    def forward(self, features: np.ndarray, params: np.ndarray) -> np.ndarray:
        states = amplitude_encode(features)             # (B, 32)
        out = states @ self.unitary(params).T           # psi' = U psi  (row-vector form)
        probs = np.abs(out) ** 2
        return probs @ self.signs                       # (B, 5)

    # ---- full circuit for a single input: used for tests / drawing / hardware
    def full_circuit(self, feature_vector: np.ndarray, params: np.ndarray) -> QuantumCircuit:
        amps = amplitude_encode(np.asarray(feature_vector)[None, :])[0]
        qc = QuantumCircuit(N)
        qc.append(StatePreparation(amps), range(N))
        bound = self.circuit.assign_parameters(dict(zip(self.param_list, map(float, params))))
        qc.compose(bound, inplace=True)
        return qc
