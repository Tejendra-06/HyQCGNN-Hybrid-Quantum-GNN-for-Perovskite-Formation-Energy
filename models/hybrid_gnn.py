"""Hybrid quantum-classical GNN, HyQCGNN (paper Section II.B and workflow II.B.4).

    graph --GENConv--> 5x5 symmetric matrix --flatten--> 25 numbers
          --amplitude encoding + trainable ansatz + read-out--> <Z_0..Z_4>
          --trainable scale (w, b)--> predicted formation energy

All trainable numbers live in ONE flat vector (GENConv weights + 30 circuit
angles + 5 weights + 1 bias) because Nevergrad optimises a flat vector.
`get_flat_params` / `set_flat_params` convert between that vector and the
module. No gradients are ever needed, so the quantum part can be a plain
numpy function.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import config
from .classical_gnn import GENConvReducer
from .quantum_layer import N_QUANTUM_PARAMS, QuantumLayer


class HybridQuantumGNN(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden: int = config.GNN_HIDDEN):
        super().__init__()
        self.reducer = GENConvReducer(node_dim, edge_dim, hidden)
        self.quantum = QuantumLayer()
        self.q_params = nn.Parameter(torch.rand(N_QUANTUM_PARAMS) * 2 * np.pi)
        self.out_weight = nn.Parameter(torch.ones(config.N_QUBITS) / config.N_QUBITS)
        self.out_bias = nn.Parameter(torch.zeros(1))
        self.shots = None   # None = exact expectation values; an int = shot-noise estimate
        self.shot_rng = np.random.default_rng(config.SEED)

    @torch.no_grad()
    def forward(self, data) -> torch.Tensor:
        matrix = self.reducer(data).double().cpu().numpy()              # (B, 25)
        z = self.quantum.forward(matrix, self.q_params.double().cpu().numpy(),
                                 shots=self.shots, rng=self.shot_rng)   # (B, 5)
        z = torch.from_numpy(z).to(self.out_weight.dtype)
        return z @ self.out_weight + self.out_bias


def get_flat_params(model: nn.Module) -> np.ndarray:
    return parameters_to_vector(model.parameters()).detach().cpu().numpy().astype(np.float64)


def set_flat_params(model: nn.Module, vector: np.ndarray) -> None:
    vec = torch.as_tensor(np.asarray(vector), dtype=torch.float32)
    vector_to_parameters(vec, model.parameters())
