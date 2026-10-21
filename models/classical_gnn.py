"""Classical GENConv model (paper Section II.B.1).

`GENConvReducer` is the part shared by the classical and the hybrid model. It
runs one GENConv layer (softmax aggregation with a learnable temperature,
message normalisation with a learnable scale, edge features included) and then
*reduces* the graph to one scalar per node and one scalar per edge. Those
scalars are arranged into a symmetric 5x5 matrix per graph (diagonal = node
scalars, off-diagonal = edge scalars), flattened to 25 numbers. This is the
"intermediate graph / weight matrix" of the paper's workflow (steps 3-4).

`ClassicalGENConvModel` = reducer + a single linear read-out (25 -> 1), the
classical counterpart to the quantum layer.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import GENConv

import config


class GENConvReducer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden: int = config.GNN_HIDDEN):
        super().__init__()
        self.n = config.N_ATOMS
        self.conv = GENConv(
            node_dim,
            hidden,
            aggr="softmax",        # generalised aggregation ...
            t=1.0,
            learn_t=True,          # ... with a learnable (inverse) temperature
            msg_norm=True,
            learn_msg_scale=True,  # message normalisation with learnable scale s
            num_layers=1,          # single fully connected layer (paper footnote 1)
            norm=None,
            edge_dim=edge_dim,
        )
        self.node_head = nn.Linear(hidden, 1)
        self.edge_head = nn.Linear(2 * hidden + edge_dim, 1)

    def forward(self, data) -> torch.Tensor:
        h = torch.tanh(self.conv(data.x, data.edge_index, data.edge_attr))
        node_scalar = self.node_head(h).squeeze(-1)                      # (N,)
        src, dst = data.edge_index
        edge_in = torch.cat([h[src], h[dst], data.edge_attr], dim=-1)
        edge_scalar = self.edge_head(edge_in).squeeze(-1)                # (E,)

        n_graphs = data.num_graphs
        graph_id = data.batch[src]
        local_src = src - data.ptr[graph_id]
        local_dst = dst - data.ptr[graph_id]

        mat = torch.zeros(n_graphs, self.n, self.n, dtype=edge_scalar.dtype, device=h.device)
        mat[graph_id, local_src, local_dst] = edge_scalar
        mat = 0.5 * (mat + mat.transpose(1, 2))                          # symmetric weights
        mat = mat + torch.diag_embed(node_scalar.view(n_graphs, self.n))  # node scalars on diagonal
        return mat.reshape(n_graphs, self.n * self.n)                    # (B, 25)


class ClassicalGENConvModel(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, hidden: int = config.GNN_HIDDEN):
        super().__init__()
        self.reducer = GENConvReducer(node_dim, edge_dim, hidden)
        self.readout = nn.Linear(config.N_ATOMS ** 2, 1)

    def forward(self, data) -> torch.Tensor:
        return self.readout(self.reducer(data)).squeeze(-1)
