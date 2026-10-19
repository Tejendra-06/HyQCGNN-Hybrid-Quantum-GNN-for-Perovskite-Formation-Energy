"""Splitting and scaling of the graph data set (shared by every model).

`load_splits()` returns train / validation / test lists of PyG `Data` objects
(196 / 25 / 25 by default). Node and edge features are standardised with
statistics computed on the TRAINING graphs only, so nothing leaks from the
validation or test sets. `flatten_graphs()` produces the tabular view used by
the XGBoost baseline.
"""
from dataclasses import dataclass

import numpy as np
import torch
from torch_geometric.data import Batch, Data

import config


@dataclass
class Splits:
    train: list
    val: list
    test: list


def _split_indices(n: int):
    rng = np.random.default_rng(config.SEED)
    perm = rng.permutation(n)
    a = config.N_TRAIN
    b = a + config.N_VAL
    c = b + config.N_TEST
    return perm[:a], perm[a:b], perm[b:c]


def _standardise(graphs_by_split: dict, train_key: str = "train") -> None:
    train = graphs_by_split[train_key]
    x = torch.cat([g.x for g in train])
    e = torch.cat([g.edge_attr for g in train])
    x_mu, x_sd = x.mean(0), x.std(0).clamp_min(1e-8)
    e_mu, e_sd = e.mean(0), e.std(0).clamp_min(1e-8)
    for graphs in graphs_by_split.values():
        for g in graphs:
            g.x = (g.x - x_mu) / x_sd
            g.edge_attr = (g.edge_attr - e_mu) / e_sd


def load_splits() -> Splits:
    if not config.GRAPH_PATH.exists():
        raise SystemExit("Run `python fetch_data.py` then `python featurize.py` first.")
    graphs = torch.load(config.GRAPH_PATH, weights_only=False)
    graphs = graphs[: config.N_TOTAL]
    tr, va, te = _split_indices(len(graphs))
    by_split = {
        "train": [graphs[i].clone() for i in tr],
        "val": [graphs[i].clone() for i in va],
        "test": [graphs[i].clone() for i in te],
    }
    _standardise(by_split)
    return Splits(**by_split)


def as_batch(graphs: list) -> Batch:
    """Whole split as ONE batch - gradient-free training uses full-batch loss."""
    return Batch.from_data_list(graphs)


def flatten_graphs(graphs: list):
    """Tabular view for XGBoost: all node features + upper-triangle edge features.

    Returns (X, y, feature_names).
    """
    n = config.N_ATOMS
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    names = [f"{f}({i})" for i in range(n) for f in config.NODE_FEATURES]
    names += [f"{f}({i},{j})" for (i, j) in pairs for f in config.EDGE_FEATURES]

    rows, ys = [], []
    for g in graphs:
        edge_lookup = {
            (int(s), int(d)): g.edge_attr[k].numpy()
            for k, (s, d) in enumerate(g.edge_index.t().tolist())
        }
        node_part = g.x.numpy().reshape(-1)
        edge_part = np.concatenate([edge_lookup[p] for p in pairs])
        rows.append(np.concatenate([node_part, edge_part]))
        ys.append(float(g.y))
    return np.array(rows), np.array(ys), names
