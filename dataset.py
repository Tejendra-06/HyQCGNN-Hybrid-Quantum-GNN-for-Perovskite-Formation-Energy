"""Splitting and scaling of the graph data set (shared by every model).

`load_splits()` returns train / validation / test lists of PyG `Data` objects
(196 / 25 / 25 for the paper-sized data set, 80/10/10 % for `dataset="all"`),
keeping only the columns of the chosen feature set
(see config.FEATURE_SETS). Node and edge features are standardised with
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


def split_sizes(n: int, dataset: str = "paper"):
    if dataset == "paper":
        return config.N_TRAIN, config.N_VAL, config.N_TEST
    f_train, f_val, _ = config.ALL_SPLIT_FRACTIONS
    n_train, n_val = round(f_train * n), round(f_val * n)
    return n_train, n_val, n - n_train - n_val


def _split_indices(n: int, seed: int, dataset: str = "paper"):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_train, n_val, n_test = split_sizes(n, dataset)
    a = n_train
    b = a + n_val
    c = b + n_test
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


def _select_columns(g: Data, feature_set: str) -> Data:
    node_names, edge_names = config.FEATURE_SETS[feature_set]
    g = g.clone()
    g.x = g.x[:, [config.NODE_FEATURES.index(f) for f in node_names]]
    g.edge_attr = g.edge_attr[:, [config.EDGE_FEATURES.index(f) for f in edge_names]]
    return g


def load_splits(feature_set: str = config.DEFAULT_FEATURE_SET, seed: int = config.SEED,
                dataset: str = "paper") -> Splits:
    """dataset="paper": the 246-structure sample; "all": every ABO3 oxide."""
    if dataset == "paper":
        path, hint = config.GRAPH_PATH, "`python fetch_data.py` then `python featurize.py`"
    else:
        path, hint = config.ALL_GRAPH_PATH, "`python fetch_data.py --all` then `python featurize.py --all`"
    if not path.exists():
        raise SystemExit(f"Run {hint} first.")
    graphs = torch.load(path, weights_only=False)
    if dataset == "paper":
        graphs = graphs[: config.N_TOTAL]
    tr, va, te = _split_indices(len(graphs), seed, dataset)
    by_split = {
        "train": [_select_columns(graphs[i], feature_set) for i in tr],
        "val": [_select_columns(graphs[i], feature_set) for i in va],
        "test": [_select_columns(graphs[i], feature_set) for i in te],
    }
    _standardise(by_split)
    return Splits(**by_split)


def as_batch(graphs: list) -> Batch:
    """Whole split as ONE batch - gradient-free training uses full-batch loss."""
    return Batch.from_data_list(graphs)


def flatten_graphs(graphs: list, feature_set: str = config.DEFAULT_FEATURE_SET):
    """Tabular view for XGBoost: all node features + upper-triangle edge features.

    Returns (X, y, feature_names).
    """
    node_names, edge_names = config.FEATURE_SETS[feature_set]
    n = config.N_ATOMS
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    names = [f"{f}({i})" for i in range(n) for f in node_names]
    names += [f"{f}({i},{j})" for (i, j) in pairs for f in edge_names]

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
