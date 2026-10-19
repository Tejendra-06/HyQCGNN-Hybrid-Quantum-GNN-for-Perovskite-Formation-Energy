"""Step 2 - turn every crystal structure into a graph (paper Section II.A).

Run:  python featurize.py

Nodes  = the 5 atoms of the primitive cell, described by 7 electronic features:
         atomic number, Ewald site energy, electronegativity, electron affinity,
         first ionization potential, cationic radius, anionic radius.
Edges  = all 20 directed atom pairs (fully connected, homogeneous graph) with
         4 features: inverse Euclidean distance, Coulomb potential (off-diagonal
         Coulomb-matrix term Z_i*Z_j/r), electron-affinity difference and
         electronegativity difference.

Node order is fixed to [A, B, O, O, O] (A = larger cation) so that "atom 0",
"atom 1"... mean the same thing in every graph - this is what makes the
flattened XGBoost table and the 5x5 adjacency matrix comparable across samples.
"""
import numpy as np
import pandas as pd
import torch
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.core import Element, Structure
from torch_geometric.data import Data

import config


def _safe(value, default=0.0) -> float:
    return default if value is None or np.isnan(value) else float(value)


def _site_table(structure: Structure):
    """Return per-site feature rows (unordered) plus the oxidation-decorated structure."""
    decorated = structure.copy()
    decorated.add_oxidation_state_by_guess()  # needed for Ewald + ionic radii
    ewald = EwaldSummation(decorated)

    rows = []
    for i, site in enumerate(decorated):
        species = site.specie
        element = Element(species.symbol)
        oxi = species.oxi_state or 0
        ionic_radius = _safe(getattr(species, "ionic_radius", None))
        ion_energies = element.ionization_energies
        rows.append(
            dict(
                symbol=species.symbol,
                oxi=oxi,
                z=element.Z,
                ewald=float(ewald.get_site_energy(i)),
                chi=_safe(element.X),
                ea=_safe(element.electron_affinity),
                ie=_safe(ion_energies[0] if ion_energies else None),
                r_cat=ionic_radius if oxi > 0 else 0.0,
                r_an=ionic_radius if oxi < 0 else 0.0,
                original_index=i,
            )
        )
    return rows, decorated


def _order_sites(rows):
    """[A, B, O, O, O]: cations by decreasing ionic radius, then the oxygens."""
    cations = sorted((r for r in rows if r["symbol"] != "O"), key=lambda r: -r["r_cat"])
    oxygens = [r for r in rows if r["symbol"] == "O"]
    return cations + oxygens


def structure_to_graph(structure: Structure, target: float) -> Data:
    rows, decorated = _site_table(structure)
    ordered = _order_sites(rows)
    n = len(ordered)
    assert n == config.N_ATOMS, f"expected {config.N_ATOMS} atoms, got {n}"

    node_x = np.array(
        [[r["z"], r["ewald"], r["chi"], r["ea"], r["ie"], r["r_cat"], r["r_an"]] for r in ordered],
        dtype=np.float32,
    )

    # periodic minimum-image distances, re-indexed into our A,B,O,O,O order
    dist = decorated.distance_matrix
    perm = [r["original_index"] for r in ordered]
    dist = dist[np.ix_(perm, perm)]

    src, dst, edge_x = [], [], []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            d = dist[i, j]
            src.append(i)
            dst.append(j)
            edge_x.append(
                [
                    1.0 / d,                                     # inverse distance
                    ordered[i]["z"] * ordered[j]["z"] / d,       # Coulomb potential
                    abs(ordered[i]["ea"] - ordered[j]["ea"]),    # EA difference
                    abs(ordered[i]["chi"] - ordered[j]["chi"]),  # electronegativity diff
                ]
            )

    return Data(
        x=torch.tensor(node_x),
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.tensor(edge_x, dtype=torch.float32),
        y=torch.tensor([target], dtype=torch.float32),
    )


def main() -> None:
    if not config.RAW_PATH.exists():
        raise SystemExit("Run `python fetch_data.py` first.")
    df: pd.DataFrame = pd.read_pickle(config.RAW_PATH)

    graphs, skipped = [], 0
    for idx, row in df.iterrows():
        try:
            graphs.append(structure_to_graph(row["structure"], row[config.TARGET_COLUMN]))
        except Exception as exc:  # e.g. oxidation state guess failed
            skipped += 1
            print(f"  skipped structure {idx}: {exc}")

    torch.save(graphs, config.GRAPH_PATH)
    print(f"Saved {len(graphs)} graphs ({skipped} skipped) -> {config.GRAPH_PATH}")
    g = graphs[0]
    print(f"  node features {tuple(g.x.shape)}, edge features {tuple(g.edge_attr.shape)}")


if __name__ == "__main__":
    main()
