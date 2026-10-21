"""Step 2 - turn every crystal structure into a graph (paper Section II.A).

Run:  python featurize.py          (data/graphs.pt from the 246-structure sample)
      python featurize.py --all    (data/graphs_all.pt from every ABO3 oxide)

Nodes  = the 5 atoms of the primitive cell, described by 7 electronic features:
         atomic number, Ewald site energy, electronegativity, electron affinity,
         first ionization potential, cationic radius, anionic radius.
Edges  = all 20 directed atom pairs (fully connected, homogeneous graph) with
         4 features: inverse Euclidean distance, Coulomb potential (off-diagonal
         Coulomb-matrix term Z_i*Z_j/r), electron-affinity difference and
         electronegativity difference.

Node order is fixed to [A, B, O, O, O] so that "atom 0",
"atom 1"... mean the same thing in every graph - this is what makes the
flattened XGBoost table and the 5x5 adjacency matrix comparable across samples.
The data set already stores sites in that order (A = 12-fold cage site, B = the
octahedral site; checked against the geometry of every undistorted cell), so the
order is taken from the structure itself rather than guessed from ionic radii.
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import torch
from pymatgen.analysis.ewald import EwaldSummation
from pymatgen.core import Element, Structure
from torch_geometric.data import Data

import config


def _safe(value, default=0.0) -> float:
    return default if value is None or np.isnan(value) else float(value)


def _guess_oxidation_states(structure: Structure) -> dict:
    """Oxidation state per element, with O fixed at -2.

    pymatgen's charge-balanced guess fails for about a third of the Castelli
    ABO3 oxides (many are hypothetical, e.g. CaScO3 cannot reach +6 on the
    cations) and then silently leaves every state at 0 - which zeroes the Ewald
    energies and ionic radii. Fallback: pick the pair of known cation states whose
    sum is closest to +6, preferring common states.
    """
    guesses = structure.composition.oxi_state_guesses()
    if guesses:
        return dict(guesses[0])
    a, b = [el for el in structure.composition.elements if el.symbol != "O"]

    def options(el):
        states = [s for s in el.oxidation_states if s > 0] or [1]
        return [(s, s not in el.common_oxidation_states) for s in states]

    best = min(
        ((pa, pb) for pa in options(a) for pb in options(b)),
        key=lambda p: (abs(p[0][0] + p[1][0] - 6), p[0][1] + p[1][1]),
    )
    return {a.symbol: best[0][0], b.symbol: best[1][0], "O": -2}


def _ionic_radius(species) -> float:
    """Shannon radius of the ion; the element's average cationic radius if unknown."""
    radius = _safe(getattr(species, "ionic_radius", None))
    if radius == 0.0 and species.oxi_state > 0:
        radius = _safe(Element(species.symbol).average_cationic_radius)
    return radius


def _site_table(structure: Structure):
    """Return per-site feature rows (unordered) plus the oxidation-decorated structure."""
    decorated = structure.copy()
    decorated.add_oxidation_state_by_element(_guess_oxidation_states(structure))
    ewald = EwaldSummation(decorated)  # needed for the Ewald site energies

    rows = []
    for i, site in enumerate(decorated):
        species = site.specie
        element = Element(species.symbol)
        oxi = species.oxi_state or 0
        ionic_radius = _ionic_radius(species)
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
    """[A, B, O, O, O]: cations in the data set's site order (A first), then the oxygens."""
    cations = [r for r in rows if r["symbol"] != "O"]
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="featurize every ABO3 oxide")
    args = parser.parse_args()
    raw_path = config.ALL_RAW_PATH if args.all else config.RAW_PATH
    graph_path = config.ALL_GRAPH_PATH if args.all else config.GRAPH_PATH
    if not raw_path.exists():
        raise SystemExit(f"Run `python fetch_data.py{' --all' if args.all else ''}` first.")
    df: pd.DataFrame = pd.read_pickle(raw_path)

    # pymatgen warns for every ion without a tabulated Shannon radius (handled above)
    warnings.filterwarnings("ignore", message=".*[Rr]adius.*")
    # 46 of the 246 sampled cells cannot be charge balanced with O at -2 (see above); for
    # those pymatgen returns the site energies without the uniform-background term
    warnings.filterwarnings("ignore", message="Per atom energies for charged structures")
    graphs, skipped = [], 0
    for idx, row in df.iterrows():
        try:
            graphs.append(structure_to_graph(row["structure"], row[config.TARGET_COLUMN]))
        except Exception as exc:  # e.g. oxidation state guess failed
            skipped += 1
            print(f"  skipped structure {idx}: {exc}")

    torch.save(graphs, graph_path)
    print(f"Saved {len(graphs)} graphs ({skipped} skipped) -> {graph_path}")
    g = graphs[0]
    print(f"  node features {tuple(g.x.shape)}, edge features {tuple(g.edge_attr.shape)}")


if __name__ == "__main__":
    main()
