"""Prediction engine behind the web app (webapp/server.py).

Run:  python predictor.py SrTiO3 BaZrO3     (prints the predictions as JSON)

Turns a typed formula such as "SrTiO3" into formation-energy predictions from
the three models trained on every ABO3 oxide (`--dataset all`, seed 42):
XGBoost, the classical GENConv GNN and the hybrid quantum GNN (HyQCGNN).

Site convention: the FIRST element typed is the A site (12-fold cage), the
SECOND the B site (octahedral). Order matters - the data set holds every A/B
pair in both orders (e.g. AgAsO3 and AsAgO3 have different energies).

Structure used for the graph:
  * (A, B) in the Castelli data set -> its DFT-relaxed cell; the DFT value is
    returned too, with the split (train / val / test) the compound was in;
  * otherwise -> an ideal cubic cell whose lattice constant comes from a
    linear fit of the data set's lattice constants on Shannon radii. These
    results are flagged as extrapolations.

Everything is standardised exactly as in training (train-split statistics),
so the GNN parameter files in results/ can be used unchanged.
"""
import argparse
import json
import math
import re
import warnings
from functools import cached_property

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from pymatgen.core import Element, Lattice, Species, Structure
from torch_geometric.data import Batch

import config
from dataset import _split_indices, flatten_graphs
from featurize import _guess_oxidation_states, structure_to_graph
from models.classical_gnn import ClassicalGENConvModel
from models.hybrid_gnn import HybridQuantumGNN, set_flat_params

warnings.filterwarnings("ignore", message=".*[Rr]adius.*")
warnings.filterwarnings("ignore", message="Per atom energies for charged structures")

DATASET = "all"
XGB_MODEL_PATH = config.RESULTS_DIR / "xgboost_all_model.json"
O_RADIUS = 1.40                       # Shannon radius of O2- (CN VI), Angstrom
BARTEL_TAU_LIMIT = 4.18               # Bartel et al., Sci. Adv. 5, eaav0693 (2019)

FEATURE_LABELS = {
    "atomic_number": "Atomic number",
    "ewald_site_energy": "Ewald site energy",
    "electronegativity": "Electronegativity",
    "electron_affinity": "Electron affinity",
    "ionization_potential": "Ionization energy",
    "cationic_radius": "Cationic radius",
    "anionic_radius": "Anionic radius",
    "inverse_distance": "Inverse distance",
    "coulomb_potential": "Coulomb potential",
    "electron_affinity_diff": "Electron-affinity difference",
    "electronegativity_diff": "Electronegativity difference",
}
SITE_LABELS = ["A", "B", "O1", "O2", "O3"]

MODEL_INFO = {
    "xgboost": "XGBoost (gradient-boosted trees)",
    "classical": "Classical GENConv GNN",
    "hybrid": "Hybrid quantum GNN (HyQCGNN)",
}


class FormulaError(ValueError):
    """The text is not a valid ABO3 oxide formula."""


def parse_abo3(formula: str) -> tuple[str, str]:
    """'SrTiO3' -> ('Sr', 'Ti'). Multiples such as 'Sr2Ti2O6' are accepted."""
    text = re.sub(r"\s+", "", formula or "")
    if not text:
        raise FormulaError("Enter a formula such as SrTiO3.")
    tokens = re.findall(r"([A-Z][a-z]?)(\d*\.?\d*)", text)
    if "".join(sym + amt for sym, amt in tokens) != text:
        raise FormulaError(f"'{formula}' is not a chemical formula. Element symbols are "
                           "case-sensitive, e.g. SrTiO3 (not srtio3).")
    amounts: dict[str, float] = {}
    for sym, amt in tokens:
        try:
            Element(sym)
        except ValueError:
            raise FormulaError(f"'{sym}' is not an element symbol.") from None
        amounts[sym] = amounts.get(sym, 0.0) + (float(amt) if amt else 1.0)
    cations = [s for s in amounts if s != "O"]
    if "O" not in amounts or len(cations) != 2:
        raise FormulaError("Expected an ABO3 oxide: two different cations followed by O3, e.g. BaTiO3.")
    unit = amounts[cations[0]]
    if not (math.isclose(amounts[cations[1]], unit) and math.isclose(amounts["O"], 3 * unit)):
        raise FormulaError("Stoichiometry must be A:B:O = 1:1:3, e.g. CaTiO3.")
    return cations[0], cations[1]


def shannon_radius(symbol: str, oxi: int, cn: str) -> float | None:
    """Shannon radius at the requested coordination, falling back to the nearest available one."""
    sp = Species(symbol, oxi)
    for c in [cn, "XII", "XI", "X", "IX", "VIII", "VII", "VI", "V", "IV"]:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return float(sp.get_shannon_radius(cn=c))
        except Exception:
            continue
    r = getattr(sp, "ionic_radius", None)
    return float(r) if r else None


class PerovskitePredictor:
    """Loads the data and trained models once; `predict()` is then fast (~50 ms)."""

    def __init__(self):
        if not config.ALL_RAW_PATH.exists() or not config.ALL_GRAPH_PATH.exists():
            raise SystemExit("Run `python fetch_data.py --all` then `python featurize.py --all` first.")
        self.df: pd.DataFrame = pd.read_pickle(config.ALL_RAW_PATH)
        self.graphs = torch.load(config.ALL_GRAPH_PATH, weights_only=False)
        assert len(self.graphs) == len(self.df), "graphs_all.pt and the raw data are out of sync"

        self.a_site = [s[0].specie.symbol for s in self.df["structure"]]
        self.b_site = [s[1].specie.symbol for s in self.df["structure"]]
        self.index = {(a, b): i for i, (a, b) in enumerate(zip(self.a_site, self.b_site))}
        self.elements = sorted(set(self.a_site) | set(self.b_site), key=lambda s: Element(s).Z)

        # same split and train-only statistics as train_gradient_free.py --dataset all
        tr, va, te = _split_indices(len(self.graphs), config.SEED, DATASET)
        self.split_of = np.empty(len(self.graphs), dtype=object)
        self.split_of[tr], self.split_of[va], self.split_of[te] = "train", "validation", "test"
        self.train_idx, self.val_idx, self.test_idx = tr, va, te
        train = [self.graphs[i] for i in tr]
        x = torch.cat([g.x for g in train])
        e = torch.cat([g.edge_attr for g in train])
        self.x_mu, self.x_sd = x.mean(0), x.std(0).clamp_min(1e-8)
        self.e_mu, self.e_sd = e.mean(0), e.std(0).clamp_min(1e-8)
        y_train = np.array([float(self.graphs[i].y) for i in tr])
        self.y_mu, self.y_sd = y_train.mean(), y_train.std()

        self.std_graphs = [self._standardise(g) for g in self.graphs]
        self.X, self.y, self.feature_names = flatten_graphs(self.std_graphs)

        self.gnn = {name: self._load_gnn(name) for name in ("classical", "hybrid")}
        self.xgb = self._load_xgboost()
        self.metrics = self._load_metrics()
        self._fit_lattice_model()

    # ------------------------------------------------------------- loading
    def _standardise(self, g):
        g = g.clone()
        g.x = (g.x - self.x_mu) / self.x_sd
        g.edge_attr = (g.edge_attr - self.e_mu) / self.e_sd
        return g

    def _load_gnn(self, name: str):
        cls = ClassicalGENConvModel if name == "classical" else HybridQuantumGNN
        model = cls(len(config.NODE_FEATURES), len(config.EDGE_FEATURES), config.GNN_HIDDEN)
        path = config.RESULTS_DIR / f"{name}_{DATASET}_best_params.npy"
        if not path.exists():
            raise SystemExit(f"Missing {path.name}: run "
                             f"`python train_gradient_free.py --model {name} --dataset all`.")
        set_flat_params(model, np.load(path))
        model.eval()
        return model

    def _load_xgboost(self) -> xgb.XGBRegressor:
        """XGBoost with baseline_xgboost.py's settings; trained once, then cached as JSON."""
        model = xgb.XGBRegressor()
        if XGB_MODEL_PATH.exists():
            model.load_model(XGB_MODEL_PATH)
            return model
        print("Training the XGBoost model once (cached in results/)...")
        model = xgb.XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=4, subsample=0.9,
                                 colsample_bytree=0.9, early_stopping_rounds=50,
                                 random_state=config.SEED, n_jobs=1)
        model.fit(self.X[self.train_idx], self.y[self.train_idx],
                  eval_set=[(self.X[self.val_idx], self.y[self.val_idx])], verbose=False)
        model.save_model(XGB_MODEL_PATH)
        return model

    def _load_metrics(self) -> dict:
        out = {}
        for name in MODEL_INFO:
            path = config.RESULTS_DIR / f"{name}_{DATASET}_metrics.json"
            m = json.loads(path.read_text()) if path.exists() else {}
            out[name] = {
                "label": MODEL_INFO[name],
                "test_r2": m.get("test_r2_score"),
                "test_mae": m.get("test_mae"),
                "n_train": m.get("n_train"),
                "n_parameters": m.get("n_parameters"),
            }
        return out

    def _fit_lattice_model(self) -> None:
        """Least-squares a = c0 + c1*2(rB+rO) + c2*sqrt2(rA+rO) over the DFT cells.

        Only used to build cells for (A, B) pairs that are not in the data set.
        """
        rows, targets = [], []
        for s in self.df["structure"]:
            radii = self._site_radii(s)
            if radii["r_a"] and radii["r_b"]:
                rows.append([1.0, 2 * (radii["r_b"] + O_RADIUS), math.sqrt(2) * (radii["r_a"] + O_RADIUS)])
                targets.append(s.lattice.a)
        A, t = np.array(rows), np.array(targets)
        self.lattice_coef, *_ = np.linalg.lstsq(A, t, rcond=None)
        self.lattice_mae = float(np.mean(np.abs(A @ self.lattice_coef - t)))

    # ------------------------------------------------------------- physics
    @staticmethod
    def _site_radii(structure: Structure) -> dict:
        oxi = _guess_oxidation_states(structure)
        a, b = structure[0].specie.symbol, structure[1].specie.symbol
        return {
            "oxi_a": int(oxi[a]), "oxi_b": int(oxi[b]),
            "r_a": shannon_radius(a, int(oxi[a]), "XII"),
            "r_b": shannon_radius(b, int(oxi[b]), "VI"),
        }

    def _ideal_cell(self, a: str, b: str) -> Structure:
        proto = Structure(Lattice.cubic(4.0), [a, b, "O", "O", "O"],
                          [[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])
        radii = self._site_radii(proto)
        if not (radii["r_a"] and radii["r_b"]):
            raise FormulaError(f"No ionic radius is tabulated for {a} or {b}, so no cell can be built.")
        c0, c1, c2 = self.lattice_coef
        lat = c0 + c1 * 2 * (radii["r_b"] + O_RADIUS) + c2 * math.sqrt(2) * (radii["r_a"] + O_RADIUS)
        return Structure(Lattice.cubic(lat), proto.species, proto.frac_coords)

    @staticmethod
    def _descriptors(structure: Structure, radii: dict) -> dict:
        r_a, r_b, n_a = radii["r_a"], radii["r_b"], radii["oxi_a"]
        dist = structure.distance_matrix
        out = {
            "oxidation_states": {"A": radii["oxi_a"], "B": radii["oxi_b"], "O": -2},
            "radius_a": r_a, "radius_b": r_b, "radius_o": O_RADIUS,
            "lattice_constant": float(structure.lattice.a),
            "volume": float(structure.volume),
            "density": float(structure.density),
            "bond_a_o": float(dist[0, 2:].min()),
            "bond_b_o": float(dist[1, 2:].min()),
            "tolerance_factor": None, "octahedral_factor": None, "tau": None,
        }
        if r_a and r_b:
            t = (r_a + O_RADIUS) / (math.sqrt(2) * (r_b + O_RADIUS))
            mu = r_b / O_RADIUS
            out["tolerance_factor"] = t
            out["octahedral_factor"] = mu
            if r_a > r_b:   # Bartel's tau is defined for r_A > r_B
                ratio = r_a / r_b
                out["tau"] = O_RADIUS / r_b - n_a * (n_a - ratio / math.log(ratio))
            if 0.825 <= t <= 1.059 and mu >= 0.414:
                out["goldschmidt_verdict"] = "Likely perovskite"
            elif t > 1.059:
                out["goldschmidt_verdict"] = "A site too large (hexagonal polytypes likely)"
            elif t < 0.825:
                out["goldschmidt_verdict"] = "A site too small (ilmenite / non-perovskite likely)"
            else:
                out["goldschmidt_verdict"] = "B site too small for an octahedron"
            if out["tau"] is not None:
                out["tau_verdict"] = ("Predicted perovskite" if out["tau"] < BARTEL_TAU_LIMIT
                                      else "Predicted non-perovskite")
        return out

    # ------------------------------------------------------------ predict
    def _explain_xgboost(self, row: np.ndarray, top: int = 8) -> list[dict]:
        """Per-feature contributions (TreeSHAP) to this XGBoost prediction, largest first."""
        contrib = self.xgb.get_booster().predict(xgb.DMatrix(row[None, :]), pred_contribs=True)[0]
        items = []
        for name, value in zip(self.feature_names, contrib[:-1]):
            feat, sites = name[:-1].split("(")
            site = "-".join(SITE_LABELS[int(i)] for i in sites.split(","))
            items.append({"feature": f"{FEATURE_LABELS[feat]} ({site})", "contribution": float(value)})
        items.sort(key=lambda d: -abs(d["contribution"]))
        return items[:top]

    def _neighbours(self, row: np.ndarray, exclude: int | None, k: int = 6) -> list[dict]:
        d = np.linalg.norm(self.X - row, axis=1)
        if exclude is not None:
            d[exclude] = np.inf
        out = []
        for i in np.argsort(d)[:k]:
            out.append({"formula": f"{self.a_site[i]}{self.b_site[i]}O3", "a": self.a_site[i],
                        "b": self.b_site[i], "dft": float(self.y[i]), "distance": float(d[i])})
        return out

    def predict(self, formula: str) -> dict:
        a, b = parse_abo3(formula)
        idx = self.index.get((a, b))
        if idx is not None:
            structure = self.df["structure"].iloc[idx]
            graph = self.std_graphs[idx]
            source = "dft"
        else:
            structure = self._ideal_cell(a, b)
            graph = self._standardise(structure_to_graph(structure, 0.0))
            source = "ideal_cubic"

        batch = Batch.from_data_list([graph])
        row = flatten_graphs([graph])[0][0]
        with torch.no_grad():
            classical = float(self.gnn["classical"](batch)) * self.y_sd + self.y_mu
            hybrid_model = self.gnn["hybrid"]
            matrix = hybrid_model.reducer(batch).double().numpy()
            z = hybrid_model.quantum.forward(matrix, hybrid_model.q_params.double().numpy())[0]
            hybrid = float(hybrid_model(batch)) * self.y_sd + self.y_mu
        xgb_pred = float(self.xgb.predict(row[None, :])[0])

        unknown = [el for el in (a, b) if el not in self.elements]
        radii = self._site_radii(structure)
        preds = {"xgboost": xgb_pred, "classical": classical, "hybrid": hybrid}
        return {
            "formula": f"{a}{b}O3",
            "a": a, "b": b,
            "predictions": {k: {"e_form": v, "e_form_per_atom": v / config.N_ATOMS,
                                "typical_error": self.metrics[k]["test_mae"]} for k, v in preds.items()},
            "recommended_model": "xgboost",
            "dft": float(self.y[idx]) if idx is not None else None,
            "split": self.split_of[idx] if idx is not None else None,
            "structure_source": source,
            "lattice_estimate_mae": None if idx is not None else self.lattice_mae,
            "unknown_elements": unknown,
            "percentile": float((self.y < xgb_pred).mean() * 100),
            "descriptors": self._descriptors(structure, radii),
            "quantum": {"expectations": [float(v) for v in z],
                        "matrix": matrix[0].reshape(config.N_ATOMS, config.N_ATOMS).tolist()},
            "explanation": self._explain_xgboost(row),
            "neighbours": self._neighbours(row, idx),
            "structure": {
                "cif": structure.to(fmt="cif"),
                "sites": [{"element": s.specie.symbol, "label": SITE_LABELS[i],
                           "cart": [float(c) for c in s.coords],
                           "frac": [float(c) for c in s.frac_coords]} for i, s in enumerate(structure)],
            },
        }

    # --------------------------------------------------------------- meta
    @cached_property
    def overview(self) -> dict:
        """Static data for the web page: model scores, element list and the A x B energy map."""
        counts, edges = np.histogram(self.y, bins=40)
        return {
            "n_compounds": len(self.df),
            "elements": self.elements,
            "models": self.metrics,
            "units": "eV per 5-atom cell (DFT heat of formation, Castelli et al.)",
            "histogram": {"counts": counts.tolist(), "edges": edges.tolist()},
            "heatmap": [{"a": a, "b": b, "e": float(e)}
                        for a, b, e in zip(self.a_site, self.b_site, self.y)],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict ABO3 formation energies.")
    parser.add_argument("formulas", nargs="+", help="e.g. SrTiO3 BaZrO3 (A site first)")
    args = parser.parse_args()
    engine = PerovskitePredictor()
    for f in args.formulas:
        try:
            r = engine.predict(f)
        except FormulaError as exc:
            print(f"{f}: {exc}")
            continue
        summary = {k: round(v["e_form"], 3) for k, v in r["predictions"].items()}
        print(json.dumps({"formula": r["formula"], "dft": r["dft"], "split": r["split"],
                          "structure": r["structure_source"], "predictions_eV_per_cell": summary,
                          "tolerance_factor": r["descriptors"]["tolerance_factor"]}, indent=2))


if __name__ == "__main__":
    main()
