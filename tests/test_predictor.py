"""Checks for the prediction engine behind the web app.

Run:  pytest tests/ -q      (from the quantum-project/ folder)

The engine tests need data/graphs_all.pt and the `_all` result files; they are
skipped if those are missing.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config  # noqa: E402
from predictor import FormulaError, PerovskitePredictor, parse_abo3  # noqa: E402


@pytest.mark.parametrize("text, expected", [
    ("SrTiO3", ("Sr", "Ti")),
    ("TiSrO3", ("Ti", "Sr")),          # order typed = site assignment
    ("Sr2Ti2O6", ("Sr", "Ti")),
    (" Ba Zr O3 ", ("Ba", "Zr")),
])
def test_parse_valid(text, expected):
    assert parse_abo3(text) == expected


@pytest.mark.parametrize("text", ["", "srtio3", "SrTiO2", "SrO3", "XxTiO3", "SrTiFeO3", "SrTiO3+"])
def test_parse_invalid(text):
    with pytest.raises(FormulaError):
        parse_abo3(text)


needs_data = pytest.mark.skipif(
    not (config.ALL_GRAPH_PATH.exists() and (config.RESULTS_DIR / "hybrid_all_best_params.npy").exists()),
    reason="needs the --dataset all graphs and trained parameters",
)


@pytest.fixture(scope="module")
def engine():
    return PerovskitePredictor()


@needs_data
def test_engine_reproduces_reported_test_scores(engine):
    """The app must serve exactly the models described in results/*_all_metrics.json."""
    import torch
    from torch_geometric.data import Batch

    te = engine.test_idx
    y = engine.y[te]
    batch = Batch.from_data_list([engine.std_graphs[i] for i in te])
    with torch.no_grad():
        preds = {
            "xgboost": engine.xgb.predict(engine.X[te]),
            "classical": engine.gnn["classical"](batch).numpy() * engine.y_sd + engine.y_mu,
            "hybrid": engine.gnn["hybrid"](batch).numpy() * engine.y_sd + engine.y_mu,
        }
    for name, p in preds.items():
        assert r2_score(y, p) == pytest.approx(engine.metrics[name]["test_r2"], abs=1e-4), name


@needs_data
def test_known_compound_uses_dft_cell(engine):
    r = engine.predict("SrTiO3")
    assert r["structure_source"] == "dft"
    assert r["dft"] == pytest.approx(-0.24, abs=1e-3)
    assert r["descriptors"]["tolerance_factor"] == pytest.approx(1.0, abs=0.01)
    assert len(r["quantum"]["expectations"]) == config.N_QUBITS


@needs_data
def test_site_order_matters(engine):
    assert engine.predict("SrTiO3")["dft"] != engine.predict("TiSrO3")["dft"]


@needs_data
def test_unknown_element_is_flagged(engine):
    r = engine.predict("NdNiO3")
    assert r["structure_source"] == "ideal_cubic"
    assert r["unknown_elements"] == ["Nd"]
    assert np.isfinite(r["predictions"]["xgboost"]["e_form"])
    assert 3.3 < r["descriptors"]["lattice_constant"] < 5.1
