"""XGBoost learning curve on ALL ABO3 oxides (diagnostic, not part of the paper).

Run:  python xgboost_learning_curve.py

Question: is the gap to the paper's XGBoost R^2 (0.861) about how many / which
samples are used? We featurize every ABO3 oxide in matbench_perovskites (not
just the 246-sample subset), hold out a fixed random test set of N_TEST_LC
structures, and train XGBoost (baseline settings) on growing subsets of the
rest. 1/8 of each training subset is used for early stopping (the paper's
25/196 ratio). Repeated over config.SEEDS.

Featurized graphs are cached in data/graphs_all_oxides.pt (first run is slow).
Writes results/xgboost_learning_curve.json and .png.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import r2_score
from xgboost import XGBRegressor

import config
from dataset import flatten_graphs
from featurize import structure_to_graph
from fetch_data import is_abo3_oxide

ALL_GRAPHS_PATH = config.DATA_DIR / "graphs_all_oxides.pt"
N_TEST_LC = 500
TRAIN_SIZES = [196, 400, 800, 1600, None]  # None = every non-test structure


def load_all_graphs() -> list:
    if ALL_GRAPHS_PATH.exists():
        return torch.load(ALL_GRAPHS_PATH, weights_only=False)
    from matminer.datasets import load_dataset

    df = load_dataset(config.MATBENCH_NAME)
    df = df[df["structure"].apply(is_abo3_oxide)].reset_index(drop=True)
    print(f"Featurizing {len(df)} ABO3 oxides (cached after this run)...")
    graphs, skipped = [], 0
    for k, row in df.iterrows():
        try:
            graphs.append(structure_to_graph(row["structure"], row[config.TARGET_COLUMN]))
        except Exception:
            skipped += 1
        if (k + 1) % 250 == 0:
            print(f"  {k + 1}/{len(df)}")
    print(f"  {len(graphs)} graphs, {skipped} skipped")
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, ALL_GRAPHS_PATH)
    return graphs


def fit_and_score(x_tr, y_tr, x_te, y_te, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(y_tr))
    n_es = max(1, len(perm) // 8)
    es, fit = perm[:n_es], perm[n_es:]
    model = XGBRegressor(
        n_estimators=2000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        early_stopping_rounds=50,
        random_state=seed,
    )
    model.fit(x_tr[fit], y_tr[fit], eval_set=[(x_tr[es], y_tr[es])], verbose=False)
    pred = model.predict(x_te)
    return {
        "r2_linear_fit": float(np.corrcoef(pred, y_te)[0, 1] ** 2),
        "r2_score": float(r2_score(y_te, pred)),
        "mae": float(np.mean(np.abs(pred - y_te))),
    }


def main() -> None:
    graphs = load_all_graphs()
    x_all, y_all, _ = flatten_graphs(graphs)
    print(f"{len(y_all)} oxides, target mean {y_all.mean():.3f} std {y_all.std():.3f}")

    results = {}
    for seed in config.SEEDS:
        perm = np.random.default_rng(seed).permutation(len(y_all))
        test, pool = perm[:N_TEST_LC], perm[N_TEST_LC:]
        for size in TRAIN_SIZES:
            n = len(pool) if size is None else size
            train = pool[:n]
            scores = fit_and_score(x_all[train], y_all[train], x_all[test], y_all[test], seed)
            results.setdefault(n, []).append(scores)
            print(f"seed {seed:2d}  train {n:5d}  R2(fit) {scores['r2_linear_fit']:.3f}  "
                  f"R2 {scores['r2_score']:.3f}  MAE {scores['mae']:.3f}")

    summary = {
        n: {m: {"mean": float(np.mean([r[m] for r in runs])),
                "std": float(np.std([r[m] for r in runs], ddof=1))} for m in runs[0]}
        for n, runs in results.items()
    }
    print(f"\n{'train size':>10s}  {'R2 linear fit':>17s}  {'R2 score':>17s}  {'MAE':>17s}")
    for n, s in summary.items():
        print(f"{n:>10d}  " + "  ".join(f"{s[m]['mean']:>8.3f} +/- {s[m]['std']:.3f}"
                                         for m in ("r2_linear_fit", "r2_score", "mae")))

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "xgboost_learning_curve.json").write_text(
        json.dumps({"n_test": N_TEST_LC, "summary": summary}, indent=2))

    sizes = list(summary)
    means = np.array([summary[n]["r2_linear_fit"]["mean"] for n in sizes])
    stds = np.array([summary[n]["r2_linear_fit"]["std"] for n in sizes])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(sizes, means, yerr=stds, marker="o", capsize=3, label="XGBoost (this code)")
    ax.axhline(0.861, color="C3", ls="--", label="paper XGBoost (0.861)")
    ax.set_xscale("log")
    ax.set_xlabel("training structures")
    ax.set_ylabel(f"test $R^2$ (linear fit, {N_TEST_LC} held-out oxides)")
    ax.set_title("XGBoost learning curve, all ABO3 oxides")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / "xgboost_learning_curve.png", dpi=150)
    print(f"Saved {config.RESULTS_DIR / 'xgboost_learning_curve.png'}")


if __name__ == "__main__":
    main()
