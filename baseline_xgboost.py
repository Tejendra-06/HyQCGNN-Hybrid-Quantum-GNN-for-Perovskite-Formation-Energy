"""XGBoost baseline + feature-importance analysis (paper Section II.C, Figs. 3c and 4).

Run:  python baseline_xgboost.py   [--features reduced] [--seed N] [--dataset all]

All graphs have the same size and connectivity, so each one is flattened into a
table row (35 node features + 40 upper-triangle edge features). XGBoost is
trained on the 196 training rows, early-stopped on the 25 validation rows and
scored once on the 25 test rows. It also writes the feature-importance bar
chart used to check that the engineered descriptors are physically sensible.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

import config
from dataset import flatten_graphs, load_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", choices=list(config.FEATURE_SETS), default=config.DEFAULT_FEATURE_SET)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--dataset", choices=["paper", "all"], default="paper")
    args = parser.parse_args()
    tag = "xgboost" if args.features == "full" else f"xgboost_{args.features}"
    if args.dataset != "paper":
        tag += f"_{args.dataset}"
    if args.seed != config.SEED:
        tag += f"_s{args.seed}"

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    splits = load_splits(args.features, args.seed, args.dataset)
    x_tr, y_tr, names = flatten_graphs(splits.train, args.features)
    x_va, y_va, _ = flatten_graphs(splits.val, args.features)
    x_te, y_te, _ = flatten_graphs(splits.test, args.features)

    model = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
        early_stopping_rounds=50,
        random_state=args.seed,
        n_jobs=1,  # results change with the thread count; one thread = same on every machine
    )
    model.fit(x_tr, y_tr, eval_set=[(x_va, y_va)], verbose=False)
    pred = model.predict(x_te)

    metrics = {
        "model": "xgboost",
        "features": args.features,
        "seed": args.seed,
        "dataset": args.dataset,
        "n_train": len(splits.train),
        "best_iteration": int(model.best_iteration),
        "test_mse": float(np.mean((pred - y_te) ** 2)),
        "test_mae": float(mean_absolute_error(y_te, pred)),
        "test_r2_linear_fit": float(np.corrcoef(pred, y_te)[0, 1] ** 2),
        "test_r2_score": float(r2_score(y_te, pred)),
    }
    print(json.dumps(metrics, indent=2))
    (config.RESULTS_DIR / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez(config.RESULTS_DIR / f"{tag}_predictions.npz", y_true=y_te, y_pred=pred)

    importances = pd.Series(model.feature_importances_, index=names).sort_values()
    (config.RESULTS_DIR / f"{tag}_importance.json").write_text(
        json.dumps({k: float(v) for k, v in importances.items()}, indent=2))
    top = importances.tail(20)
    fig, ax = plt.subplots(figsize=(7, 7))
    top.plot.barh(ax=ax, color="#4c9f50")
    ax.set_xlabel("Importance")
    ax.set_title(f"XGBoost feature importances (top 20, {args.features} features)")
    fig.tight_layout()
    fig.savefig(config.RESULTS_DIR / f"{tag}_feature_importance.png", dpi=150)
    print("Top features:\n", top.iloc[::-1].head(5).to_string())


if __name__ == "__main__":
    main()
