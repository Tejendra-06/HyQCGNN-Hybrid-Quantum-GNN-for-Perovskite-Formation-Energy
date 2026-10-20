"""Hyper-parameter search for the XGBoost baseline.

Run:  python tune_xgboost.py   [--seed N] [--n-iter 60]

Checks whether the gap to the paper's XGBoost R^2 (0.861) is just untuned
settings. A randomised search picks hyper-parameters by 5-fold cross-validation
on the 196 TRAINING rows only (25 validation rows are too few to choose
settings reliably), refits on the training rows and scores the test rows once.
Writes results/xgboost_tuned[_s<seed>]_metrics.json and _predictions.npz.
"""
import argparse
import json

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV
from xgboost import XGBRegressor

import config
from dataset import flatten_graphs, load_splits

SEARCH_SPACE = {
    "n_estimators": [100, 200, 400, 800, 1500],
    "learning_rate": [0.01, 0.02, 0.05, 0.1, 0.2],
    "max_depth": [2, 3, 4, 5, 6, 8],
    "min_child_weight": [1, 2, 4, 8],
    "subsample": [0.6, 0.8, 1.0],
    "colsample_bytree": [0.3, 0.5, 0.7, 0.9, 1.0],
    "reg_lambda": [0.1, 1.0, 5.0, 20.0],
    "gamma": [0.0, 0.01, 0.1],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--n-iter", type=int, default=60)
    args = parser.parse_args()
    tag = "xgboost_tuned" if args.seed == config.SEED else f"xgboost_tuned_s{args.seed}"

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    splits = load_splits(seed=args.seed)
    x_tr, y_tr, _ = flatten_graphs(splits.train)
    x_te, y_te, _ = flatten_graphs(splits.test)

    search = RandomizedSearchCV(
        XGBRegressor(random_state=args.seed, n_jobs=1),
        SEARCH_SPACE,
        n_iter=args.n_iter,
        scoring="neg_mean_squared_error",
        cv=KFold(5, shuffle=True, random_state=args.seed),
        random_state=args.seed,
        n_jobs=-1,
    )
    search.fit(x_tr, y_tr)
    pred = search.best_estimator_.predict(x_te)

    metrics = {
        "model": "xgboost_tuned",
        "seed": args.seed,
        "cv_mse": float(-search.best_score_),
        "best_params": search.best_params_,
        "test_mse": float(np.mean((pred - y_te) ** 2)),
        "test_mae": float(mean_absolute_error(y_te, pred)),
        "test_r2_linear_fit": float(np.corrcoef(pred, y_te)[0, 1] ** 2),
        "test_r2_score": float(r2_score(y_te, pred)),
    }
    print(json.dumps(metrics, indent=2))
    (config.RESULTS_DIR / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
    np.savez(config.RESULTS_DIR / f"{tag}_predictions.npz", y_true=y_te, y_pred=pred)


if __name__ == "__main__":
    main()
