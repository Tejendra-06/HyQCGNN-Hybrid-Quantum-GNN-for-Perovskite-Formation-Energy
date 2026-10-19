"""Step 4 - parity plots (true vs predicted formation energy), like paper Fig. 3.

Run (after training the models you want to compare):  python evaluate.py

Reads results/<model>_predictions.npz for classical, hybrid and xgboost (any
that exist), draws one panel per model with a linear fit and its R^2, and
saves results/parity_plots.png.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

TITLES = {
    "classical": "GENConv (classical, gradient-free)",
    "hybrid": "HyQCGNN (hybrid quantum)",
    "xgboost": "XGBoost",
}


def main() -> None:
    available = [m for m in TITLES if (config.RESULTS_DIR / f"{m}_predictions.npz").exists()]
    if not available:
        raise SystemExit("No predictions found in results/. Train a model first.")

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4.5), squeeze=False)
    for ax, name in zip(axes[0], available):
        data = np.load(config.RESULTS_DIR / f"{name}_predictions.npz")
        y_true, y_pred = data["y_true"], data["y_pred"]
        slope, intercept = np.polyfit(y_pred, y_true, 1)
        r2 = np.corrcoef(y_pred, y_true)[0, 1] ** 2
        grid = np.linspace(y_pred.min(), y_pred.max(), 50)
        ax.scatter(y_pred, y_true, s=18, alpha=0.8)
        ax.plot(grid, slope * grid + intercept, color="C3")
        ax.set_xlabel("Predicted formation energy")
        ax.set_ylabel("True formation energy")
        ax.set_title(f"{TITLES[name]}\n$R^2$ = {r2:.3f}")
    fig.tight_layout()
    path = config.RESULTS_DIR / "parity_plots.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
