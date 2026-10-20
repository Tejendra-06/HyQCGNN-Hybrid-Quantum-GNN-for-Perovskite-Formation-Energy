"""Step 4 - parity plots (true vs predicted formation energy), like paper Fig. 3.

Run (after training the models you want to compare):  python evaluate.py

Reads results/<model>_predictions.npz for classical, hybrid and xgboost (and
their *_reduced variants; any that exist), draws one panel per model with a
linear fit and its R^2, and saves results/parity_plots.png.

Axes: true energy on x, predicted on y. The paper's Fig. 3 has the same layout
but labels its x-axis "Predicted" - its 3a/3b panels share identical x values
while y differs, so x must be the true value. R^2 is symmetric, so this does
not change the numbers.
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
    candidates = [(m, t) for m, t in TITLES.items()]
    candidates += [(f"{m}_reduced", f"{t} [reduced features]") for m, t in TITLES.items()]
    available = [(m, t) for m, t in candidates if (config.RESULTS_DIR / f"{m}_predictions.npz").exists()]
    if not available:
        raise SystemExit("No predictions found in results/. Train a model first.")

    fig, axes = plt.subplots(1, len(available), figsize=(5 * len(available), 4.5), squeeze=False)
    for ax, (name, title) in zip(axes[0], available):
        data = np.load(config.RESULTS_DIR / f"{name}_predictions.npz")
        y_true, y_pred = data["y_true"], data["y_pred"]
        slope, intercept = np.polyfit(y_true, y_pred, 1)
        r2 = np.corrcoef(y_true, y_pred)[0, 1] ** 2
        grid = np.linspace(y_true.min(), y_true.max(), 50)
        ax.scatter(y_true, y_pred, s=18, alpha=0.8)
        ax.plot(grid, slope * grid + intercept, color="C3")
        ax.set_xlabel("True formation energy")
        ax.set_ylabel("Predicted formation energy")
        ax.set_title(f"{title}\n$R^2$ = {r2:.3f}", fontsize=10)
    fig.tight_layout()
    path = config.RESULTS_DIR / "parity_plots.png"
    fig.savefig(path, dpi=150)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
