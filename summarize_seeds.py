"""Mean +/- std of test metrics over repeated runs with different seeds.

Run (after training every model for every seed in config.SEEDS, e.g.
`python train_gradient_free.py --model hybrid --seed 1`):

    python summarize_seeds.py

Each seed gives a different 196/25/25 split, initialisation and optimiser run,
so the spread shows how much a single 25-sample test score can be trusted.
Prints a table and writes results/seed_summary.json.
"""
import json

import numpy as np

import config

VARIANTS = [
    "xgboost", "xgboost_tuned",
    "classical", "classical_noscale", "classical_it10000",
    "hybrid", "hybrid_noscale", "hybrid_it10000",
]
METRICS = ["test_r2_linear_fit", "test_r2_score", "test_mae"]


def _metrics_path(variant: str, seed: int):
    suffix = "" if seed == config.SEED else f"_s{seed}"
    return config.RESULTS_DIR / f"{variant}{suffix}_metrics.json"


def main() -> None:
    summary = {}
    for variant in VARIANTS:
        runs = [
            json.loads(path.read_text())
            for path in (_metrics_path(variant, s) for s in config.SEEDS)
            if path.exists()
        ]
        if not runs:
            continue
        summary[variant] = {"n_seeds": len(runs)}
        for m in METRICS:
            values = np.array([r[m] for r in runs])
            summary[variant][m] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "values": values.round(4).tolist(),
            }
    if not summary:
        raise SystemExit("No metrics found in results/. Train some models first.")

    print(f"{'model':20s} {'seeds':>5s}  " + "  ".join(f"{m:>22s}" for m in METRICS))
    for variant, s in summary.items():
        cells = "  ".join(f"{s[m]['mean']:>12.3f} +/- {s[m]['std']:.3f}" for m in METRICS)
        print(f"{variant:20s} {s['n_seeds']:>5d}  {cells}")

    out = config.RESULTS_DIR / "seed_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
