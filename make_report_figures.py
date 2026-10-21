"""Result figures and numbers for the project report.

Run (after every experiment in the report, see README):  python make_report_figures.py

Reads results/ and writes into report_figures/:
  fig_parity.png         true vs predicted, seed 42 (report Figure 8)
  fig_seeds.png          test R^2 per seed and mean +/- std (Figure 9)
  fig_ablations.png      optimiser, feature set and data size, seed 42 (Figure 10)
  fig_learning_curve.png XGBoost R^2 vs training-set size (Figure 11)
  fig_importance.png     XGBoost gain importance over 5 seeds (Figure 12)
  report_numbers.json    every number quoted in the report

The diagrams (workflow, graph encoding, architecture, circuit, training loop) and
the data / training-curve figures do not depend on these runs and are not redrawn.
"Mean +/- std" uses the population std (ddof=0) over config.SEEDS, as in the report.
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config

OUT = config.ROOT / "report_figures"
RES = config.RESULTS_DIR

INK, MUTED, AXIS, GRID = "#1f2937", "#6b7280", "#9ca3af", "#e5e7eb"
XGB, CLS, HYB, GREY = "#4a5568", "#E69F00", "#0072B2", "#9ca3af"
PAPER = {"xgboost": 0.861, "classical": 0.759, "hybrid": 0.626}
SITE = ["A", "B", "O1", "O2", "O3"]

plt.rcParams.update({
    "font.size": 11, "axes.edgecolor": AXIS, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def metrics(tag: str) -> dict:
    return json.loads((RES / f"{tag}_metrics.json").read_text())


def seed_tag(variant: str, seed: int) -> str:
    return variant if seed == config.SEED else f"{variant}_s{seed}"


def title(ax, text, sub=None):
    ax.set_title(text, loc="left", fontweight="bold", fontsize=13, pad=22 if sub else 8)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, color=MUTED, fontsize=10)


def seed_stats(variant: str, key: str = "test_r2_linear_fit") -> dict:
    values = np.array([metrics(seed_tag(variant, s))[key] for s in config.SEEDS])
    return {"mean": float(values.mean()), "std": float(values.std()), "values": values.round(4).tolist()}


# ------------------------------------------------------------------ parity
def fig_parity(numbers):
    panels = [("xgboost", "XGBoost", XGB), ("classical", "Classical GNN", CLS), ("hybrid", "Hybrid QGNN", HYB)]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    lo, hi = -0.4, 3.3
    for ax, (tag, name, color) in zip(axes, panels):
        d = np.load(RES / f"{tag}_predictions.npz")
        yt, yp = d["y_true"], d["y_pred"]
        m = metrics(tag)
        slope, icpt = np.polyfit(yt, yp, 1)
        ax.plot([lo, hi], [lo, hi], "--", color="#cbd5e1", lw=1, label="y = x (perfect)")
        ax.plot([lo, hi], [slope * lo + icpt, slope * hi + icpt], color=color, lw=2, label="linear fit")
        ax.scatter(yt, yp, s=30, color=color, edgecolor="white", lw=0.6, zorder=3)
        ax.set(xlim=(lo, hi), ylim=(lo, hi), xlabel="True formation energy (eV)")
        title(ax, f"{name}: R² = {m['test_r2_linear_fit']:.3f}",
              f"paper: {PAPER[tag]} · MAE {m['test_mae']:.3f} eV · {len(yt)} test crystals")
    axes[0].set_ylabel("Predicted formation energy (eV)")
    axes[0].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_parity.png")
    plt.close(fig)


# ------------------------------------------------------------------- seeds
SEED_VARIANTS = [
    ("xgboost", "XGBoost", XGB, 1.0, "xgboost"),
    ("xgboost_tuned", "XGBoost (tuned)", XGB, 0.6, None),
    ("classical", "Classical GNN", CLS, 1.0, "classical"),
    ("classical_it10000", "Classical GNN\n10k its", CLS, 0.6, None),
    ("hybrid", "Hybrid QGNN", HYB, 1.0, "hybrid"),
    ("hybrid_it10000", "Hybrid QGNN\n10k its", HYB, 0.8, None),
    ("hybrid_frozenq", "Hybrid\nfrozen circuit", HYB, 0.5, None),
]


def fig_seeds(numbers):
    fig, ax = plt.subplots(figsize=(11, 4.6))
    rng = np.random.default_rng(0)
    for i, (variant, label, color, alpha, paper) in enumerate(SEED_VARIANTS):
        s = numbers["seed_summary"][variant]["test_r2_linear_fit"]
        jitter = rng.uniform(-0.12, 0.12, len(s["values"]))
        ax.scatter(i + jitter, s["values"], s=28, color=color, alpha=0.45 * alpha + 0.15, zorder=2)
        ax.errorbar(i + 0.28, s["mean"], yerr=s["std"], fmt="o", ms=9, color=color, alpha=alpha,
                    capsize=4, lw=2, zorder=3)
        ax.text(i + 0.42, s["mean"], f"{s['mean']:.2f}", va="center", fontweight="bold")
        if paper:
            ax.scatter(i, PAPER[paper], marker="D", s=70, facecolor="white", edgecolor=INK, lw=1.5, zorder=4)
            ax.text(i, PAPER[paper] + 0.045, f"paper {PAPER[paper]}", ha="center", color=MUTED, fontsize=10)
    ax.set_xticks(range(len(SEED_VARIANTS)), [v[1] for v in SEED_VARIANTS])
    ax.set(ylim=(0, 1), ylabel="Test R² (linear fit)")
    ax.yaxis.grid(color=GRID)
    ax.set_axisbelow(True)
    s = numbers["seed_summary"]
    h, c = s["hybrid"]["test_r2_linear_fit"]["mean"], s["classical"]["test_r2_linear_fit"]["mean"]
    leader = "beats" if h > c else "trails"
    title(ax, f"Across {len(config.SEEDS)} seeds the hybrid (R² {h:.2f}) {leader} the classical GNN "
              f"({c:.2f}); XGBoost leads",
          "small dots = individual seeds · filled circle = mean ± std · "
          "hollow diamond = value reported in the paper")
    fig.tight_layout()
    fig.savefig(OUT / "fig_seeds.png")
    plt.close(fig)


# --------------------------------------------------------------- ablations
def bars(ax, x, h, color, alpha=1.0, label=None, width=0.36):
    b = ax.bar(x, h, width, color=color, alpha=alpha, label=label)
    for rect, v in zip(b, h):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)


def fig_ablations(numbers):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    ax = axes[0]
    opt = numbers["optimizers"]
    names = list(opt)
    b = ax.bar(range(len(names)), list(opt.values()), color=[HYB] + [GREY] * (len(names) - 1))
    for rect, v in zip(b, opt.values()):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=10)
    ax.set_xticks(range(len(names)), names, rotation=25, ha="right", fontsize=10)
    ax.set(ylim=(0, 1), ylabel="Test R² (linear fit)")
    title(ax, "Hybrid: optimiser", "seed 42, 2,000 evaluations each")

    models = [("xgboost", "XGBoost", XGB), ("classical", "Classical GNN", CLS), ("hybrid", "Hybrid QGNN", HYB)]
    x = np.arange(3)
    ax = axes[1]
    feats = numbers["features"]
    for i, (k, _, color) in enumerate(models):
        bars(ax, [x[i] - 0.18], [feats[k]["full"]], color, label="full (7 node + 4 edge)" if i == 0 else None)
        bars(ax, [x[i] + 0.18], [feats[k]["reduced"]], color, 0.4,
             label="reduced (2 node + 1 edge)" if i == 0 else None)
    ax.set_xticks(x, [m[1] for m in models], fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    title(ax, "Feature set", "seed 42 · solid = full, faded = reduced")

    ax = axes[2]
    alld = numbers["all_data"]
    n_all = alld["hybrid"]["n_train"]
    for i, (k, _, color) in enumerate(models):
        bars(ax, [x[i] - 0.18], [alld[k]["paper_split"]], color, 0.4,
             label="196 training crystals" if i == 0 else None)
        bars(ax, [x[i] + 0.18], [alld[k]["all_data"]], color,
             label=f"{n_all:,} training crystals" if i == 0 else None)
    ax.set_xticks(x, [m[1] for m in models], fontsize=10)
    ax.set_ylim(0, 1)
    ax.legend(frameon=False, loc="upper right", fontsize=9)
    title(ax, "Data size", "seed 42 · faded = paper split, solid = all ABO3 (80/10/10)")
    fig.tight_layout()
    fig.savefig(OUT / "fig_ablations.png")
    plt.close(fig)


# ---------------------------------------------------------- learning curve
def fig_learning_curve(numbers):
    lc = numbers["learning_curve"]
    sizes = [int(k) for k in lc]
    mean = np.array([lc[k]["mean"] for k in lc])
    std = np.array([lc[k]["std"] for k in lc])
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.errorbar(sizes, mean, yerr=std, fmt="-o", color=XGB, lw=2.5, ms=7, capsize=5)
    for n, m in zip(sizes, mean):
        ax.text(n * 1.03, m - 0.02, f"{m:.2f}", va="top")
    ax.axhline(PAPER["xgboost"], ls="--", color=MUTED, lw=1.2)
    ax.text(sizes[-1], PAPER["xgboost"] + 0.008, f"paper's XGBoost R² = {PAPER['xgboost']}",
            ha="right", color=MUTED)
    ax.axvline(sizes[0], ls=":", color=HYB, lw=1.2)
    ax.text(sizes[0] * 1.03, 0.62, f"paper-sized training set ({sizes[0]})", color=HYB)
    ax.set_xscale("log")
    ax.set_xticks(sizes, [f"{n:,}" for n in sizes])
    ax.minorticks_off()
    ax.set(ylim=(0.6, 1.0), xlabel="Training crystals (log scale)", ylabel="Test R² (linear fit)")
    ax.yaxis.grid(color=GRID)
    ax.set_axisbelow(True)
    above = [n for n, m in zip(sizes, mean) if m >= PAPER["xgboost"]]
    head = (f"XGBoost passes the paper's {PAPER['xgboost']} once it sees about {above[0]:,} crystals"
            if above else f"XGBoost stays below the paper's {PAPER['xgboost']}")
    title(ax, head, f"fixed test set of {numbers['learning_curve_n_test']} ABO3 oxides · "
                    f"{len(config.SEEDS)} seeds · mean ± std")
    fig.tight_layout()
    fig.savefig(OUT / "fig_learning_curve.png")
    plt.close(fig)


# -------------------------------------------------------------- importance
def pretty(name: str) -> str:
    feature, idx = name[:-1].split("(")
    nodes = [int(i) for i in idx.split(",")]
    feature = feature.replace("_", " ")
    if len(nodes) == 1:
        return f"{feature} (site {SITE[nodes[0]]})"
    return f"{feature} ({SITE[nodes[0]]}–{SITE[nodes[1]]})"


def fig_importance(numbers):
    top = numbers["importance_top"][:15]
    by_type = numbers["importance_by_type"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"wspace": 0.75})
    labels = [pretty(n) for n, _ in top][::-1]
    vals = [v for _, v in top][::-1]
    colors = [HYB if n == "ionization_potential(0)" else GREY for n, _ in top][::-1]
    a1.barh(labels, vals, color=colors)
    for i, v in enumerate(vals):
        a1.text(v + 0.003, i, f"{v:.3f}", va="center", fontsize=9)
    a1.set_xlabel(f"Mean gain importance ({len(config.SEEDS)} seeds)")
    title(a1, "Top 15 single features", "blue = the paper's top feature (A-site ionisation energy)")
    types = list(by_type)[::-1]
    share = [by_type[t] for t in types]
    a2.barh([t.replace("_", " ") for t in types], share, color=GREY)
    for i, v in enumerate(share):
        a2.text(v + 0.004, i, f"{v:.0%}", va="center", fontsize=10)
    a2.set_xlabel("Share of total importance")
    title(a2, "Summed by feature type")
    fig.savefig(OUT / "fig_importance.png")
    plt.close(fig)


def collect() -> dict:
    numbers = {"seed_summary": {}}
    for variant in ["xgboost", "xgboost_tuned", "classical", "classical_it10000",
                    "hybrid", "hybrid_it10000", "hybrid_frozenq"]:
        numbers["seed_summary"][variant] = {
            key: seed_stats(variant, key) for key in ["test_r2_linear_fit", "test_r2_score", "test_mae"]
        }
    for variant in ["hybrid", "hybrid_it10000", "hybrid_frozenq"]:
        numbers["seed_summary"][variant]["test_r2_linear_fit_shots"] = seed_stats(variant, "test_r2_linear_fit_shots")

    numbers["optimizers"] = {
        "NGOpt (default)": metrics("hybrid")["test_r2_linear_fit"],
        "CMA-ES": metrics("hybrid_cma")["test_r2_linear_fit"],
        "TwoPointsDE": metrics("hybrid_twopointsde")["test_r2_linear_fit"],
        "PSO": metrics("hybrid_pso")["test_r2_linear_fit"],
        "(1+1)-ES": metrics("hybrid_oneplusone")["test_r2_linear_fit"],
    }
    numbers["features"] = {m: {"full": metrics(m)["test_r2_linear_fit"],
                               "reduced": metrics(f"{m}_reduced")["test_r2_linear_fit"]}
                           for m in ["xgboost", "classical", "hybrid"]}
    numbers["all_data"] = {m: {"paper_split": metrics(m)["test_r2_linear_fit"],
                               "all_data": metrics(f"{m}_all")["test_r2_linear_fit"],
                               "all_data_mae": metrics(f"{m}_all")["test_mae"],
                               "n_train": metrics(f"{m}_all")["n_train"]}
                           for m in ["xgboost", "classical", "hybrid"]}

    lc = json.loads((RES / "xgboost_learning_curve.json").read_text())
    numbers["learning_curve_n_test"] = lc["n_test"]
    numbers["learning_curve"] = {k: v["r2_linear_fit"] for k, v in lc["summary"].items()}

    imps = [json.loads((RES / f"{seed_tag('xgboost', s)}_importance.json").read_text()) for s in config.SEEDS]
    mean_imp = {k: float(np.mean([imp[k] for imp in imps])) for k in imps[0]}
    numbers["importance_top"] = sorted(mean_imp.items(), key=lambda kv: -kv[1])
    total = sum(mean_imp.values())
    by_type = {}
    for k, v in mean_imp.items():
        by_type[k.split("(")[0]] = by_type.get(k.split("(")[0], 0.0) + v / total
    numbers["importance_by_type"] = dict(sorted(by_type.items(), key=lambda kv: -kv[1]))

    numbers["single_runs"] = {t: metrics(t) for t in ["xgboost", "xgboost_tuned", "classical", "hybrid",
                                                       "hybrid_frozenq", "xgboost_reduced"]}
    return numbers


def main() -> None:
    OUT.mkdir(exist_ok=True)
    numbers = collect()
    for make in (fig_parity, fig_seeds, fig_ablations, fig_learning_curve, fig_importance):
        make(numbers)
    (OUT / "report_numbers.json").write_text(json.dumps(numbers, indent=1))
    print(f"Saved figures and report_numbers.json -> {OUT}")


if __name__ == "__main__":
    main()
