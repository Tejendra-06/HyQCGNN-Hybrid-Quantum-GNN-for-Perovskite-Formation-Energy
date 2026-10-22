"""Copy exactly what the web app needs into build/hf-space/ (the Docker build context / Space repo).

Run:  python deploy/stage_hf_space.py
Then: docker build -t hyqcgnn build/hf-space                       (test locally)
      hf upload <user>/<space> build/hf-space . --repo-type space   (publish)

About 7 MB of data and model files are included; nothing else from the project is.
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "build" / "hf-space"

FILES = [
    "Dockerfile", "requirements-app.txt",
    "config.py", "dataset.py", "featurize.py", "predictor.py",
    "models/__init__.py", "models/classical_gnn.py", "models/hybrid_gnn.py", "models/quantum_layer.py",
    "webapp/server.py", "webapp/static/index.html", "webapp/static/app.css",
    "webapp/static/app.js", "webapp/static/scene.js",
    "data/graphs_all.pt", "data/oxide_perovskites_all.pkl",
    "results/classical_all_best_params.npy", "results/hybrid_all_best_params.npy",
    "results/xgboost_all_metrics.json", "results/classical_all_metrics.json", "results/hybrid_all_metrics.json",
]


def main() -> None:
    missing = [f for f in FILES if not (ROOT / f).exists()]
    if missing:
        raise SystemExit("Missing (see README 'Web app' for how to create them):\n  " + "\n  ".join(missing))
    if OUT.exists():
        shutil.rmtree(OUT)
    for f in FILES:
        (OUT / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / f, OUT / f)
    shutil.copy2(ROOT / "deploy" / "space_README.md", OUT / "README.md")
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file()) / 1e6
    print(f"Staged {len(FILES) + 1} files ({size:.1f} MB) -> {OUT}")


if __name__ == "__main__":
    main()
