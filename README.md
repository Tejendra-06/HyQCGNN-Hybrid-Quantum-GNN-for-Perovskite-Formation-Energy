# HyQCGNN - Hybrid Quantum Graph Neural Network for formation-energy prediction

Re-implementation of *"Hybrid Quantum Graph Neural Network for Molecular Property
Prediction"* (Vitz et al., arXiv:2405.05205). It predicts the DFT formation energy of
ABO3 oxide perovskites with (a) a classical GENConv GNN, (b) a hybrid quantum-classical
GNN, and (c) an XGBoost baseline. The GNNs are trained **without gradients** using
Nevergrad's NGOpt.

## Setup
```bash
pip install -r requirements.txt
```

## Run order
```bash
python fetch_data.py                       # 1. download + filter ABO3 oxides
python featurize.py                        # 2. structures -> graphs
python train_gradient_free.py --model classical
python train_gradient_free.py --model hybrid      # add --iterations 200 for a quick test
python baseline_xgboost.py
python evaluate.py                         # parity plots
pytest tests/ -q                           # quantum layer sanity check
```

Add `--features reduced` to `train_gradient_free.py` / `baseline_xgboost.py` to use
only ionization energy, Ewald site energy and Coulomb potential - the only feature
types in the paper's Fig. 4. Results are saved with a `_reduced` suffix.
`evaluate.py` plots true energy on x and predicted on y (the paper's Fig. 3 layout;
its x-axis label "Predicted" appears to be swapped).

### Extra experiments
```bash
python train_gradient_free.py --model hybrid --seed 1   # repeat for every seed in config.SEEDS
python tune_xgboost.py --seed 1                        # tuned XGBoost per seed
python summarize_seeds.py                               # mean +/- std -> results/seed_summary.json
python xgboost_learning_curve.py                        # XGBoost on growing subsets of all ABO3 oxides
python make_report_figures.py                           # report Figures 8-12 + report_figures/report_numbers.json
```
Other `train_gradient_free.py` options: `--iterations N`, `--no-target-scaling`,
`--hidden N`, `--optimizer {NGOpt,CMA,TwoPointsDE,PSO,OnePlusOne}`, `--freeze-quantum`
(hybrid ablation) and `--dataset all` (every ABO3 oxide, 80/10/10 split; needs
`fetch_data.py --all` and `featurize.py --all`). Each option adds a suffix to the result
file names. Hybrid runs also report the test R^2 under shot noise (`config.EVAL_SHOTS`).

The final report is `HyQCGNN_Final_Project_Report.docx`, with figures in `report_figures/`.

## Web app: predict any ABO3 oxide
```bash
python -m uvicorn webapp.server:app --port 8765     # or double-click run_app.bat on Windows
```
Open http://localhost:8765 and type a formula, e.g. `SrTiO3`. **The first element is the A site
(12-fold cage) and the second is the B site (octahedral)**; `TiSrO3` is a different compound.
It needs the `--dataset all` artefacts: `fetch_data.py --all`, `featurize.py --all` and the two
`train_gradient_free.py --model {classical,hybrid} --dataset all` runs. The XGBoost model is
trained on first start and cached as `results/xgboost_all_model.json`.

For each formula the app shows:
- formation energy from XGBoost (the headline value, test R^2 0.96), the classical GNN and the hybrid
  quantum GNN, each with its test-set MAE, plus the DFT value and its train/val/test split when known;
- Goldschmidt tolerance factor, octahedral factor and Bartel's tau, from Shannon radii;
- an interactive 3D unit cell, SHAP contributions for the XGBoost prediction, the hybrid model's
  5x5 encoded matrix and qubit <Z> read-out, and the most similar known compounds.

Other tabs: **Batch** (paste or upload up to 500 formulas, download CSV), **Explore** (A x B map of
every DFT energy, click to predict) and **Models** (scores and method). Pairs that are not in
the data set are predicted on an ideal cubic cell whose lattice constant is fitted from Shannon
radii, and are flagged as extrapolations, as are elements the models never saw.

REST API (interactive docs at `/docs`): `GET /api/predict/SrTiO3`, `POST /api/batch`
with `{"formulas": [...]}`, `GET /api/overview`. Command line: `python predictor.py SrTiO3 BaZrO3`.

### Deploying (Docker / Hugging Face Spaces)
The app needs about 500 MB of RAM, so 512 MB free tiers are too small; a free Hugging Face Docker
Space (16 GB) works. `deploy/stage_hf_space.py` copies only what the app needs (~7 MB of code,
data and trained parameters) into `build/hf-space/`, with the Space card from `deploy/space_README.md`.
```bash
python deploy/stage_hf_space.py
docker build -t hyqcgnn build/hf-space && docker run -p 7860:7860 hyqcgnn   # local test: http://localhost:7860
hf auth login                                                             # once, with a write token
hf repo create <user>/hyqcgnn --repo-type space --space-sdk docker
hf upload <user>/hyqcgnn build/hf-space . --repo-type space                # builds and serves the Space
```
The image installs `requirements-app.txt` (pinned, CPU-only PyTorch) and trains the XGBoost model
during the build. Re-run the stage and upload commands to publish changes.

## Files
| File | Role |
|---|---|
| `config.py` | all constants (paths, split sizes, feature names, 2000 iterations) |
| `fetch_data.py` | downloads MatBench perovskites, keeps ABO3 oxides, samples 246 |
| `featurize.py` | builds one PyG graph per crystal (5 nodes x 7 features, 20 edges x 4 features) |
| `dataset.py` | 196/25/25 split, train-only standardisation, tabular view for XGBoost |
| `models/classical_gnn.py` | GENConv layer + reduction to a 5x5 matrix; classical model |
| `models/quantum_layer.py` | amplitude encoding + ansatz + read-out circuit (Qiskit) |
| `models/hybrid_gnn.py` | GENConv -> quantum layer -> scaled output; flat-parameter helpers |
| `train_gradient_free.py` | NGOpt training loop, validation model selection, test evaluation |
| `baseline_xgboost.py` | XGBoost baseline and feature importances |
| `evaluate.py` | true-vs-predicted plots with R^2 |
| `tune_xgboost.py` | randomised hyper-parameter search for XGBoost (5-fold CV on train) |
| `summarize_seeds.py` | mean +/- std of test metrics across seeds |
| `xgboost_learning_curve.py` | XGBoost R^2 vs training-set size on all ABO3 oxides |
| `make_report_figures.py` | result figures and numbers used in the report |
| `tests/test_quantum_layer.py` | fast quantum layer == full Qiskit circuit |
| `predictor.py` | prediction engine: formula -> structure -> graph -> all three models |
| `webapp/server.py` | FastAPI server and REST API |
| `webapp/static/` | web interface (HTML / CSS / JS) |
| `Dockerfile`, `requirements-app.txt` | web app container (pinned app-only dependencies) |
| `deploy/` | stages the Hugging Face Space (`stage_hf_space.py`) and its card |
| `tests/test_predictor.py` | formula parsing; app models reproduce the reported test scores |

## Notes on interpretation
The paper leaves a few details open; choices made here: node order is [A, B, O, O, O];
GENConv's output is reduced to one scalar per node (diagonal) and per edge (off-diagonal)
to form the 5x5 matrix; the 25 entries are zero-padded to 32 amplitudes; the 5 qubit
<Z> values are combined with trainable weights and a bias. The data subset (246
structures) is a seeded random sample, so exact R^2 values will differ from the paper.
