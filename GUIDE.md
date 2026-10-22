# How to run the project

A step-by-step guide to set up HyQCGNN, train the models, use the web app and deploy it.
All commands are run from the `quantum-project` folder. Windows commands are shown first;
macOS / Linux equivalents are noted where they differ.

---

## 1. Quick start: just run the web app

If the project is already set up (the `.venv`, `data/` and `results/` folders exist):

**Windows:** double-click **`run_app.bat`**. It starts the server and opens the browser.

**Or from a terminal:**
```bash
.venv\Scripts\activate                              # macOS / Linux: source .venv/bin/activate
python -m uvicorn webapp.server:app --port 8765
```
Then open **http://localhost:8765**. The first start takes 20–30 seconds while the data and models
load, and on the very first run the XGBoost model is also trained once and saved as
`results/xgboost_all_model.json`. Stop the server with `Ctrl+C`.

If anything is missing, follow the full setup below.

---

## 2. Requirements

| Need | Notes |
|---|---|
| Python 3.11 or newer | tested with Python 3.13 |
| About 3 GB of disk | mostly PyTorch, Qiskit and pymatgen |
| About 1 GB of free RAM | the running app uses ~500 MB |
| Internet (first time only) | to install packages and download the data set |

---

## 3. One-time setup

### 3.1 Create a virtual environment and install packages
```bash
python -m venv .venv
.venv\Scripts\activate                              # macOS / Linux: source .venv/bin/activate
pip install -r requirements.txt
```

### 3.2 Download and prepare the data
```bash
python fetch_data.py --all       # downloads the Castelli perovskites (MatBench), keeps ABO3 oxides
python featurize.py              # graphs for the 246-structure paper sample  -> data/graphs.pt
python featurize.py --all        # graphs for all 2,652 ABO3 oxides           -> data/graphs_all.pt
```
`fetch_data.py --all` saves both the full set and the 246-structure sample. The download is cached
by matminer, so later runs are fast.

### 3.3 Train the models the web app uses
```bash
python train_gradient_free.py --model classical --dataset all     # ~1 minute
python train_gradient_free.py --model hybrid --dataset all        # ~1-2 minutes
python baseline_xgboost.py --dataset all
```
These write `results/*_all_best_params.npy` and `results/*_all_metrics.json`, which the app loads.
Expected test scores: XGBoost R² ≈ 0.96, classical GNN ≈ 0.57, hybrid quantum GNN ≈ 0.55.

### 3.4 Check that everything works
```bash
pytest tests/ -q
```
All 19 tests should pass. They include a check that the app's models reproduce the scores in
`results/*_all_metrics.json`.

---

## 4. Using the web app

Start the server (section 1) and open http://localhost:8765.

### Predict tab
Type a formula and press **Predict**, or click one of the example chips.

- **The first element is the A site, the second is the B site.** `SrTiO3` means Sr on the A site
  (12-fold cage) and Ti on the B site (octahedral). `TiSrO3` is a different compound with a
  different energy.
- Element symbols are case-sensitive: `SrTiO3`, not `srtio3`.
- Multiples are accepted (`Sr2Ti2O6` = `SrTiO3`). Other stoichiometries are rejected.

What you get:

| Section | Meaning |
|---|---|
| Headline value | XGBoost formation energy (eV per 5-atom cell), ± its typical test error |
| Model comparison | the same prediction from all three models, with the DFT value as a dotted line |
| Crystal structure | interactive 3D cell: drag to rotate, switch 1×1×1 / 2×2×2 / 3×3×3, toggle octahedra |
| Structural stability | Goldschmidt tolerance factor, Bartel τ, oxidation states, radii, bond lengths |
| Why this prediction? | SHAP: which descriptors pushed the XGBoost value down or up |
| Inside the quantum layer | the 5×5 matrix encoded into the circuit and the 5 qubit ⟨Z⟩ read-outs |
| Distribution / similar compounds | where the value sits among known perovskites; click a neighbour to predict it |

How much to trust a result. The badges and notices tell you:
- **DFT structure + test/validation split:** a genuine prediction on an unseen compound.
- **Training split:** the models saw this compound while training, so the error shown is optimistic.
- **Extrapolation:** the A/B pair is not in the data set, so the app uses an idealised cubic cell
  built from ionic radii. Treat the value as approximate.
- **"Never appeared in training":** the element is outside the 52 the models know (e.g. Nd, Ce, U).
  The value is qualitative only.

### Batch tab
Paste formulas (one per line or comma-separated) or load a `.csv` / `.txt` file, up to 500 at once.
Click column headers to sort, click a formula to open it, and use **Download CSV** to save the table.

### Explore tab
A map of the DFT formation energy for every A–B pair (A on the rows, B on the columns). Cooler
colours are more stable. Click a cell to predict that compound.

### Models tab
Test-set scores of the three models and a summary of the method.

The ☀/☾ button in the top bar switches between dark and light themes.

---

## 5. Other ways to use the predictor

**Command line**
```bash
python predictor.py SrTiO3 BaZrO3 LaAlO3
```

**REST API** (while the server runs; interactive docs at http://localhost:8765/docs)
```bash
curl http://localhost:8765/api/predict/SrTiO3
curl -X POST http://localhost:8765/api/batch -H "Content-Type: application/json" -d "{\"formulas\": [\"SrTiO3\", \"BaZrO3\"]}"
```

| Endpoint | Returns |
|---|---|
| `GET /api/predict/{formula}` | full prediction for one compound |
| `POST /api/predict` | same, body `{"formula": "SrTiO3"}` |
| `POST /api/batch` | one summary row per formula, body `{"formulas": [...]}` |
| `GET /api/overview` | model scores, element list and data for the explorer map |

---

## 6. Reproducing the research results

These are the experiments behind the report (`HyQCGNN_Final_Project_Report.docx`):
```bash
python train_gradient_free.py --model classical          # paper-sized 196/25/25 split
python train_gradient_free.py --model hybrid
python baseline_xgboost.py
python evaluate.py                                       # parity plots

python train_gradient_free.py --model hybrid --seed 1    # repeat for seeds 1-4 (config.SEEDS)
python tune_xgboost.py --seed 1
python summarize_seeds.py                                # mean ± std -> results/seed_summary.json
python xgboost_learning_curve.py
python make_report_figures.py                            # report figures -> report_figures/
```
Useful training options: `--iterations 200` (quick test), `--features reduced`,
`--optimizer CMA`, `--freeze-quantum`, `--hidden N`, `--no-target-scaling`. See `README.md`
for the full list.

---

## 7. Deploying online (Hugging Face Spaces)

The app needs ~500 MB of RAM, so 512 MB free tiers (e.g. Render) are too small. A free
Hugging Face Docker Space (16 GB RAM) works well.

```bash
python deploy/stage_hf_space.py                 # copies only what the app needs into build/hf-space (~7 MB)

# optional local test (Docker Desktop must be running; needs ~4 GB free RAM while building)
docker build -t hyqcgnn build/hf-space
docker run -p 7860:7860 hyqcgnn                 # http://localhost:7860

# publish
hf auth login                                   # once; paste a "write" token from huggingface.co/settings/tokens
hf repo create <your-username>/hyqcgnn --repo-type space --space-sdk docker
hf upload <your-username>/hyqcgnn build/hf-space . --repo-type space
```
Hugging Face then builds the image (5–10 minutes) and serves it at
`https://<your-username>-hyqcgnn.hf.space`. To publish changes later, run the stage and upload
commands again.

---

## 8. Troubleshooting

| Problem | Fix |
|---|---|
| `Run python fetch_data.py --all then python featurize.py --all first` | the data files are missing: run section 3.2 |
| `Missing classical_all_best_params.npy` (or hybrid) | run the training commands in section 3.3 |
| `error while attempting to bind on address ... 8000` / port in use | another program uses that port: use `--port 8765` (the default here) or any free port |
| Page shows "Server unavailable" | the server is not running or still loading: wait ~30 s, then refresh |
| Page looks unchanged after editing the UI | hard-refresh the browser with `Ctrl+F5` |
| 3D views are blank | the browser has WebGL turned off; the rest of the page still works |
| `'srtio3' is not a chemical formula` | element symbols are case-sensitive: write `SrTiO3` |
| Background jobs or Docker builds get killed | the PC is low on memory: close Docker Desktop and other heavy apps, then retry |
| `ModuleNotFoundError` | the virtual environment is not active: run `.venv\Scripts\activate` first |
