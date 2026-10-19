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
| `tests/test_quantum_layer.py` | fast quantum layer == full Qiskit circuit |

## Notes on interpretation
The paper leaves a few details open; choices made here: node order is [A, B, O, O, O];
GENConv's output is reduced to one scalar per node (diagonal) and per edge (off-diagonal)
to form the 5x5 matrix; the 25 entries are zero-padded to 32 amplitudes; the 5 qubit
<Z> values are combined with trainable weights and a bias. The data subset (246
structures) is a seeded random sample, so exact R^2 values will differ from the paper.
