---
title: HyQCGNN Perovskite Predictor
emoji: 💎
colorFrom: indigo
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
short_description: Formation energy of ABO3 perovskites, quantum GNN
---

# HyQCGNN Perovskite Predictor

Type an ABO3 oxide formula (A-site cation first, e.g. `SrTiO3`) to get its DFT formation energy
predicted by three models trained on 2,652 Castelli perovskites: XGBoost (test R² 0.96), a
classical GENConv GNN and a hybrid quantum GNN (5-qubit circuit). The page also shows the
Goldschmidt / Bartel stability descriptors, an interactive 3D crystal, SHAP explanations and the
quantum layer's read-out. Batch screening, a chemical-space explorer and a REST API (`/docs`)
are included.

Re-implementation of Vitz et al., *Hybrid Quantum Graph Neural Network for Molecular Property
Prediction*, [arXiv:2405.05205](https://arxiv.org/abs/2405.05205).
