"""Central configuration for the HyQCGNN project.

Every number that comes from the paper (arXiv:2405.05205) or that a user may
want to tweak lives here, so the other files contain no magic constants.
"""
from pathlib import Path

# ----------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RAW_PATH = DATA_DIR / "oxide_perovskites_raw.pkl"   # output of fetch_data.py
GRAPH_PATH = DATA_DIR / "graphs.pt"                 # output of featurize.py

# --------------------------------------------------------------- dataset
MATBENCH_NAME = "matbench_perovskites"  # Castelli perovskites via MatBench
TARGET_COLUMN = "e_form"                # formation energy (eV/unit cell)
SEED = 42                      # default seed: data split, model init, optimiser
SEEDS = [42, 1, 2, 3, 4]       # seeds for repeated runs (summarize_seeds.py)

# Paper: 196 train / 25 validation / 25 test samples (246 in total)
N_TOTAL = 246
N_TRAIN = 196
N_VAL = 25
N_TEST = 25

# ----------------------------------------------------------- graph layout
N_ATOMS = 5      # ABO3 primitive cell = 5 atoms = 5 graph nodes
N_QUBITS = 5     # one qubit per node
AMP_DIM = 2 ** N_QUBITS  # 32 amplitudes; the 5x5=25 adjacency entries are zero-padded

NODE_FEATURES = [
    "atomic_number",
    "ewald_site_energy",
    "electronegativity",
    "electron_affinity",
    "ionization_potential",
    "cationic_radius",
    "anionic_radius",
]
EDGE_FEATURES = [
    "inverse_distance",
    "coulomb_potential",
    "electron_affinity_diff",
    "electronegativity_diff",
]

# Feature subsets. featurize.py always stores ALL features above; the subset
# is picked at load time (dataset.load_splits), so switching needs no re-featurizing.
# "reduced" = the only feature types in the paper's XGBoost importance plot (Fig. 4),
# which suggests the paper's final models used just these.
FEATURE_SETS = {
    "full": (NODE_FEATURES, EDGE_FEATURES),
    "reduced": (["ionization_potential", "ewald_site_energy"], ["coulomb_potential"]),
}
DEFAULT_FEATURE_SET = "full"

# ------------------------------------------------------------ model sizes
GNN_HIDDEN = 4          # width of the GENConv embedding before reduction

# ------------------------------------------------- gradient-free training
N_ITERATIONS = 2000     # paper: 2000 Nevergrad (NGOpt) iterations
LOG_EVERY = 50
