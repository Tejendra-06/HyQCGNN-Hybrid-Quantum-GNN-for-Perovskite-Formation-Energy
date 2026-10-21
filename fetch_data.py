"""Step 1 - download the perovskite dataset and keep only ABO3 oxides.

Run:  python fetch_data.py          (the 246-structure sample used everywhere)
      python fetch_data.py --all    (also save every ABO3 oxide, for subset_study.py
                                     and `--dataset all` training)

Source: the Castelli perovskites data set as packaged in MatBench
(`matbench_perovskites`, 18,928 five-atom cubic perovskites with DFT formation
energies). We keep oxides with formula ABO3, draw a reproducible random sample
of N_TOTAL structures (paper: 246) and store them in data/.
"""
import argparse

import numpy as np
from pymatgen.core import Structure

import config


def is_abo3_oxide(structure: Structure) -> bool:
    """True if the 5-atom cell is A + B + 3 O (three distinct elements)."""
    amounts = structure.composition.get_el_amt_dict()
    if len(amounts) != 3 or "O" not in amounts:
        return False
    return amounts["O"] == 3 and all(
        amt == 1 for el, amt in amounts.items() if el != "O"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", help="also save every ABO3 oxide")
    args = parser.parse_args()
    from matminer.datasets import load_dataset  # imported lazily: slow import

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading '{config.MATBENCH_NAME}' (cached by matminer after first run)...")
    df = load_dataset(config.MATBENCH_NAME)  # columns: structure, e_form
    print(f"  full data set: {len(df)} structures")

    mask = df["structure"].apply(is_abo3_oxide)
    df = df[mask].reset_index(drop=True)
    print(f"  ABO3 oxides:   {len(df)} structures")

    if len(df) < config.N_TOTAL:
        raise RuntimeError(
            f"Only {len(df)} ABO3 oxides found, need {config.N_TOTAL}. Lower N_TOTAL in config.py."
        )

    if args.all:
        df.to_pickle(config.ALL_RAW_PATH)
        print(f"Saved all {len(df)} ABO3 oxides -> {config.ALL_RAW_PATH}")

    rng = np.random.default_rng(config.SEED)
    keep = rng.choice(len(df), size=config.N_TOTAL, replace=False)
    df = df.iloc[np.sort(keep)].reset_index(drop=True)

    df.to_pickle(config.RAW_PATH)
    print(f"Saved {len(df)} structures -> {config.RAW_PATH}")
    print(df[config.TARGET_COLUMN].describe().to_string())


if __name__ == "__main__":
    main()
