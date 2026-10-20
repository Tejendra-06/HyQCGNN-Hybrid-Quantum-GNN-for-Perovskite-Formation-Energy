"""Step 3 - gradient-free training with Nevergrad's NGOpt (paper Sections II.B.3, II.D, III).

Run:
    python train_gradient_free.py --model classical
    python train_gradient_free.py --model hybrid
    (optional: --iterations 200 for a quick smoke test,
               --features reduced to use only the paper's Fig. 4 feature types,
               --seed N for another data split / initialisation / optimiser run,
               --no-target-scaling to train on raw energies)

The target is standardised with TRAIN mean/std by default: the model learns
(y - mu) / sd and its outputs are mapped back with * sd + mu, so all reported
losses and metrics are in the original energy units.

Loop (paper workflow, 2000 iterations):
  1. ask NGOpt for a candidate parameter vector
  2. load it into the model
  3. compute the training MSE (whole training set as one batch) and tell NGOpt
  4. evaluate the validation MSE and remember the best-so-far vector
After all iterations the best-on-validation parameters are evaluated ONCE on the
held-out test set. Outputs go to results/: metrics JSON, predictions, best
parameter vector and the loss curves.
"""
import argparse
import json
import time

import nevergrad as ng
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, r2_score

import config
from dataset import as_batch, load_splits
from models.classical_gnn import ClassicalGENConvModel
from models.hybrid_gnn import HybridQuantumGNN, get_flat_params, set_flat_params


def build_model(name: str, node_dim: int, edge_dim: int):
    if name == "classical":
        return ClassicalGENConvModel(node_dim, edge_dim)
    if name == "hybrid":
        return HybridQuantumGNN(node_dim, edge_dim)
    raise ValueError(name)


@torch.no_grad()
def predict(model, batch) -> np.ndarray:
    return model(batch).detach().cpu().numpy().astype(np.float64)


def mse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((pred - y) ** 2))


def fit_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """R^2 of a simple linear fit of true vs predicted (the paper's figure of merit)."""
    return float(np.corrcoef(y_pred, y_true)[0, 1] ** 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["classical", "hybrid"], required=True)
    parser.add_argument("--iterations", type=int, default=config.N_ITERATIONS)
    parser.add_argument("--features", choices=list(config.FEATURE_SETS), default=config.DEFAULT_FEATURE_SET)
    parser.add_argument("--seed", type=int, default=config.SEED)
    parser.add_argument("--no-target-scaling", dest="target_scaling", action="store_false")
    args = parser.parse_args()
    tag = args.model
    if args.features != "full":
        tag += f"_{args.features}"
    if not args.target_scaling:
        tag += "_noscale"
    if args.iterations != config.N_ITERATIONS:
        tag += f"_it{args.iterations}"
    if args.seed != config.SEED:
        tag += f"_s{args.seed}"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    splits = load_splits(args.features, args.seed)
    train_b, val_b, test_b = (as_batch(s) for s in (splits.train, splits.val, splits.test))
    y_train, y_val, y_test = (b.y.numpy().astype(np.float64) for b in (train_b, val_b, test_b))
    if args.target_scaling:
        y_mu, y_sd = y_train.mean(), y_train.std()
    else:
        y_mu, y_sd = 0.0, 1.0

    def predict_energy(batch) -> np.ndarray:
        return predict(model, batch) * y_sd + y_mu

    model = build_model(args.model, train_b.x.shape[1], train_b.edge_attr.shape[1])
    x0 = get_flat_params(model)
    print(f"model={args.model}  features={args.features}  trainable parameters={x0.size}  "
          f"iterations={args.iterations}")

    parametrization = ng.p.Array(init=x0)
    parametrization.random_state = np.random.RandomState(args.seed)
    optimizer = ng.optimizers.NGOpt(parametrization=parametrization, budget=args.iterations)

    best_val, best_vec = np.inf, x0.copy()
    train_curve, val_curve = [], []
    start = time.time()

    for it in range(1, args.iterations + 1):
        candidate = optimizer.ask()
        set_flat_params(model, candidate.value)
        train_loss = mse(predict_energy(train_b), y_train)
        optimizer.tell(candidate, train_loss)

        val_loss = mse(predict_energy(val_b), y_val)
        train_curve.append(train_loss)
        val_curve.append(val_loss)
        if val_loss < best_val:
            best_val, best_vec = val_loss, np.array(candidate.value, copy=True)

        if it % config.LOG_EVERY == 0 or it == 1:
            print(f"iter {it:5d}  train MSE {train_loss:.4f}  val MSE {val_loss:.4f}  "
                  f"best val {best_val:.4f}  ({time.time() - start:.0f}s)")

    # ---- final, one-shot evaluation on the untouched test set
    set_flat_params(model, best_vec)
    pred_test = predict_energy(test_b)
    metrics = {
        "model": args.model,
        "features": args.features,
        "seed": args.seed,
        "target_scaling": args.target_scaling,
        "iterations": args.iterations,
        "n_parameters": int(x0.size),
        "best_val_mse": best_val,
        "test_mse": mse(pred_test, y_test),
        "test_mae": float(mean_absolute_error(y_test, pred_test)),
        "test_r2_linear_fit": fit_r2(y_test, pred_test),
        "test_r2_score": float(r2_score(y_test, pred_test)),
        "seconds": time.time() - start,
    }
    print(json.dumps(metrics, indent=2))

    out = config.RESULTS_DIR
    (out / f"{tag}_metrics.json").write_text(json.dumps(metrics, indent=2))
    np.save(out / f"{tag}_best_params.npy", best_vec)
    np.savez(out / f"{tag}_predictions.npz", y_true=y_test, y_pred=pred_test)
    np.savez(out / f"{tag}_curves.npz", train=train_curve, val=val_curve)


if __name__ == "__main__":
    main()
