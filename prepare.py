"""
Fixed data preparation for cosmological emulator autoresearch.
Loads training data, standardizes it, and provides evaluation utilities.

This file is READ-ONLY for the AI agent. Do not modify.

Usage:
    python prepare.py          # verify data exists and print stats
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 900   # wall-clock budget for autoresearch evaluation (15 minutes)
TOTAL_EPOCHS = 10000  # total epochs for the full training run (matches train_nn.py default)

TRACER = os.environ.get("EMULATOR_TRACER", "LRG2")            # set to "" for no tracer prefix

_DEFAULT_DATA_DIR = os.path.expanduser("~/scratch/bedcosmo/num_tracers/emulator/training_data/bao/base_omegak_w_wa/covar/v3")
DATA_DIR = os.environ.get("EMULATOR_DATA_DIR", _DEFAULT_DATA_DIR)

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _drop_nonfinite_rows(x, y, split_name=""):
    """Drop rows with any NaN/Inf in x or y. Returns filtered arrays."""
    bad_x = (~np.isfinite(x)).any(axis=1)
    bad_y = (~np.isfinite(y)).any(axis=1)
    bad = bad_x | bad_y
    n_bad = bad.sum()
    if n_bad > 0:
        keep = ~bad
        x, y = x[keep], y[keep]
        print(f"Dropped {n_bad} {split_name} rows with non-finite values.")
    return x, y


def load_data():
    """Load train/test .npz files from DATA_DIR. Returns numpy arrays."""
    prefix = f"{TRACER}_" if TRACER else ""
    train = np.load(os.path.join(DATA_DIR, f"{prefix}train.npz"), allow_pickle=True)
    test = np.load(os.path.join(DATA_DIR, f"{prefix}test.npz"), allow_pickle=True)

    x_train = train["x"].astype(np.float32)
    y_train = train["y"].astype(np.float32)
    x_test = test["x"].astype(np.float32)
    y_test = test["y"].astype(np.float32)

    x_train, y_train = _drop_nonfinite_rows(x_train, y_train, "train")
    x_test, y_test = _drop_nonfinite_rows(x_test, y_test, "test")

    if len(x_train) == 0 or len(x_test) == 0:
        raise ValueError("After dropping non-finite rows, train or test set is empty.")

    return x_train, y_train, x_test, y_test


def standardize(x_train, y_train, x_test, y_test):
    """Z-score normalization with min-sigma guard. Stats computed from training set only."""
    x_mu = x_train.mean(axis=0, keepdims=True)
    x_sigma = np.maximum(
        x_train.std(axis=0, keepdims=True),
        1e-6 * (x_train.max(axis=0, keepdims=True) - x_train.min(axis=0, keepdims=True)),
    ) + 1e-8

    y_mu = y_train.mean(axis=0, keepdims=True)
    y_sigma = np.maximum(
        y_train.std(axis=0, keepdims=True),
        1e-6 * (y_train.max(axis=0, keepdims=True) - y_train.min(axis=0, keepdims=True)),
    ) + 1e-8

    x_train_n = (x_train - x_mu) / x_sigma
    x_test_n = (x_test - x_mu) / x_sigma
    y_train_n = (y_train - y_mu) / y_sigma
    y_test_n = (y_test - y_mu) / y_sigma

    for name, arr in [("x_train", x_train_n), ("y_train", y_train_n),
                      ("x_test", x_test_n), ("y_test", y_test_n)]:
        if not np.all(np.isfinite(arr)):
            raise ValueError(
                f"Standardization produced non-finite values in {name}. "
                "Check for NaN/Inf in data or extreme outliers."
            )

    return x_train_n, y_train_n, x_test_n, y_test_n


# ---------------------------------------------------------------------------
# Module-level loading: data is loaded and moved to GPU on import
# ---------------------------------------------------------------------------

_raw = load_data()
_x_train_n, _y_train_n, _x_test_n, _y_test_n = standardize(*_raw)

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
x_train = torch.from_numpy(_x_train_n).to(_device)
y_train = torch.from_numpy(_y_train_n).to(_device)
x_test = torch.from_numpy(_x_test_n).to(_device)
y_test = torch.from_numpy(_y_test_n).to(_device)

IN_DIM = x_train.shape[1]
OUT_DIM = y_train.shape[1]

N_TRAIN = x_train.shape[0]

print(f"Data loaded: x_train={x_train.shape}, y_train={y_train.shape}, "
      f"x_test={x_test.shape}, y_test={y_test.shape}, device={_device}")

# ---------------------------------------------------------------------------
# Utilities (imported by train.py)
# ---------------------------------------------------------------------------

def make_dataloader(x, y, batch_size, shuffle=True):
    """Thin wrapper around TensorDataset + DataLoader."""
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate_test_mse(model):
    """
    Ground-truth metric: MSE on standardized test set.
    Returns float. Lower is better.
    DO NOT MODIFY — this is the fixed evaluation metric.
    """
    model.eval()
    preds = model(x_test)
    mse = torch.nn.functional.mse_loss(preds, y_test).item()
    return mse


# ---------------------------------------------------------------------------
# Main: verify data exists and print stats
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"\nData directory: {DATA_DIR}, Tracer: {TRACER or '(none)'}")
    print(f"Train: {x_train.shape[0]} samples, {x_train.shape[1]} inputs -> {y_train.shape[1]} outputs")
    print(f"Test:  {x_test.shape[0]} samples, {x_test.shape[1]} inputs -> {y_test.shape[1]} outputs")
    print(f"Device: {_device}")
    print(f"\nx_train stats: mean={x_train.mean().item():.4f}, std={x_train.std().item():.4f}")
    print(f"y_train stats: mean={y_train.mean().item():.4f}, std={y_train.std().item():.4f}")
    print(f"x_test  stats: mean={x_test.mean().item():.4f}, std={x_test.std().item():.4f}")
    print(f"y_test  stats: mean={y_test.mean().item():.4f}, std={y_test.std().item():.4f}")
    print("\nDone! Data ready for training.")
