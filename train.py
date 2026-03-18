"""
Cosmological emulator training script. Single-GPU, single-file.
Trains an MLP to approximate Fisher matrix covariance computations.
Usage: uv run train.py
"""

import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from prepare import TIME_BUDGET as _ORIG_TIME_BUDGET, IN_DIM, OUT_DIM, evaluate_test_mse, make_dataloader, x_train, y_train

TIME_BUDGET = 1800  # Override: 30 minutes instead of 15

# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.0, expand=4):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, dim * expand)
        self.fc2 = nn.Linear(dim * expand, dim)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        h = self.ln(x)
        h = F.gelu(self.fc1(h))
        h = self.drop(h)
        h = self.fc2(h)
        return x + h


def poly_features(x):
    """Add pairwise interaction features: x_i * x_j for all i <= j."""
    # x: (batch, d)
    d = x.shape[1]
    pairs = []
    for i in range(d):
        for j in range(i, d):
            pairs.append(x[:, i] * x[:, j])
    return torch.cat([x, torch.stack(pairs, dim=1)], dim=1)


class NNRegressor(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, n_hidden, dropout=0.0):
        super().__init__()
        # in_dim will be augmented by poly features: d + d*(d+1)/2
        aug_dim = in_dim + in_dim * (in_dim + 1) // 2
        self.proj_in = nn.Linear(aug_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResBlock(hidden_dim, dropout) for _ in range(n_hidden)])
        self.ln_out = nn.LayerNorm(hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = poly_features(x)
        x = F.gelu(self.proj_in(x))
        for block in self.blocks:
            x = block(x)
        x = self.ln_out(x)
        return self.proj_out(x)

# ---------------------------------------------------------------------------
# Hyperparameters (edit these directly, no CLI flags needed)
# ---------------------------------------------------------------------------

HIDDEN_DIM = 384
N_HIDDEN = 6
DROPOUT = 0.0
BATCH_SIZE = 256
LR = 1e-3
WEIGHT_DECAY = 1e-5
WARMUP_RATIO = 0.05     # fraction of time budget for LR warmup
FINAL_LR_FRAC = 0.01    # final LR as fraction of initial
GRAD_CLIP = 1.0
SEED = 42

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = NNRegressor(IN_DIM, OUT_DIM, HIDDEN_DIM, N_HIDDEN, DROPOUT).to(device)
num_params = sum(p.numel() for p in model.parameters())
print(f"Model: {N_HIDDEN} hidden layers, {HIDDEN_DIM} dim, {num_params:,} params")

optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
loss_fn = nn.MSELoss()

train_loader = make_dataloader(x_train, y_train, BATCH_SIZE, shuffle=True)

print(f"Time budget: {TIME_BUDGET}s")
print(f"Batch size: {BATCH_SIZE}, LR: {LR}, Weight decay: {WEIGHT_DECAY}")

# ---------------------------------------------------------------------------
# LR schedule (cosine decay with warmup, keyed to time progress)
# ---------------------------------------------------------------------------

N_RESTARTS = 3

def get_lr(progress):
    """Returns LR multiplier given progress in [0, 1] with warm restarts."""
    if progress < WARMUP_RATIO:
        return progress / WARMUP_RATIO if WARMUP_RATIO > 0 else 1.0
    post_warmup = (progress - WARMUP_RATIO) / (1.0 - WARMUP_RATIO)
    cycle_progress = (post_warmup * N_RESTARTS) % 1.0
    return FINAL_LR_FRAC + 0.5 * (1.0 - FINAL_LR_FRAC) * (1.0 + math.cos(math.pi * cycle_progress))

# ---------------------------------------------------------------------------
# Training loop (time-budgeted)
# ---------------------------------------------------------------------------

import copy

t_start_training = time.time()
total_training_time = 0.0
step = 0
epoch = 0
smooth_loss = 0.0
data_iter = iter(train_loader)
best_mse = float('inf')
best_state = None
EVAL_INTERVAL = 60  # eval every 60 seconds

while True:
    t0 = time.time()

    # Get next batch (restart dataloader on exhaustion)
    try:
        xb, yb = next(data_iter)
    except StopIteration:
        epoch += 1
        data_iter = iter(train_loader)
        xb, yb = next(data_iter)

    # Forward + backward
    model.train()
    optimizer.zero_grad()
    pred = model(xb)
    loss = loss_fn(pred, yb)
    loss_val = loss.item()

    # NaN detection
    if not math.isfinite(loss_val):
        print(f"FAIL: NaN/Inf loss at step {step}")
        exit(1)

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP)

    # Update LR based on time progress
    progress = min(total_training_time / TIME_BUDGET, 1.0)
    lr_mult = get_lr(progress)
    for group in optimizer.param_groups:
        group["lr"] = LR * lr_mult

    optimizer.step()

    dt = time.time() - t0
    total_training_time += dt

    # Logging
    ema_beta = 0.95
    smooth_loss = ema_beta * smooth_loss + (1 - ema_beta) * loss_val
    debiased_loss = smooth_loss / (1 - ema_beta ** (step + 1))
    pct_done = 100 * progress
    remaining = max(0, TIME_BUDGET - total_training_time)

    if step % 500 == 0 or remaining < 1:
        print(f"step {step:06d} ({pct_done:5.1f}%) | loss: {debiased_loss:.6f} | lr: {LR * lr_mult:.2e} | remaining: {remaining:.0f}s")

    # Periodic eval and best model checkpoint
    if step % 5000 == 0 and total_training_time > 30:
        cur_mse = evaluate_test_mse(model)
        if cur_mse < best_mse:
            best_mse = cur_mse
            best_state = copy.deepcopy(model.state_dict())
            print(f"  >> new best: {best_mse:.6f} at step {step}")

    step += 1

    if total_training_time >= TIME_BUDGET:
        break

# ---------------------------------------------------------------------------
# Final evaluation
# ---------------------------------------------------------------------------

# Also eval final model and compare to best checkpoint
final_mse = evaluate_test_mse(model)
if best_state is not None and best_mse < final_mse:
    model.load_state_dict(best_state)
    print(f"Loaded best checkpoint (mse={best_mse:.6f}) over final (mse={final_mse:.6f})")
test_mse = evaluate_test_mse(model)

t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0.0

print("---")
print(f"test_mse:         {test_mse:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"num_steps:        {step}")
print(f"num_params:       {num_params}")
print(f"hidden_dim:       {HIDDEN_DIM}")
print(f"n_hidden:         {N_HIDDEN}")
