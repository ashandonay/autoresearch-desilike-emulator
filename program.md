# autoresearch — desilike emulator

This is an experiment to have the LLM do its own research on neural network emulators for cosmological computations (BAO/shapefit Fisher matrix covariances).

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a branch name**: e.g. `autoresearch/bao-base-unscaled`. The branch must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<name>` from current master.
3. **Read the in-scope files**:
   - `program.md` — this file, the experiment protocol.
   - `prepare.py` — fixed constants, data loading, standardization, evaluation. **Do not modify.**
4. **Set environment variables** for the data:
   - `EMULATOR_DATA_DIR`: path to the data directory containing `{TRACER}_train.npz` and `{TRACER}_test.npz`
   - `EMULATOR_TRACER`: tracer name (default: "LRG2")
5. **Create `train.py`** following the template below.
6. **Initialize `experiments.md`** with the first experiment reasoning.
7. **Confirm and go**: Confirm setup looks good, then kick off experimentation.

## Architecture

The model is a residual MLP (ResNet regressor):
- `proj_in`: Linear(in_dim, hidden_dim) + SiLU activation
- `N_HIDDEN` residual blocks, each: Linear(dim, dim*expand) → SiLU → Dropout → Linear(dim*expand, dim) + skip connection
- `proj_out`: Linear(hidden_dim, out_dim)

Known good configs:
- **base cosmo model**: 24dim, 6blocks, expand=4 (~28K params)
- **base_omegak_w_wa cosmo model**: 48dim, 6blocks, expand=4 (~113K params)

## train.py template

`train.py` is the only file you create/modify. It must:

1. **Import from prepare.py**: `TOTAL_EPOCHS`, `IN_DIM`, `OUT_DIM`, `N_TRAIN`, `evaluate_test_mse`, `make_dataloader`, `x_train`, `y_train`
2. **Use epoch-based LR scheduling**: `progress = epoch / TOTAL_EPOCHS` (10,000 epochs total). The schedule shape should match the full production training run.
3. **Cosine warm restarts with decaying peaks**: `N_RESTARTS` cycles, `LR_GAMMA` decay factor per cycle (< 1.0 to prevent overfitting at later cycles).
4. **Evaluate every `EVAL_EVERY` epochs** (default: 1000) using `evaluate_test_mse(model)`.
5. **Early stopping**: Stop after `PATIENCE` consecutive evals with no improvement (default: 3).
6. **Log results to `results.tsv`**: Append one row per eval checkpoint with these tab-separated columns:
   ```
   experiment	epoch	test_mse	best_mse	hidden_dim	n_hidden	expand	lr	lr_gamma	n_restarts	batch_size	description
   ```
   The file should be created with a header if it doesn't exist, and appended to otherwise. Flush after each write.
7. **Best model checkpointing**: Keep `best_state = copy.deepcopy(model.state_dict())` and restore it at the end.

Key hyperparameters to expose at the top of the file:
```
HIDDEN_DIM, N_HIDDEN, DROPOUT, EXPAND, BATCH_SIZE, LR, WEIGHT_DECAY,
WARMUP_FRAC, FINAL_LR_FRAC, N_RESTARTS, LR_GAMMA, GRAD_CLIP, SEED,
EVAL_EVERY, PATIENCE, EXPERIMENT, DESCRIPTION
```

## What you CAN do

- Modify `train.py` — model architecture, optimizer, hyperparameters, training loop, batch size, model size, etc.

## What you CANNOT do

- Modify `prepare.py`. It is read-only. It contains the fixed evaluation, data loading, and training constants.
- Install new packages or add dependencies. Use only what's in `pyproject.toml`.
- Modify the evaluation harness. `evaluate_test_mse` in `prepare.py` is the ground truth metric.

## The goal

**Get the lowest test_mse.** This is the MSE on the standardized test set, computed by `evaluate_test_mse()`.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Removing something and getting equal or better results is a win.

## Logging results

### experiments.md

Before each experiment, append a brief entry explaining *why* you're trying this change and what you expect:

```
## Experiment 3: 32dim 8blocks (was 24dim 6blocks)
Increasing capacity to handle higher dynamic range in scaled data.
Expect lower test_mse if the model was capacity-limited.
**Result**: test_mse=0.000080 — confirmed, 20% improvement.
```

### results.tsv

`results.tsv` accumulates across experiments on the branch. Each eval checkpoint is one row, so you can plot loss curves across experiments by grouping on the `experiment` column. Never overwrite — always append.

## The experiment loop

LOOP FOREVER:

1. Look at current results: what's the best test_mse so far? What has/hasn't worked?
2. Log reasoning to `experiments.md`: what you're trying and why.
3. Edit `train.py` with the experimental change.
4. `git commit` the change.
5. Run: `uv run train.py > run.log 2>&1`
6. Check results: `tail -20 run.log` for the summary block.
7. If test_mse improved, keep the commit. If worse, `git reset --hard HEAD~1`.
8. Repeat.

**Run experiments in parallel** on both GPUs (`CUDA_VISIBLE_DEVICES=0/1`) when possible.

**Crashes**: If a run crashes with a fixable bug (typo, missing import), fix and re-run. If fundamentally broken, skip it and move on.

**NEVER STOP**: Once the loop begins, do NOT pause to ask the human if you should continue. You are autonomous. If you run out of ideas, think harder — try combining previous near-misses, try more radical changes. The loop runs until the human interrupts you.

## Environment Variables

- `EMULATOR_DATA_DIR`: path to data directory (default: `~/scratch/bedcosmo/num_tracers/emulator/training_data/bao/base_omegak_w_wa/covar/v3`)
- `EMULATOR_TRACER`: tracer name, e.g. "LRG2", "BGS" (default: "LRG2")
