# Autoresearch desilike emulator

## Autoresearch Loop

When running the iterative hyperparameter/architecture optimization loop:

1. **Log reasoning for each experiment**: Before running an experiment, append a 1-2 sentence explanation of *why* you're trying this change to `experiments.md`. Include what you expect to happen and what result would confirm or reject your hypothesis. Example:
   ```
   ## Experiment 7: 48dim 6blocks (was 8blocks)
   Reducing depth from 8 to 6 blocks. Fewer blocks means faster per-step throughput,
   so more gradient updates in the 30-min budget. If the model has enough capacity at
   6 blocks, the extra steps should improve convergence.
   **Result**: test_mse=0.000101 (was 0.000166) — confirmed, 39% improvement.
   ```

2. **Track all results in `results.tsv`** (gitignored, branch-specific). Never overwrite this file when switching branches — back it up or commit it first.

3. **Commit improvements** to `train.py` with test_mse in the commit message.

4. **Run experiments in parallel** on both GPUs (CUDA_VISIBLE_DEVICES=0/1) when possible.

## Environment Variables

- `EMULATOR_ANALYSIS`: "bao" or "shapefit" (default: "bao")
- `EMULATOR_COSMO_MODEL`: e.g. "base", "base_w", "base_omegak_w_wa" (default: "base_omegak_w_wa")
- `EMULATOR_QUANTITY`: "mean" or "covar" (default: "covar")
- `EMULATOR_DATA_VERSION`: e.g. "3" (default: "3")
- `EMULATOR_TRACER`: e.g. "LRG2", "BGS" (default: "LRG2")
