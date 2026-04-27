# Phase 2: FNO Architecture and Training

## Context

I am building a research project: an FNO (Fourier Neural Operator) that takes an implied volatility surface as input and returns Heston stochastic volatility model parameters — amortized calibration in a single forward pass.

**Phases 0–1 are complete.** The repo `neural-operator-calibration/` contains:
- `data/generate_heston.py` — Heston IV surface generator with Latin Hypercube Sampling
- `data/dataset.py` — PyTorch Dataset with normalization, supports masking
- `models/fno_calibrator.py` — FNO with 4 Fourier layers, adaptive pooling, parameter head
- `models/mlp_baseline.py` — MLP baseline (flattened 320-dim input → 4-layer MLP → 5 params)
- `train.py`, `eval.py`, `configs/default.yaml`
- Data: `data/heston_train_100k.h5` (100k surfaces), `data/heston_val_10k.h5` (10k), `data/heston_test_10k.h5` (10k), `data/heston_ood_5k.h5` (5k OOD)
- Each surface is shape (16, 20) — 16 log-moneyness points × 20 time-to-expiry points
- Parameters: (κ, θ, ξ, ρ, v₀) — mean reversion, long-run variance, vol-of-vol, correlation, initial variance

**Phase 2 goal:** Serious training and architecture tuning. Train FNO and MLP on the full 100k dataset, achieve <10% relative error on in-distribution parameter recovery, run ablations, and produce the core comparison table.

## Task 1: Training infrastructure improvements

Update `train.py` with the following:

**Mixed precision training:**
```python
scaler = torch.amp.GradScaler('cuda')
with torch.amp.autocast('cuda', dtype=torch.float16):
    pred = model(surface)
    loss = criterion(pred, params)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

**Learning rate schedule:** Use OneCycleLR (more aggressive than CosineAnnealing, typically converges faster):
```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, epochs=200,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,  # 10% warmup
    anneal_strategy='cos'
)
# Step per batch, not per epoch
```

**Early stopping:** Stop if validation loss hasn't improved for 30 epochs. Save best checkpoint by validation loss.

**Per-parameter loss weighting (optional experiment):**
Some parameters are harder to recover than others. ρ and ξ affect the smile shape subtly; κ affects the term structure. Try a weighted MSE where harder parameters get higher weight:
```python
# Start with equal weights, then try:
weights = torch.tensor([1.0, 1.0, 2.0, 2.0, 1.0])  # extra weight on ξ, ρ
loss = (weights * (pred - target) ** 2).mean()
```
Log whether this helps vs. uniform weighting.

**Gradient clipping:**
```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**W&B logging — log everything that matters:**
- Training loss (per batch and per epoch)
- Validation loss (per epoch)
- Per-parameter RMSE in original units: κ_rmse, θ_rmse, ξ_rmse, ρ_rmse, v0_rmse
- Per-parameter relative error (median and 95th percentile)
- Learning rate
- Gradient norm
- Model parameter count
- Training time per epoch

## Task 2: Train FNO — main model

**Hyperparameters for the primary run:**
```yaml
model:
  type: fno
  n_modes: [12, 12]         # Fourier modes per dimension
  hidden_channels: 64        # Width of Fourier layers
  n_layers: 4                # Number of Fourier blocks
  pool_size: [4, 5]          # Adaptive pooling before parameter head
  head_dims: [256, 64]       # Parameter head MLP hidden dims

training:
  batch_size: 64
  epochs: 200
  lr: 1e-3
  weight_decay: 1e-4
  scheduler: onecycle
  grad_clip: 1.0
  early_stopping_patience: 30

data:
  train: data/heston_train_100k.h5
  val: data/heston_val_10k.h5
  test: data/heston_test_10k.h5
  ood: data/heston_ood_5k.h5
```

Run 3 seeds (42, 43, 44) and report mean ± std of all metrics. On a single A100, each run should take ~2–4 hours (100k samples, batch 64, 200 epochs).

## Task 3: Train MLP — Horvath-style baseline

Same training setup, same data, same number of seeds. The MLP should have **roughly the same parameter count** as the FNO — print both counts at the start and adjust MLP width if they differ by more than 2×.

```python
# Example adjustment:
fno_params = sum(p.numel() for p in fno_model.parameters())
# Set MLP hidden dim so total params ≈ fno_params
# Typical: MLP with dims [320, 512, 256, 128, 5] ≈ ~230k params
# FNO with 64 channels, 4 layers ≈ ~500k–1M params
# If FNO is much larger, increase MLP width or depth to match
```

## Task 4: Ablation studies

Run these ablations to understand what matters in the architecture. Each is a single-seed run (seed=42) with one change from the primary FNO config:

1. **Modes ablation:** n_modes = [4, 4], [8, 8], [12, 12], [16, 16] — how many Fourier modes are needed?
2. **Depth ablation:** n_layers = 2, 4, 6, 8 — how deep does the operator need to be?
3. **Width ablation:** hidden_channels = 32, 64, 128 — how wide?
4. **Pooling ablation:** pool_size = [1, 1] (global avg pool) vs [4, 5] vs [8, 10] — how much spatial info does the parameter head need?
5. **Loss ablation:** MSE vs Huber loss (δ=1.0) vs weighted MSE (2× weight on ξ, ρ)

Create `scripts/run_ablations.sh` that runs all ablations sequentially with descriptive W&B run names:
```bash
python train.py --config configs/ablation_modes4.yaml --wandb_name "ablation/modes_4x4"
python train.py --config configs/ablation_modes8.yaml --wandb_name "ablation/modes_8x8"
# etc.
```

Or better: create `configs/ablations.yaml` that lists all ablation configs and a script that iterates.

## Task 5: Comprehensive evaluation

Update `eval.py` to produce a full evaluation report. For each model (FNO, MLP, each ablation), on each test set (in-distribution, OOD):

**Metrics to compute:**

1. **Per-parameter RMSE** (in original units):
   ```
   κ RMSE: 0.23 (±0.02 across seeds)
   θ RMSE: 0.008
   ξ RMSE: 0.05
   ρ RMSE: 0.03
   v₀ RMSE: 0.006
   ```

2. **Per-parameter relative error** — |pred - true| / |true|:
   Report median and 95th percentile. The 95th percentile catches catastrophic failures.

3. **IV reconstruction RMSE** — the ultimate metric. For each test sample:
   - Take predicted parameters θ̂
   - Re-price the full IV surface using the Heston pricer with θ̂
   - Compare to the input surface
   - Report RMSE in implied vol points (multiply by 100 for percentage points)
   ```
   IV reconstruction RMSE: 0.35 vol points (= 0.0035 in σ units)
   ```
   This is more meaningful than parameter RMSE because different parameter errors have different IV impact.

4. **Per-parameter scatter plots** — for each parameter, plot predicted vs true on the test set. Save to `figures/eval/scatter_{param}.png`. Perfect recovery = diagonal line.

5. **Error stratification by parameter region** — split the test set into quintiles by each parameter value and report RMSE per quintile. This reveals if the model is worse at extremes.

6. **Summary comparison table:**
   ```
   | Model          | Params | κ RMSE | θ RMSE | ξ RMSE | ρ RMSE | v₀ RMSE | IV RMSE (vol pts) | Train time |
   |----------------|--------|--------|--------|--------|--------|---------|-------------------|------------|
   | FNO (primary)  | 520k   | 0.23   | 0.008  | 0.05   | 0.03   | 0.006   | 0.35              | 2.1h       |
   | MLP baseline   | 510k   | 0.28   | 0.010  | 0.07   | 0.04   | 0.007   | 0.52              | 1.8h       |
   ```

Save the full table as `results/comparison_table.csv` and as a LaTeX table in `results/comparison_table.tex`.

## Task 6: Diagnostic visualizations

Create `scripts/plot_diagnostics.py` that generates:

1. **Training curves** — loss vs epoch for FNO and MLP on the same plot
2. **Per-parameter learning dynamics** — RMSE vs epoch for each of the 5 parameters, showing which parameters converge first
3. **Worst-case analysis** — find the 10 test samples with highest IV reconstruction error, plot their true vs predicted surfaces side by side as heatmaps. This reveals failure modes (usually extreme ρ or high ξ).
4. **Ablation summary plot** — bar chart of IV RMSE for each ablation variant

Save all to `figures/diagnostics/`.

## What success looks like at the end of Phase 2

**Must-hit targets (if these fail, debug before proceeding):**
- FNO per-parameter relative error < 10% (median) on in-distribution test for all 5 parameters
- FNO IV reconstruction RMSE < 1.0 vol point on in-distribution test
- FNO matches or beats MLP on the fixed 16×20 grid (same resolution as training)

**Expected but not required:**
- ρ and ξ are the hardest parameters — relative error may be 5–15% while κ, θ, v₀ are < 5%
- OOD test error is 2–3× the in-distribution error (this is expected and fine)
- Modes ablation shows diminishing returns past 8×8 modes
- Depth ablation shows 4 layers is near-optimal (2 layers underfits, 8 layers doesn't help)

**Failure modes to watch for:**
- If training loss drops but validation loss plateaus early → overfitting. Increase weight decay or add dropout after the adaptive pooling layer.
- If ρ RMSE is 3× worse than other parameters → the model isn't learning the smile skew. Try adding ρ-weighted loss or check if your IV surfaces actually show visible skew variation across ρ values.
- If IV reconstruction RMSE is large (>2 vol points) even though parameter RMSE is small → the model is recovering parameters in a flat direction of the loss landscape (two different parameter sets produce similar surfaces). This is a fundamental identifiability issue with Heston — especially the κ-θ tradeoff. Consider reparameterizing: predict (κ, κθ, ξ, ρ, v₀) instead of (κ, θ, ξ, ρ, v₀) to break the degeneracy.

Do NOT proceed to Phase 3 until FNO achieves <10% median relative error on in-distribution data. If it doesn't after reasonable tuning, the issue is likely data (not enough samples, bad parameter prior) or architecture (too few modes, wrong pooling).
