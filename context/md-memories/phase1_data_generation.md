# Phase 1: Serious Synthetic Data Generation

## Context

I am building a research project: an FNO (Fourier Neural Operator) that takes an implied volatility surface as input and returns Heston stochastic volatility model parameters. Phase 0 is complete — I have a working end-to-end pipeline in `neural-operator-calibration/` with:

- `data/generate_heston.py` — generates synthetic Heston IV surfaces
- `data/dataset.py` — PyTorch Dataset for HDF5 surfaces
- `models/fno_calibrator.py` — FNO architecture (4 Fourier layers, 64 channels, adaptive pooling → parameter head)
- `models/mlp_baseline.py` — MLP baseline (flattened surface → 4-layer MLP → parameters)
- `train.py`, `eval.py` — training and evaluation scripts
- `configs/default.yaml` — hyperparameters

Phase 1 goal: scale up from 1000 smoke-test surfaces to a production-quality dataset of 120k surfaces (100k train / 10k val / 10k test) with proper parameter sampling, data quality audits, and an out-of-distribution test set.

## Task 1: Improve parameter sampling

Replace uniform random sampling with Latin Hypercube Sampling for better coverage of the 5D parameter space.

```python
from scipy.stats.qmc import LatinHypercube

# Heston parameter ranges:
# κ (mean reversion):    [0.5,  5.0]
# θ (long-run variance): [0.02, 0.15]
# ξ (vol of vol):        [0.1,  1.0]
# ρ (correlation):       [-0.95, -0.3]
# v₀ (initial variance): [0.01, 0.12]

sampler = LatinHypercube(d=5, seed=42)
samples = sampler.random(n=100000)  # shape (100000, 5), values in [0, 1]
# Scale to parameter ranges
params[:, 0] = samples[:, 0] * (5.0 - 0.5) + 0.5      # κ
params[:, 1] = samples[:, 1] * (0.15 - 0.02) + 0.02    # θ
params[:, 2] = samples[:, 2] * (1.0 - 0.1) + 0.1       # ξ
params[:, 3] = samples[:, 3] * (-0.3 - (-0.95)) + (-0.95)  # ρ
params[:, 4] = samples[:, 4] * (0.12 - 0.01) + 0.01    # v₀
```

Additionally, apply the Feller condition filter: 2κθ > ξ² ensures the variance process stays positive. Samples that violate this should be kept but flagged — they are still valid Heston parameters (the model still works, variance just hits zero occasionally), but they may produce numerical issues. Log how many samples violate Feller.

## Task 2: Scale up data generation

Generate the following datasets, all saved as HDF5:

1. **`data/heston_train_100k.h5`** — 100,000 surfaces, LHS sampling from the full parameter range
2. **`data/heston_val_10k.h5`** — 10,000 surfaces, separate LHS draw (different seed)
3. **`data/heston_test_10k.h5`** — 10,000 surfaces, separate LHS draw (different seed)
4. **`data/heston_ood_5k.h5`** — 5,000 surfaces from the TAILS of the parameter distribution:
   - ρ ∈ [-0.95, -0.85] (extreme negative correlation)
   - ξ ∈ [0.8, 1.2] (high vol-of-vol, extended past training range)
   - κ ∈ [0.5, 1.0] (slow mean reversion)
   - θ and v₀ at their extremes

This is the out-of-distribution test set. Performance here tells us how the model behaves on unseen market regimes.

**Parallelization:** The Heston pricing for each parameter set is independent. Use `multiprocessing.Pool` or `joblib.Parallel` with `n_jobs=-1` to parallelize across CPU cores. Each surface (16×20 = 320 prices + IV inversions) should take ~0.1–0.5 seconds depending on the pricer. 100k surfaces at 0.3s each = ~8 hours single-core, ~1 hour on 8 cores.

Add a progress bar (`tqdm`) and checkpoint: save intermediate results every 10,000 surfaces so a crash doesn't lose everything.

```python
# Checkpointing pattern:
CHUNK_SIZE = 10000
for chunk_start in range(0, n_total, CHUNK_SIZE):
    chunk_end = min(chunk_start + CHUNK_SIZE, n_total)
    # Generate surfaces for this chunk
    # Append to HDF5 (use resizable datasets)
    print(f"Saved {chunk_end}/{n_total}")
```

## Task 3: Grid improvements

Modify the grid to be more financially realistic:

**Log-moneyness grid (strike dimension):**
- 16 points in [-0.5, 0.5]
- Use denser spacing near ATM (k=0) where options are most liquid
- Suggested: `np.concatenate([np.linspace(-0.5, -0.1, 4), np.linspace(-0.1, 0.1, 8), np.linspace(0.1, 0.5, 4)])`
- This gives 4 deep OTM puts, 8 near-ATM, 4 deep OTM calls

**Time-to-expiry grid (maturity dimension):**
- 20 points from 0.02 to 2.0 years
- Geometric spacing: `np.geomspace(0.02, 2.0, 20)` — this gives ~7 points in [0, 0.25] (short-dated) which is where the smile is most complex and where models diverge most

Store the grid coordinates in the HDF5 file as metadata:
```python
f.create_dataset('grid_k', data=grid_k)      # shape (16,)
f.create_dataset('grid_tau', data=grid_tau)   # shape (20,)
```

## Task 4: Arbitrage audit

After generating each dataset, run an automated arbitrage audit and store results. Check three conditions at every grid point:

**1. Butterfly condition (convexity in strike):**
For each maturity τ, the call price C(K,τ) must be convex in K. In IV terms, this is equivalent to the Durrleman condition, but a simpler numerical check is:

```python
# In total variance space: w(k,τ) = σ²(k,τ) · τ
# Second derivative of w w.r.t. k must satisfy the Durrleman condition:
# g(k) = (1 - k·w'/(2w))² - w'²/4·(1/w + 1/4) + w''/2 ≥ 0
# For a simpler check, just verify call prices are convex:
# C(K_{i-1}) - 2·C(K_i) + C(K_{i+1}) ≥ 0 for all interior strikes
```

**2. Calendar spread condition (monotonicity in maturity):**
Total implied variance w(k,τ) = σ²(k,τ)·τ must be non-decreasing in τ at each strike k:
```python
# For each strike k_i:
# w(k_i, τ_{j+1}) ≥ w(k_i, τ_j) for all j
total_var = iv_surface ** 2 * grid_tau[None, :]  # shape (16, 20)
calendar_violations = (np.diff(total_var, axis=1) < -1e-8).sum()
```

**3. Positivity:**
All implied volatilities must be positive: σ(k,τ) > 0 everywhere.

**Expected result:** Heston-generated surfaces should be arbitrage-free by construction. Any violations indicate numerical issues in the pricer or IV inverter. Log the counts:

```
Arbitrage audit for heston_train_100k.h5:
  Total surfaces: 100,000
  Surfaces with butterfly violations: 0 (0.00%)
  Surfaces with calendar violations: 12 (0.01%)  # these are numerical noise
  Surfaces with negative IV: 0 (0.00%)
  Surfaces skipped (pricing failure): 347 (0.35%)
```

If calendar violations exceed 1%, investigate the IV inversion at short maturities — this is the most numerically fragile region.

## Task 5: Data quality notebook

Create `notebooks/data_quality.ipynb` that generates:

1. **Parameter distribution histograms** — 5 histograms (one per parameter), showing the LHS sampling coverage
2. **Sample IV surfaces** — plot 9 randomly selected surfaces as heatmaps with log-moneyness on x-axis and time-to-expiry on y-axis, IV value as color. Title each with its parameter values.
3. **IV smile slices** — for 3 sample surfaces, plot IV vs log-moneyness at 3 different maturities (short τ=0.05, medium τ=0.5, long τ=1.5). This shows the smile shape.
4. **Arbitrage audit summary** — print the violation counts from Task 4
5. **Parameter correlation matrix** — show that LHS gives low correlation between parameters
6. **Feller condition analysis** — what fraction of samples violate 2κθ > ξ², and do these correlate with numerical issues

Save all figures to `figures/data_quality/` as PNGs.

## Task 6: Update dataset class

Update `data/dataset.py` to handle the new datasets:
- Accept a list of HDF5 files (train, val, test) or a single file with a split argument
- Compute normalization statistics (mean, std) from training set only
- Apply same normalization to val/test/ood sets
- Save normalization stats to a JSON file alongside the data so eval.py can denormalize
- Add an option to augment by randomly masking grid points (set to 0 or NaN) — this will be used in Phase 3 for the missing-quotes experiment. Don't enable by default, just make the flag available.

```python
class HestonDataset(Dataset):
    def __init__(self, h5_path, norm_stats=None, mask_fraction=0.0):
        ...
        if norm_stats is None:
            self.param_mean = self.parameters.mean(axis=0)
            self.param_std = self.parameters.std(axis=0)
        else:
            self.param_mean = norm_stats['mean']
            self.param_std = norm_stats['std']

    def __getitem__(self, idx):
        surface = torch.FloatTensor(self.surfaces[idx]).unsqueeze(0)  # (1, 16, 20)
        params = torch.FloatTensor(self.parameters[idx])
        params_norm = (params - self.param_mean) / self.param_std

        if self.mask_fraction > 0:
            mask = torch.rand_like(surface) > self.mask_fraction
            surface = surface * mask

        return surface, params_norm

    def get_norm_stats(self):
        return {'mean': self.param_mean, 'std': self.param_std}
```

## What success looks like at the end of Phase 1

1. Four HDF5 files exist: train (100k), val (10k), test (10k), OOD (5k)
2. Arbitrage violations are < 0.1% across all datasets
3. Data quality notebook shows clean parameter distributions, realistic IV surfaces, and expected smile shapes
4. Normalization stats are saved and consistent across train/val/test
5. Total data generation time is under 3 hours (parallelized)
6. The data pipeline is deterministic (fixed seeds) and reproducible

Do NOT retrain the models yet — that's Phase 2. Just produce the data.
