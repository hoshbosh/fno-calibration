# Phase 0: Foundation and Infrastructure

## Context

I am building a research project on amortized parametric calibration of stochastic volatility models using Fourier Neural Operators (FNO). The core idea: train an FNO that takes an implied volatility (IV) surface as input and returns the Heston model parameters that generated it. This replaces iterative calibration (Levenberg-Marquardt + PDE solver) with a single forward pass.

This is Phase 0 of a 6-phase project. The goal is to get a minimal end-to-end pipeline running: generate synthetic Heston IV surfaces, feed them through an FNO, and regress onto Heston parameters. Accuracy will be bad — that's fine. The point is that every piece runs.

## Project structure

Create the following repo structure:

```
neural-operator-calibration/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml          # All hyperparameters and paths
├── data/
│   ├── generate_heston.py    # Synthetic Heston IV surface generator
│   └── dataset.py            # PyTorch Dataset class for HDF5 surfaces
├── models/
│   ├── fno_calibrator.py     # FNO-based calibration operator
│   └── mlp_baseline.py       # MLP baseline (Horvath-style)
├── train.py                  # Training loop
├── eval.py                   # Evaluation and metrics
├── notebooks/
│   └── data_quality.ipynb    # Data visualization and arbitrage audit
└── scripts/
    └── smoke_test.sh         # End-to-end smoke test
```

## Dependencies

```
torch>=2.0
neuraloperator>=0.3
QuantLib-Python>=1.34    # Try this first for Heston pricing
numpy
scipy
h5py
pyyaml
wandb
matplotlib
py_vollib_vectorized     # For fast IV inversion from prices
```

If QuantLib-Python installation fails or is too painful, fall back to a pure numpy/scipy implementation of Heston characteristic function pricing via the COS method (Fang-Oosterlee 2008). The key formula is:

```
Heston characteristic function:
φ(u) = exp(C(u,τ) + D(u,τ)·v₀ + i·u·log(S/K))

where C and D satisfy Riccati ODEs with known closed-form solutions
involving κ (mean reversion), θ (long-run variance), ξ (vol of vol),
ρ (correlation), v₀ (initial variance).
```

Price a European call via Fourier inversion of the characteristic function (Gil-Pelaez or COS method). Then invert Black-Scholes to get implied vol. `py_vollib_vectorized` handles the BS inversion step.

## Data generation: `data/generate_heston.py`

Generate synthetic Heston model IV surfaces. Each surface corresponds to one set of Heston parameters.

**Heston parameter ranges (sample uniformly for now — Latin hypercube in Phase 1):**
- κ (mean reversion speed): [0.5, 5.0]
- θ (long-run variance): [0.02, 0.15]
- ξ (vol of vol): [0.1, 1.0]
- ρ (spot-vol correlation): [-0.95, -0.3]
- v₀ (initial variance): [0.01, 0.12]

Additionally fix: S₀ = 1.0 (normalized spot), r = 0.02 (risk-free rate), q = 0.0 (no dividends).

**Grid:**
- Log-moneyness: k = log(K/F) where F = S₀·exp((r-q)·τ) is the forward. Use 16 points linearly spaced in [-0.5, 0.5].
- Time-to-expiry: τ from 0.02 to 2.0 years. Use 20 points with geometric spacing (denser at short maturities: `np.geomspace(0.02, 2.0, 20)`).

**For each parameter sample:**
1. Compute call prices C(K,τ) for all 16×20 grid points using Heston analytic pricing
2. Convert call prices to implied volatilities σ_IV(k,τ) via Black-Scholes inversion
3. If any IV inversion fails (negative price, numerical issues), flag and skip that sample
4. Store the 16×20 IV surface and the 5 parameters

**Output format:** HDF5 file with datasets:
- `surfaces`: shape (N, 16, 20), dtype float32 — the IV values
- `parameters`: shape (N, 5), dtype float32 — (κ, θ, ξ, ρ, v₀)
- `grid_k`: shape (16,) — log-moneyness grid
- `grid_tau`: shape (20,) — time-to-expiry grid

For the smoke test, generate N=1000 surfaces. This should take under 10 minutes.

**Important numerical issues to handle:**
- Some parameter combinations produce very small or negative call prices at deep OTM strikes. Catch these and either clip or skip.
- IV inversion can fail for very deep OTM options. Use a robust inverter with fallback (bisection if Newton fails).
- Check that all generated surfaces satisfy basic no-arbitrage: IV should be positive everywhere, and total implied variance w = σ²τ should be non-decreasing in τ at each strike.

## Dataset class: `data/dataset.py`

Standard PyTorch Dataset that loads the HDF5 file:
- `__getitem__` returns (surface, parameters) as torch tensors
- surface shape: (1, 16, 20) — 1 channel
- parameters shape: (5,)
- Normalize parameters to zero mean, unit variance using training set statistics
- Store normalization constants (mean, std per parameter) as dataset attributes so we can denormalize at eval time

## FNO calibrator: `models/fno_calibrator.py`

Architecture:

```
Input: IV surface, shape (batch, 1, 16, 20)
  │
  ├── Lifting: pointwise Conv2d(1, 64, kernel_size=1)
  │
  ├── FNO Block 1: SpectralConv2d(64, 64, modes1=12, modes2=12) + Conv2d(64, 64, 1) + GELU
  ├── FNO Block 2: same
  ├── FNO Block 3: same
  ├── FNO Block 4: same
  │
  ├── Projection: pointwise Conv2d(64, 128, 1) → GELU → Conv2d(128, 1, 1)
  │
  ├── Global average pooling over spatial dims → (batch, 1)
  │   Wait — this loses too much info. Instead:
  ├── Flatten after projection: (batch, 128, 16, 20) → (batch, 128*16*20)
  │   Actually this defeats resolution invariance. Use:
  ├── Adaptive average pooling to fixed size (4, 5) → flatten → (batch, 128*4*5)
  │
  ├── Parameter head MLP: Linear(2560, 256) → GELU → Linear(256, 64) → GELU → Linear(64, 5)
  │
  Output: predicted parameters, shape (batch, 5)
```

IMPORTANT: For the FNO to be resolution-invariant, you cannot flatten the spatial output directly (that bakes in the grid size). Use `torch.nn.AdaptiveAvgPool2d((4, 5))` to pool to a fixed spatial size regardless of input resolution, then flatten. This preserves resolution invariance while giving the parameter head enough spatial information.

Use the `neuraloperator` library's `SpectralConv2d` if available, or implement manually:

```python
# Manual SpectralConv2d (if neuraloperator API is awkward):
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.modes1 = modes1
        self.modes2 = modes2

    def forward(self, x):
        # x shape: (batch, channels, size1, size2)
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros_like(x_ft)
        # Multiply low-frequency modes
        out_ft[:, :, :self.modes1, :self.modes2] = \
            torch.einsum("bixy,ioxy->boxy", x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            torch.einsum("bixy,ioxy->boxy", x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
```

Each FNO block:
```python
def forward(self, x):
    return self.activation(self.spectral_conv(x) + self.pointwise_conv(x))
```

## MLP baseline: `models/mlp_baseline.py`

The Horvath-style baseline. Takes the same IV surface but flattened:

```
Input: IV surface flattened, shape (batch, 320)   # 16*20 = 320
  │
  ├── Linear(320, 512) → GELU
  ├── Linear(512, 256) → GELU
  ├── Linear(256, 128) → GELU
  ├── Linear(128, 5)
  │
  Output: predicted parameters, shape (batch, 5)
```

Match the parameter count roughly to the FNO so comparisons are fair. Print param counts for both models at startup.

## Training: `train.py`

- Loss: MSE on normalized parameters
- Optimizer: AdamW, lr=1e-3, weight_decay=1e-4
- Scheduler: CosineAnnealingLR over total epochs
- Batch size: 64
- Epochs: 100 for smoke test (200 for real runs in Phase 2)
- Log to W&B: total loss, per-parameter RMSE (κ, θ, ξ, ρ, v₀ separately), learning rate
- Save best checkpoint by validation loss
- Config via YAML file

```python
# Per-parameter RMSE logging (denormalized):
for i, name in enumerate(['kappa', 'theta', 'xi', 'rho', 'v0']):
    rmse = torch.sqrt(((pred_denorm[:, i] - true_denorm[:, i]) ** 2).mean())
    wandb.log({f'val_rmse/{name}': rmse.item()})
```

## Evaluation: `eval.py`

Load best checkpoint, run on test set, report:
- Per-parameter RMSE (denormalized, in original units)
- Per-parameter relative error: |pred - true| / |true|, median and 95th percentile
- Overall IV reconstruction error: for each test sample, take predicted parameters, re-generate the IV surface using the Heston pricer, compare to the input surface. Report IV RMSE in vol points (e.g., 0.5 vol points = 0.005 in σ units).
- Print a summary table

## Smoke test: `scripts/smoke_test.sh`

```bash
#!/bin/bash
set -e
echo "=== Generating 1000 Heston surfaces ==="
python data/generate_heston.py --n_samples 1000 --output data/heston_smoke.h5

echo "=== Training FNO (20 epochs) ==="
python train.py --config configs/default.yaml --data data/heston_smoke.h5 --epochs 20 --model fno --wandb_mode disabled

echo "=== Training MLP baseline (20 epochs) ==="
python train.py --config configs/default.yaml --data data/heston_smoke.h5 --epochs 20 --model mlp --wandb_mode disabled

echo "=== Evaluating FNO ==="
python eval.py --config configs/default.yaml --data data/heston_smoke.h5 --model fno --checkpoint checkpoints/fno_best.pt

echo "=== Evaluating MLP ==="
python eval.py --config configs/default.yaml --data data/heston_smoke.h5 --model mlp --checkpoint checkpoints/mlp_best.pt

echo "=== Smoke test passed ==="
```

## What success looks like at the end of Phase 0

1. `smoke_test.sh` runs end-to-end without errors
2. Both FNO and MLP produce finite, non-NaN parameter predictions
3. Per-parameter RMSE is logged and visible
4. The accuracy will be mediocre (maybe 20-50% relative error) — that's expected with only 1000 training samples and 20 epochs
5. A W&B project exists with at least one logged run
6. `data_quality.ipynb` shows: a few sample IV surfaces plotted as heatmaps, histograms of parameter distributions, arbitrage violation counts (should be 0 or near-0)

Do not optimize anything yet. Do not tune hyperparameters. Do not worry about accuracy. Just make it run.
