# Phase 3: Resolution Invariance Experiments

## Context

I am building a research project: an FNO (Fourier Neural Operator) for amortized calibration of the Heston stochastic volatility model. The FNO takes an implied volatility surface and returns model parameters in a single forward pass.

**Phases 0–2 are complete.** The repo `neural-operator-calibration/` contains:
- Trained FNO checkpoint at `checkpoints/fno_best.pt` — achieves <10% median relative error on all 5 Heston parameters on the in-distribution 16×20 test set
- Trained MLP checkpoint at `checkpoints/mlp_best.pt` — comparable accuracy on the same 16×20 grid
- Full training infrastructure, data pipeline, evaluation scripts
- 100k Heston training surfaces on a 16×20 grid (log-moneyness × time-to-expiry)
- The FNO uses SpectralConv2d (Fourier layers) + AdaptiveAvgPool2d before the parameter head, so it can accept variable-resolution inputs

**Phase 3 is the make-or-break experiment for the paper.** The core claim is: the FNO generalizes to grid resolutions it was never trained on, while the MLP cannot. If this experiment fails to show a clear advantage, the contribution weakens significantly.

## Task 1: Multi-resolution test data generation

Generate Heston IV surfaces on multiple grid resolutions using the SAME 10k test parameter sets from `data/heston_test_10k.h5`. The parameters are identical — only the grid changes. This ensures any performance difference is due to the architecture's handling of resolution, not different test samples.

**Grids to generate:**

```python
grids = {
    'coarse_8x10':    {'n_k': 8,  'n_tau': 10},  # Half the training resolution
    'train_16x20':    {'n_k': 16, 'n_tau': 20},   # Same as training (sanity check)
    'fine_32x40':     {'n_k': 32, 'n_tau': 40},   # Double the training resolution
    'fine_64x80':     {'n_k': 64, 'n_tau': 80},   # 4× the training resolution
    'asymmetric_12x30': {'n_k': 12, 'n_tau': 30}, # More maturities than strikes
    'asymmetric_24x10': {'n_k': 24, 'n_tau': 10}, # More strikes than maturities
}
```

For each grid, use the same domain ranges:
- Log-moneyness k ∈ [-0.5, 0.5] — same range, different number of points
- Time-to-expiry τ ∈ [0.02, 2.0] — same range, same geometric spacing rule, different number of points

Use the same Heston pricer from Phase 1. Save each as a separate HDF5 file:
```
data/multiresolution/heston_test_8x10.h5
data/multiresolution/heston_test_16x20.h5    # Should match existing test set
data/multiresolution/heston_test_32x40.h5
data/multiresolution/heston_test_64x80.h5
data/multiresolution/heston_test_12x30.h5
data/multiresolution/heston_test_24x10.h5
```

Verify that `heston_test_16x20.h5` produces identical results to the original test set (same parameters → same IVs on the same grid).

## Task 2: Evaluate FNO on all grids

Load the trained FNO checkpoint (trained on 16×20 only). For each test grid:

1. Load the surface data — shape will be (10000, 1, n_k, n_tau)
2. Forward pass through the FNO — the SpectralConv2d handles variable input sizes via FFT, and AdaptiveAvgPool2d maps any spatial size to the fixed (4, 5) that the parameter head expects
3. Denormalize predicted parameters
4. Compute per-parameter RMSE and relative error
5. Compute IV reconstruction RMSE (re-price from predicted params on the ORIGINAL grid used for that test set)

**Important:** the FNO's Fourier layer truncates high-frequency modes. When the input grid has MORE points than training (32×40, 64×80), the FFT produces more modes but only the lowest `n_modes` are used — the rest are zeroed. This should work but may show some degradation. When the input grid has FEWER points (8×10), the FFT produces fewer modes — if `n_modes` exceeds the available modes, you need to handle this gracefully (only use available modes, pad the rest with zeros). Check that the SpectralConv2d implementation handles this.

```python
# Potential issue with small grids:
# If input is 8×10 and n_modes=[12,12], the FFT only produces 8×6 modes (rfft2)
# SpectralConv2d must not index beyond available modes
# Fix: min(self.modes1, x.size(-2)), min(self.modes2, x.size(-1)//2 + 1)
```

## Task 3: Evaluate MLP on all grids — demonstrating the limitation

The MLP expects a flattened 320-dimensional input (16×20). It CANNOT directly process other grid sizes. Demonstrate this limitation with three approaches:

**Approach A: Interpolation pre-processing.**
For each non-16×20 grid, interpolate the surface back to 16×20 using `scipy.interpolate.RegularGridInterpolator` (bilinear or bicubic), then flatten and feed to the MLP. This is the "fair" baseline — the MLP gets the data, just pre-processed.

```python
from scipy.interpolate import RegularGridInterpolator

def interpolate_to_training_grid(surface, grid_k_source, grid_tau_source, grid_k_target, grid_tau_target):
    """Interpolate surface from source grid to target (training) grid."""
    interp = RegularGridInterpolator(
        (grid_k_source, grid_tau_source), surface,
        method='linear', bounds_error=False, fill_value=None
    )
    K_target, T_target = np.meshgrid(grid_k_target, grid_tau_target, indexing='ij')
    return interp(np.stack([K_target.ravel(), T_target.ravel()], axis=-1)).reshape(len(grid_k_target), len(grid_tau_target))
```

**Approach B: Retrained MLP (upper bound).**
For the 32×40 grid, retrain the MLP from scratch on 32×40 training data (generate 100k surfaces on the 32×40 grid — reuse the same parameters, just different grid). This shows what the MLP COULD do if you retrained it for each new grid. This is the "unfair" comparison that highlights the FNO's advantage: FNO uses one checkpoint for all grids; MLP needs a separate checkpoint per grid.

Only do this for one or two grid sizes (32×40 and 8×10) to make the point — retraining for all 6 grids would be expensive and the argument is clear from 2 examples.

**Approach C: Direct failure.**
Show that feeding a raw 8×10 surface (80 values) or 32×40 surface (1280 values) into the MLP raises a dimension mismatch error. This is trivial but worth documenting — it demonstrates the structural limitation.

## Task 4: Variable quote count experiment

Simulate realistic market conditions where some strikes/maturities have no quoted options.

**Experiment:** Take the 16×20 test surfaces. For each, randomly mask a fraction of grid points (set to 0.0). Test fractions: 0%, 10%, 20%, 30%, 40%, 50%.

For the FNO: feed the masked surface directly. The zeros will propagate through the Fourier layers — not ideal, but the model may be robust to partial corruption.

For the MLP: same masking applied to the flattened vector. Zeros in specific positions.

**Better masking strategy (if simple zeros hurt both models equally):**
Instead of setting masked values to 0, set them to the mean IV of the surface (so they don't inject false signal). Or use a separate mask channel:

```python
# Two-channel input: (IV surface, binary mask)
surface_masked = surface.clone()
mask = torch.ones_like(surface)
drop_indices = torch.randperm(16 * 20)[:int(mask_fraction * 16 * 20)]
surface_masked.view(-1)[drop_indices] = 0
mask.view(-1)[drop_indices] = 0
input = torch.cat([surface_masked, mask], dim=0)  # shape (2, 16, 20)
```

This requires retraining the FNO with 2 input channels. If time allows, do this. If not, the zero-masking experiment still demonstrates robustness.

**Report:** Parameter recovery RMSE vs mask fraction, separately for FNO and MLP. The FNO's spectral representation should be more robust to point-level corruption because the Fourier transform distributes information across modes.

## Task 5: Irregular grid experiment

Create a "market-realistic" grid that mimics actual SPX option availability:

```python
# Realistic SPX-like grid:
# Strikes: cluster near ATM, sparse in wings
grid_k_market = np.array([-0.4, -0.3, -0.2, -0.15, -0.10, -0.07, -0.05, -0.03, -0.01,
                            0.01, 0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30])  # 17 points, irregular

# Maturities: match typical expiry dates (weekly, monthly, quarterly)
grid_tau_market = np.array([0.019, 0.038, 0.058, 0.077, 0.096,  # weekly out to 5 weeks
                             0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0])  # monthly/quarterly
# 12 maturities, 17 strikes = 204 points (vs 320 in training)
```

Generate test surfaces on this grid. Evaluate FNO (which handles 17×12 natively) vs MLP (which needs interpolation to 16×20). This is the most realistic test of the resolution-invariance claim.

## Task 6: Results compilation

Create `scripts/compile_resolution_results.py` that generates:

**Figure 1 (the key figure for the paper):** Line plot with:
- X-axis: grid resolution (labeled as "8×10", "16×20", "32×40", "64×80")
- Y-axis: parameter recovery RMSE (averaged across all 5 parameters, or show ρ specifically since it's the hardest)
- Lines: FNO (blue), MLP+interpolation (red dashed), MLP retrained (red solid, only at 2 points)
- The FNO line should be roughly flat (modest degradation). The MLP+interpolation line should degrade at non-training resolutions. The retrained MLP should match FNO at those specific resolutions (showing the MLP CAN work, but only if retrained).

**Figure 2:** Parameter recovery RMSE vs mask fraction (missing quotes experiment).

**Figure 3:** Scatter plot of predicted vs true ρ on the market-realistic irregular grid, for FNO vs MLP+interpolation. ρ is chosen because it's the most sensitive to smile shape and the hardest to recover from corrupted/interpolated data.

**Table:** Full results matrix:

```
| Grid       | FNO κ RMSE | FNO ρ RMSE | FNO IV RMSE | MLP+interp κ RMSE | MLP+interp ρ RMSE | MLP+interp IV RMSE |
|------------|------------|------------|-------------|--------------------|--------------------|---------------------|
| 8×10       |            |            |             |                    |                    |                     |
| 16×20      |            |            |             |                    |                    |                     |
| 32×40      |            |            |             |                    |                    |                     |
| 64×80      |            |            |             |                    |                    |                     |
| 12×30      |            |            |             |                    |                    |                     |
| 24×10      |            |            |             |                    |                    |                     |
| 17×12 mkt  |            |            |             |                    |                    |                     |
```

Save as CSV and LaTeX.

## What success looks like at the end of Phase 3

**The core claim holds if:**
- FNO RMSE on non-training grids is within 2× of its performance on the training grid (16×20)
- MLP+interpolation RMSE degrades significantly (>3×) on non-training grids
- FNO on the market-realistic irregular grid produces reasonable results (< 15% relative error)

**The claim partially holds if:**
- FNO degrades on coarser grids (8×10) but handles finer grids (32×40, 64×80) well
- This is still publishable — frame it as "zero-shot super-resolution transfer" rather than "full resolution invariance"

**The claim fails if:**
- FNO RMSE on non-training grids is comparable to MLP+interpolation RMSE
- This means the Fourier layers aren't providing the invariance they're supposed to
- Debug: check that the SpectralConv2d is correctly handling variable input sizes, check that the adaptive pooling isn't destroying information
- If it's unfixable: pivot the contribution to speed (amortized vs iterative) and drop resolution invariance from the paper

Phase 3 is where you know whether you have a paper or need to pivot. Spend the time to get clean results here.
