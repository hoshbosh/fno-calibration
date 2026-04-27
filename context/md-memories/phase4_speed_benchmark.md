# Phase 4: Calibration Speed Benchmark

## Context

I am building a research project: an FNO for amortized Heston calibration. The FNO takes an IV surface and returns model parameters in one forward pass.

**Phases 0–3 are complete.** Key results so far:
- FNO achieves <10% median relative error on Heston parameter recovery (16×20 grid)
- FNO generalizes to unseen grid resolutions (Phase 3 resolution invariance experiment)
- MLP baseline requires retraining or interpolation for new grids
- Full repo at `neural-operator-calibration/`

**Phase 4 goal:** Benchmark calibration speed. Compare FNO single-pass calibration against traditional iterative calibration (Levenberg-Marquardt with Heston analytic pricer) and against SVI parametric fitting. Produce the speed-vs-accuracy tradeoff table that goes in the paper.

## Task 1: Traditional Heston calibration baseline

Implement Levenberg-Marquardt calibration using `scipy.optimize.least_squares`:

```python
import numpy as np
from scipy.optimize import least_squares

def calibrate_heston_lm(observed_iv, grid_k, grid_tau, initial_guess=None):
    """
    Calibrate Heston parameters to observed IV surface using Levenberg-Marquardt.

    Args:
        observed_iv: shape (16, 20) — the target IV surface
        grid_k: shape (16,) — log-moneyness grid
        grid_tau: shape (20,) — time-to-expiry grid
        initial_guess: (κ, θ, ξ, ρ, v₀) — if None, use defaults

    Returns:
        params: calibrated (κ, θ, ξ, ρ, v₀)
        n_evals: number of forward model evaluations
        wall_time: total calibration time in seconds
        final_rmse: IV RMSE at convergence
    """
    if initial_guess is None:
        initial_guess = np.array([2.0, 0.05, 0.4, -0.7, 0.04])

    # Parameter bounds
    bounds_lower = [0.5, 0.02, 0.1, -0.95, 0.01]
    bounds_upper = [5.0, 0.15, 1.0, -0.3, 0.12]

    n_evals = [0]  # mutable counter

    def residuals(params):
        kappa, theta, xi, rho, v0 = params
        n_evals[0] += 1
        # Price using same Heston pricer from data generation
        model_iv = heston_iv_surface(kappa, theta, xi, rho, v0, grid_k, grid_tau)
        return (model_iv - observed_iv).ravel()  # shape (320,)

    t_start = time.time()
    result = least_squares(
        residuals, initial_guess,
        bounds=(bounds_lower, bounds_upper),
        method='trf',  # Trust Region Reflective (handles bounds)
        ftol=1e-8, xtol=1e-8, gtol=1e-8,
        max_nfev=500
    )
    wall_time = time.time() - t_start

    return {
        'params': result.x,
        'n_evals': n_evals[0],
        'wall_time': wall_time,
        'final_rmse': np.sqrt(np.mean(result.fun ** 2)),
        'success': result.success
    }
```

**Run on 1000 test samples** (subsample from the 10k test set). For each:
- Use the TRUE parameters as initial guess ± 20% perturbation (simulates a warm start from yesterday's calibration)
- Also run with a COLD start (fixed initial guess = [2.0, 0.05, 0.4, -0.7, 0.04]) for 100 samples to show the harder case
- Record: wall time, number of forward evals, convergence flag, final IV RMSE, parameter recovery error

**Important timing notes:**
- Use `time.perf_counter()` not `time.time()` for sub-millisecond precision
- Run on CPU only for LM (it's not GPU-accelerated) — this is the fair comparison since practitioners run LM on CPU
- Warm up the pricer with a dummy call before timing (JIT/cache effects)

## Task 2: FNO calibration timing

Time the FNO forward pass on the same 1000 test samples:

```python
def calibrate_fno(model, surface_tensor, device='cuda'):
    """
    Calibrate via FNO forward pass.

    Returns:
        params: predicted (κ, θ, ξ, ρ, v₀)
        wall_time: inference time in seconds
    """
    model.eval()
    surface_tensor = surface_tensor.to(device)

    # Warm up
    with torch.no_grad():
        _ = model(surface_tensor[:1])

    # Time batched inference
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    with torch.no_grad():
        pred = model(surface_tensor)
    torch.cuda.synchronize()
    wall_time = time.perf_counter() - t_start

    return {
        'params': denormalize(pred).cpu().numpy(),
        'wall_time': wall_time,
        'per_sample_time': wall_time / len(surface_tensor)
    }
```

Report:
- **Batched** (all 1000 at once): total time and per-sample time
- **Single-sample** (batch size 1): per-sample time (this is the real-time calibration scenario)
- **CPU inference**: same timing but with `device='cpu'` (fair comparison with LM)
- **GPU inference**: timing on GPU (the advantage scenario)

## Task 3: SVI baseline timing

SVI (Stochastic Volatility Inspired) is the practitioner baseline for fitting vol smiles. It fits 5 parameters per maturity slice:

```
w(k) = a + b * (ρ_svi * (k - m) + sqrt((k - m)² + σ²))
```

where w = σ²τ is total implied variance, and (a, b, ρ_svi, m, σ) are the SVI parameters.

**Important framing for the paper:** SVI fits the surface shape but does NOT recover Heston dynamics parameters. It tells you nothing about κ, θ, ξ, ρ, v₀. So SVI and your FNO solve DIFFERENT problems. Include SVI timing to show: "SVI fits a surface in X ms, but it doesn't give you model parameters. Our FNO gives you model parameters in Y ms." The comparison is about what you GET, not just speed.

Implement a simple per-slice SVI fitter:
```python
from scipy.optimize import minimize

def fit_svi_slice(k_grid, iv_slice, tau):
    """Fit SVI to one maturity slice."""
    w = iv_slice ** 2 * tau  # total variance

    def svi(params, k):
        a, b, rho, m, sigma = params
        return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

    def objective(params):
        return np.sum((svi(params, k_grid) - w) ** 2)

    x0 = [w.mean(), 0.1, -0.5, 0.0, 0.1]
    bounds = [(1e-6, None), (1e-6, None), (-1, 1), (-1, 1), (1e-6, None)]
    result = minimize(objective, x0, bounds=bounds, method='L-BFGS-B')
    return result.x, result.fun

def fit_svi_surface(iv_surface, grid_k, grid_tau):
    """Fit SVI independently to each maturity slice."""
    t_start = time.perf_counter()
    svi_params = []
    for j in range(len(grid_tau)):
        params, _ = fit_svi_slice(grid_k, iv_surface[:, j], grid_tau[j])
        svi_params.append(params)
    wall_time = time.perf_counter() - t_start
    return np.array(svi_params), wall_time
```

Run on the same 1000 test samples. SVI should be very fast (~1–5ms per surface).

## Task 4: Speed-accuracy tradeoff analysis

For each method, compute both speed and accuracy:

**LM + Heston:**
- Speed: wall time per calibration (warm start and cold start)
- Accuracy: IV RMSE at convergence, parameter recovery RMSE

**FNO:**
- Speed: per-sample inference time (GPU and CPU)
- Accuracy: IV RMSE, parameter recovery RMSE (from Phase 2 eval)

**SVI:**
- Speed: per-surface fitting time
- Accuracy: IV fit RMSE only (no parameter recovery — different output)

**Construct the Pareto frontier:** Plot accuracy (x-axis, IV RMSE) vs speed (y-axis, log scale, seconds per calibration). FNO should be in the bottom-left corner (fast + accurate). LM should be in the top-left (accurate but slow). SVI should be in the bottom-right (fast but different output). Save as `figures/speed_accuracy_pareto.png`.

## Task 5: Compile results table

Create the paper-ready comparison table:

```
| Method                  | Output          | Speed (per surface) | IV RMSE (vol pts) | Param RMSE (ρ) | Notes                    |
|-------------------------|-----------------|--------------------:|------------------:|---------------:|--------------------------|
| LM + Heston (warm)      | Heston params   |          ~0.5–2.0 s |             ~0.01 |          ~0.01 | Iterative, 50–200 evals  |
| LM + Heston (cold)      | Heston params   |          ~2.0–5.0 s |             ~0.02 |          ~0.02 | Needs good initial guess |
| MLP (GPU)               | Heston params   |           ~0.01 ms  |              TBD  |            TBD | Fixed grid only          |
| FNO (GPU)               | Heston params   |           ~0.05 ms  |              TBD  |            TBD | Any grid resolution      |
| FNO (CPU)               | Heston params   |           ~1–5 ms   |              TBD  |            TBD | Deployable without GPU   |
| SVI (per slice)         | Surface params  |           ~1–5 ms   |             ~0.10 |            N/A | No model dynamics        |
```

Save as `results/speed_benchmark.csv` and `results/speed_benchmark.tex`.

## Task 6: Failure analysis

Identify cases where FNO calibration fails badly (>20% relative error on any parameter). For each failure:
1. Record the true parameters
2. Record the FNO prediction
3. Compute the LM result for comparison — does LM also struggle with these?
4. Plot the IV surface and annotate what's "hard" about it (usually: high ξ + extreme ρ, or near Feller boundary)

This goes in the paper as an honest limitations discussion. Save to `results/failure_analysis.csv`.

## What success looks like at the end of Phase 4

- FNO calibration is 100–10000× faster than LM (depending on warm/cold start)
- FNO accuracy is within a factor of 2–5 of LM accuracy (LM finds the optimum; FNO approximates it)
- The speed-accuracy Pareto plot clearly shows FNO dominates the practical operating region
- SVI comparison is included with honest framing (different output, not a direct competitor)
- Failure cases are documented, and they correlate with parameter regions, not random

This phase is about collecting numbers, not building new infrastructure. Most of the code is evaluation scripts. Budget 15–20 hours.
