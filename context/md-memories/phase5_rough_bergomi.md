# Phase 5: Rough Bergomi Extension (Stretch Goal)

## Context

I am building a research project: an FNO for amortized calibration of stochastic volatility models. Phases 0–4 are complete with strong Heston results.

**Completed results:**
- FNO achieves <10% median relative error on Heston parameter recovery
- Resolution invariance demonstrated (Phase 3): FNO generalizes to unseen grids, MLP cannot
- Speed benchmark (Phase 4): FNO is ~1000× faster than Levenberg-Marquardt calibration
- All results on Heston, which has a semi-closed-form characteristic function

**Phase 5 goal:** Extend to rough Bergomi (rBergomi), where the FNO advantage is most dramatic. Rough Bergomi has NO closed-form pricing — calibration requires Monte Carlo simulation, making traditional LM calibration extremely expensive (minutes per calibration). This is where amortized calibration goes from "nice speedup" to "enabling technology."

**This phase is a stretch goal.** If compute budget or time is tight, skip it and submit the paper with Heston only. Heston alone is sufficient for ICAIF/NeurIPS workshop. Rough Bergomi elevates the story significantly but is not required.

## Rough Bergomi model background

The rough Bergomi model (Bayer-Friz-Gatheral 2016) specifies:

```
dS_t / S_t = √(V_t) dW_t
V_t = ξ₀(t) · exp(η · Ŵ_t^H - η²/2 · t^{2H})
```

where:
- Ŵ^H is a fractional Brownian motion with Hurst parameter H ∈ (0, 0.5) (the "rough" part — H < 0.5 means rougher than Brownian motion)
- η > 0 is the vol-of-vol
- ρ is the correlation between the price Brownian motion W and the driving fBM
- ξ₀(t) is the forward variance curve (typically set to the current ATM implied vol level, so v₀ = ξ₀(0))

**Parameters to calibrate:** (H, η, ρ) — only 3 parameters (vs 5 for Heston). The forward variance curve ξ₀(t) is typically treated as given (backed out from the term structure of ATM implied vols), not calibrated.

**Why there's no closed form:** The fractional Brownian motion Ŵ^H is non-Markovian — it has long memory. This means there's no finite-dimensional state space, no PDE, no characteristic function. The only way to price is Monte Carlo simulation of the paths.

## Task 1: Rough Bergomi data generation

Use the hybrid scheme of Bennedsen-Lunde-Pakkanen (2017) for simulating the Volterra process. The reference implementation is Ryan McCrickerd's repo:

```bash
pip install --break-system-packages roughbergomi
# Or clone: git clone https://github.com/ryanmccrickerd/rough_bergomi
```

If that package is unavailable or broken, implement the hybrid scheme manually:

```python
import numpy as np
from scipy.special import gamma

def simulate_rbergomi(H, eta, rho, v0, T, n_steps, n_paths):
    """
    Simulate rough Bergomi paths using the hybrid scheme.

    Args:
        H: Hurst parameter (0, 0.5)
        eta: vol-of-vol
        rho: spot-vol correlation
        v0: initial variance (= ξ₀(0))
        T: time horizon
        n_steps: number of time steps
        n_paths: number of MC paths

    Returns:
        S: terminal stock prices, shape (n_paths,)
    """
    dt = T / n_steps
    # Generate correlated Brownian increments
    dW1 = np.random.randn(n_paths, n_steps) * np.sqrt(dt)
    dW_perp = np.random.randn(n_paths, n_steps) * np.sqrt(dt)
    dW2 = rho * dW1 + np.sqrt(1 - rho**2) * dW_perp

    # Volterra process: Ŵ^H_t = ∫₀ᵗ (t-s)^{H-1/2} dW₂(s)
    # Hybrid scheme: split into coarse Riemann sum + fine correction
    # ... (implementation follows BLP 2017)

    # Variance process
    # V_t = v0 * exp(eta * Ŵ^H_t - eta²/2 * t^{2H})

    # Stock price via Euler scheme
    # log(S_T) = -1/2 ∫V_t dt + ∫√V_t dW₁

    return S_terminal
```

The key numerical detail: the kernel $(t-s)^{H-1/2}$ is singular at $s = t$, so the hybrid scheme splits the integral into a "near" part (handled analytically) and a "far" part (Riemann sum). Get this right or the MC estimator will be biased.

**Parameter ranges for data generation:**
```python
# Rough Bergomi parameters:
H_range = [0.05, 0.40]     # Hurst exponent (empirical H ≈ 0.1 for SPX)
eta_range = [0.5, 3.0]     # Vol-of-vol (higher than Heston's ξ due to different parameterization)
rho_range = [-0.95, -0.3]  # Same correlation range as Heston
v0_range = [0.01, 0.12]    # Initial variance (same range)
```

**MC settings per surface:**
- n_paths = 100,000 (for ~0.5% MC standard error on ATM options)
- n_steps = 252 (daily steps for T=1)
- Use antithetic variates to reduce variance

**Pricing:** For each (K, τ) on the 16×20 grid, simulate paths out to time τ, compute payoff max(S_τ - K, 0), take the mean, discount. Convert to IV via Black-Scholes inversion.

**Data volume:** Generate 50k surfaces (30k train / 10k val / 10k test). Each surface takes ~10–30 seconds with 100k paths. Total: ~140–400 hours single-core. **Parallelize aggressively:**

```python
from joblib import Parallel, delayed

def generate_one_surface(params, grid_k, grid_tau, n_paths=100000):
    H, eta, rho, v0 = params
    surface = np.zeros((len(grid_k), len(grid_tau)))
    for j, tau in enumerate(grid_tau):
        for i, k in enumerate(grid_k):
            K = np.exp(k)  # Forward moneyness
            price = price_call_rbergomi(H, eta, rho, v0, K, tau, n_paths)
            surface[i, j] = bs_iv(price, K, tau)
    return surface

# Parallel across surfaces (not across grid points — each surface is one MC run)
results = Parallel(n_jobs=-1, verbose=10)(
    delayed(generate_one_surface)(params[i], grid_k, grid_tau)
    for i in range(n_total)
)
```

On an 8-core spot instance: ~17–50 hours. On a 32-core instance: ~4–12 hours. Budget $50–150 of compute for this.

**Checkpointing:** Save every 1000 surfaces. A crash at surface 45,000 should not lose earlier work.

## Task 2: Train FNO on rough Bergomi

**Option A: Train from scratch.**
Same FNO architecture as Heston, but output head predicts 3 parameters (H, η, ρ) instead of 5. If v₀ is treated as given (not calibrated), use 3-output head. If v₀ is also calibrated, use 4-output head.

**Option B: Fine-tune from Heston (recommended).**
Start from the Heston-trained FNO checkpoint. Replace only the parameter head (last MLP layers). The Fourier layers have already learned what IV surfaces "look like" — the smile structure, the term structure, the curvature patterns. This prior should help with the smaller rBergomi dataset.

```python
# Load Heston-trained FNO
model = FNOCalibrator.load('checkpoints/fno_heston_best.pt')

# Replace parameter head for 3 rBergomi params
model.param_head = nn.Sequential(
    nn.Linear(model.pool_flat_dim, 256),
    nn.GELU(),
    nn.Linear(256, 64),
    nn.GELU(),
    nn.Linear(64, 3)  # H, η, ρ
)

# Fine-tune with lower LR for Fourier layers, higher for new head
optimizer = AdamW([
    {'params': model.fourier_layers.parameters(), 'lr': 1e-4},  # 10× lower
    {'params': model.param_head.parameters(), 'lr': 1e-3},      # normal
], weight_decay=1e-4)
```

**Train for 100–200 epochs on 30k rBergomi surfaces.** Monitor the same per-parameter metrics as Phase 2.

## Task 3: Speed comparison — where the story shines

The speed comparison for rough Bergomi is dramatically better than for Heston:

**Traditional rBergomi calibration:**
LM with MC pricing. Each forward eval = one MC simulation (~10–30 seconds). LM needs 50–200 iterations. Total: **~10–100 minutes per calibration.**

```python
def calibrate_rbergomi_lm(observed_iv, grid_k, grid_tau):
    """This is INTENTIONALLY SLOW — it demonstrates the problem."""
    n_evals = [0]
    def residuals(params):
        n_evals[0] += 1
        H, eta, rho = params
        model_iv = rbergomi_iv_surface(H, eta, rho, v0_given, grid_k, grid_tau, n_paths=50000)
        return (model_iv - observed_iv).ravel()

    t_start = time.perf_counter()
    result = least_squares(residuals, [0.1, 1.5, -0.7],
                          bounds=([0.05, 0.5, -0.95], [0.4, 3.0, -0.3]),
                          method='trf', max_nfev=100)  # Cap at 100 evals
    wall_time = time.perf_counter() - t_start
    return result.x, wall_time, n_evals[0]
```

Run on just 10–20 test samples (each takes minutes). This is enough to establish the timing.

**FNO rBergomi calibration:** Same as Heston — one forward pass, ~0.05ms on GPU.

**Speedup ratio:** ~10 minutes / 0.05ms = ~12,000,000×. Obviously the LM result is more accurate, but the speedup is six orders of magnitude. Even if FNO has 5% parameter error, this is transformative for real-time rough vol calibration.

## Task 4: Cross-model transfer experiment (optional bonus)

Test whether the Heston-pretrained FNO has learned transferable features:

1. Take the Heston-trained FNO (no fine-tuning on rBergomi)
2. Feed it rBergomi IV surfaces
3. It will output 5 Heston parameters — these are wrong, but do they capture something about the surface?
4. Re-price from these "wrong" Heston parameters and compare to the rBergomi surface

The hypothesis: if the Heston approximation to rBergomi surfaces has low IV RMSE, it means the Heston FNO has learned a useful representation of IV surface structure, even for surfaces not generated by Heston. This motivates the fine-tuning approach.

## Task 5: Compile rBergomi results

Add to the existing results tables from Phase 4:

```
| Method                  | Model   | Speed (per surface) | IV RMSE | Param RMSE (H) | Param RMSE (ρ) |
|-------------------------|---------|--------------------:|--------:|---------------:|---------------:|
| LM + MC (100 evals)     | rBergomi|          ~10–60 min |   ~0.02 |          ~0.01 |          ~0.01 |
| FNO from scratch        | rBergomi|           ~0.05 ms  |     TBD |            TBD |            TBD |
| FNO fine-tuned          | rBergomi|           ~0.05 ms  |     TBD |            TBD |            TBD |
```

Create `figures/rbergomi_speed_comparison.png` — a bar chart with log-scale y-axis showing calibration time: LM+MC vs FNO. The bars should be comically different in height.

## What success looks like at the end of Phase 5

- FNO recovers rBergomi parameters (H, η, ρ) with <15% median relative error
- H recovery is particularly important — getting H right distinguishes rough vol from classical vol
- Fine-tuning from Heston beats training from scratch (demonstrates transfer learning)
- Speed comparison shows 10⁵–10⁷× speedup over LM+MC
- Total compute cost for data generation stays under $150

**Failure modes:**
- MC noise in training data makes the FNO noisy → increase n_paths to 200k (more expensive but cleaner)
- H is unrecoverable → H mostly affects short-maturity smile curvature, so check if your grid has enough short-maturity points (τ < 0.1). If not, add more.
- Total compute exceeds budget → reduce to 20k training surfaces and accept higher error

**If Phase 5 fails or is cut:** Submit the paper with Heston only. The contribution (amortized calibration + resolution invariance) stands without rBergomi. Mention rBergomi as future work.
