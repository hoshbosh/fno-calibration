# Financial FNO Architecture Specification
# Save this file alongside the phase prompts. Give it to Claude Code at the start of any session.

## Overview

This is a modified Fourier Neural Operator for amortized Heston calibration.
Input: an implied volatility surface on an arbitrary strike × maturity grid.
Output: 5 Heston stochastic volatility parameters (κ, θ, ξ, ρ, v₀).

The standard FNO solves forward PDEs (function → function). This FNO solves an
inverse problem (function → parameter vector). Five modifications from standard FNO:

1. Input augmentation: 3 channels (IV + coordinates) instead of 1
2. Anisotropic Fourier modes: more modes in strike than maturity
3. Progressive mode reduction: later layers keep fewer modes
4. Adaptive pooling: collapses variable-resolution spatial dims to fixed vector
5. Dual-head readout: parameter head + optional auxiliary head

---

## Input Construction

```python
# At each grid point (k_i, τ_j), the input has 3 values:
# Channel 0: σ_IV(k_i, τ_j)   — the implied volatility
# Channel 1: k_i               — log-moneyness coordinate
# Channel 2: τ_j               — time-to-expiry coordinate

# Grid ranges:
# k (log-moneyness): [-0.5, 0.5], 16 points, denser near ATM
# τ (time-to-expiry): [0.02, 2.0], 20 points, geometric spacing

# Construction:
iv_surface = ...  # shape (batch, 1, n_k, n_tau)
k_grid = ...      # shape (n_k,)
tau_grid = ...     # shape (n_tau,)
K, T = torch.meshgrid(k_grid, tau_grid, indexing='ij')
k_channel = K.unsqueeze(0).unsqueeze(0).expand(batch, 1, n_k, n_tau)
tau_channel = T.unsqueeze(0).unsqueeze(0).expand(batch, 1, n_k, n_tau)
x = torch.cat([iv_surface, k_channel, tau_channel], dim=1)  # (batch, 3, n_k, n_tau)
```

Why: The grid is non-uniform (geometric spacing in τ). The FFT operates on grid
indices, not physical coordinates. Without coordinate channels, the network can't
distinguish τ=0.05 from τ=1.5 — it only sees "grid position 3 vs grid position 15."
Coordinates give position awareness. They also change correctly when the grid
resolution changes, supporting resolution invariance.

---

## Architecture (layer by layer)

### Lifting
```python
self.lifting = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=1)
```
Pointwise. Maps each grid point from 3 values to 64 features. No spatial mixing.
Same weights applied at every grid point.

### Fourier Layers (×4)

Each layer has two parallel paths that merge:

```python
class FourierBlock(nn.Module):
    def __init__(self, channels, modes1, modes2):
        self.spectral_conv = SpectralConv2d(channels, channels, modes1, modes2)
        self.pointwise_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x):
        return x + self.activation(self.spectral_conv(x) + self.pointwise_conv(x))
        #      ^-- residual connection (stabilizes training)
```

SPECTRAL PATH (self.spectral_conv):
- FFT the input: torch.fft.rfft2(x)
- Keep only lowest (modes1, modes2) Fourier modes, zero the rest
- Multiply kept modes by learned complex weight tensor R_φ
  Shape of R_φ: (in_channels, out_channels, modes1, modes2), complex
  Two weight tensors: one for positive frequencies, one for negative (first dim)
- Inverse FFT back to physical space: torch.fft.irfft2(out, s=input_size)
- This path captures global, low-frequency structure (overall smile shape, term structure tilt)

LOCAL PATH (self.pointwise_conv):
- 1×1 Conv2d applied independently at each grid point
- Mixes channels but no spatial mixing
- Passes through all frequency content including high-frequency detail
- Captures sharp local features the spectral truncation would destroy

MERGE: sum spectral + local outputs, apply GELU, add residual from input

PROGRESSIVE MODE SCHEDULE (anisotropic, decreasing):
```python
layer_modes = [
    (12, 8),   # Layer 1: full spectral view
    (10, 7),   # Layer 2: starting to focus
    (8, 6),    # Layer 3: narrowing
    (6, 5),    # Layer 4: only coarse structure needed for parameter readout
]
# First number = strike modes (more, because smile shape is complex)
# Second number = maturity modes (fewer, because term structure is smoother)
```

Why anisotropic: The IV smile varies rapidly in strike (skew, curvature, wings)
but smoothly in maturity (gradual flattening). More spectral resolution needed
in strike direction.

Why progressive reduction: Early layers detect full-spectrum features (sharp skew,
wing curvature). Later layers aggregate toward parameter-predictive representations
which are inherently low-frequency (a single ρ value affects the entire surface).
Also reduces parameter count and may regularize.

IMPORTANT FOR RESOLUTION INVARIANCE:
When input has fewer grid points than modes (e.g., 8×10 input with modes=[12,8]),
SpectralConv2d must gracefully handle this:
```python
# Inside SpectralConv2d.forward:
actual_modes1 = min(self.modes1, x.size(-2))
actual_modes2 = min(self.modes2, x.size(-1) // 2 + 1)
# Only multiply modes that exist; rest stay zero
```

### Projection
```python
self.projection = nn.Sequential(
    nn.Conv2d(64, 128, kernel_size=1),
    nn.GELU(),
    nn.Conv2d(128, 64, kernel_size=1)
)
```
Pointwise. One final nonlinear channel transformation before spatial aggregation.
Important because pooling (next step) averages over space — having a nonlinearity
before averaging is more expressive than averaging then applying nonlinearity.

Output shape still: (batch, 64, n_k, n_tau)

### Adaptive Pooling
```python
self.pool = nn.AdaptiveAvgPool2d((4, 5))
# Input: (batch, 64, n_k, n_tau) — any n_k, n_tau
# Output: (batch, 64, 4, 5) — always
# Flatten: (batch, 1280) — always
```

This is the bridge between function space and vector space.
- 16×20 input → each bin averages a 4×4 region
- 32×40 input → each bin averages an 8×8 region
- 8×10 input → each bin averages a 2×2 region
Output is always (4, 5) = 20 spatial cells × 64 channels = 1280 features.

Pool size (4, 5) preserves coarse spatial structure:
- Left vs right (skew direction → ρ information)
- Short vs long maturity (term structure → κ information)
Global average pooling (1,1) would lose this. Full resolution (16,20) would
lose resolution invariance.

ALTERNATIVE (try as ablation): Spectral pooling
```python
# Instead of spatial averaging, keep lowest Fourier modes:
x_ft = torch.fft.rfft2(x)
x_trunc = x_ft[:, :, :4, :3]  # (batch, 64, 4, 3), complex
x_flat = torch.cat([x_trunc.real, x_trunc.imag], dim=1).flatten(1)
# Output: (batch, 1536) — fixed regardless of input resolution
```
Theoretically cleaner (preserves spectral structure the FNO built up).
Compare against adaptive pooling in Phase 2 ablations.

### Parameter Head (primary output)
```python
self.param_head = nn.Sequential(
    nn.Linear(1280, 256),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(256, 64),
    nn.GELU(),
    nn.Dropout(0.1),
    nn.Linear(64, 5)  # → (κ, θ, ξ, ρ, v₀) normalized
)
```

Output is normalized (zero mean, unit variance per parameter using training
set statistics). Denormalize at evaluation time.

No output activation by default. If out-of-range predictions appear at test time,
add sigmoid scaling:
```python
# θ_raw = head output (unbounded)
# θ_scaled = param_min + (param_max - param_min) * sigmoid(θ_raw)
```

### Auxiliary Head (optional, for regularization)
```python
self.aux_head = nn.Sequential(
    nn.Linear(1280, 64),
    nn.GELU(),
    nn.Linear(64, 3)  # → (ATM_vol_at_τ1, skew_slope, smile_curvature)
)
```

Predicts derived surface statistics from the same pooled representation.
Forces the network to learn financially meaningful features, not just
abstract parameter mappings. Regularizes against ill-conditioning.

Only use if parameter recovery is poor without it. Discard at inference time.

---

## Full Forward Pass

```python
class FinancialFNO(nn.Module):
    def __init__(self):
        self.lifting = nn.Conv2d(3, 64, 1)
        self.fourier_layers = nn.ModuleList([
            FourierBlock(64, modes1=m1, modes2=m2)
            for m1, m2 in [(12,8), (10,7), (8,6), (6,5)]
        ])
        self.projection = nn.Sequential(
            nn.Conv2d(64, 128, 1), nn.GELU(), nn.Conv2d(128, 64, 1)
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 5))
        self.param_head = nn.Sequential(
            nn.Linear(1280, 256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 64), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(64, 5)
        )

    def forward(self, iv_surface, k_grid, tau_grid):
        # Input augmentation
        batch = iv_surface.shape[0]
        K, T = torch.meshgrid(k_grid, tau_grid, indexing='ij')
        coords = torch.stack([K, T]).unsqueeze(0).expand(batch, -1, -1, -1)
        x = torch.cat([iv_surface, coords.to(iv_surface.device)], dim=1)

        # Lifting: (batch, 3, n_k, n_tau) → (batch, 64, n_k, n_tau)
        x = self.lifting(x)

        # Fourier layers: (batch, 64, n_k, n_tau) → same shape, 4 times
        for layer in self.fourier_layers:
            x = layer(x)

        # Projection: final pointwise nonlinearity
        x = self.projection(x)

        # Pool: (batch, 64, n_k, n_tau) → (batch, 1280)
        x = self.pool(x).flatten(1)

        # Parameter head: (batch, 1280) → (batch, 5)
        params = self.param_head(x)
        return params
```

---

## SpectralConv2d Implementation

```python
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x):
        # x: (batch, in_channels, n_k, n_tau)
        x_ft = torch.fft.rfft2(x)
        # x_ft: (batch, in_channels, n_k, n_tau//2+1), complex

        # Handle variable input sizes (resolution invariance)
        m1 = min(self.modes1, x.size(-2))
        m2 = min(self.modes2, x.size(-1) // 2 + 1)

        out_ft = torch.zeros(
            x.shape[0], self.weights1.shape[1], x.size(-2), x.size(-1)//2+1,
            dtype=torch.cfloat, device=x.device
        )

        # Positive frequencies
        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bixy, ioxy -> boxy",
            x_ft[:, :, :m1, :m2],
            self.weights1[:, :, :m1, :m2]
        )

        # Negative frequencies (only if input is large enough)
        if x.size(-2) > m1:
            out_ft[:, :, -m1:, :m2] = torch.einsum(
                "bixy, ioxy -> boxy",
                x_ft[:, :, -m1:, :m2],
                self.weights2[:, :, :m1, :m2]
            )

        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
```

Note: the `if x.size(-2) > m1` guard handles the case where the input grid
is smaller than the mode count (e.g., 8-point grid with modes1=12). Without
this, negative indexing would overlap with positive and corrupt the output.

---

## Training Configuration

```yaml
model:
  in_channels: 3          # IV + k_coord + tau_coord
  hidden_channels: 64
  n_layers: 4
  layer_modes:             # (strike_modes, maturity_modes) per layer
    - [12, 8]
    - [10, 7]
    - [8, 6]
    - [6, 5]
  pool_size: [4, 5]
  head_dims: [256, 64]
  dropout: 0.1
  use_residual: true
  use_aux_head: false      # Enable if parameter recovery is poor

training:
  batch_size: 64
  epochs: 200
  lr: 1e-3
  weight_decay: 1e-4
  scheduler: onecycle
  grad_clip: 1.0
  early_stopping_patience: 30
  loss: mse                # On normalized parameters
  # loss: weighted_mse     # If ρ, ξ recovery lags: weights [1, 1, 2, 2, 1]

data:
  n_k: 16
  n_tau: 20
  k_range: [-0.5, 0.5]
  tau_range: [0.02, 2.0]
  tau_spacing: geometric
  param_ranges:
    kappa: [0.5, 5.0]
    theta: [0.02, 0.15]
    xi: [0.1, 1.0]
    rho: [-0.95, -0.3]
    v0: [0.01, 0.12]
```

---

## Key Ablations to Run (Phase 2)

1. Input channels: 1 (IV only) vs 3 (IV + coords) — does coordinate augmentation help?
2. Modes: isotropic [12,12] vs anisotropic [12,8] vs [16,6]
3. Mode schedule: constant [12,8] all layers vs progressive reduction
4. Pooling: adaptive (4,5) vs spectral (4,3 modes) vs global avg (1,1)
5. Depth: 2, 4, 6 layers
6. Width: 32, 64, 128 channels
7. Residual: with vs without skip connections
8. Loss: MSE vs weighted MSE (2× on ρ, ξ) vs Huber
9. Output target: (κ,θ,ξ,ρ,v₀) vs (κ,κθ,ξ,ρ,v₀) reparameterization
10. Auxiliary head: with vs without

Priority order: 1, 4, 2, 3 are most likely to make a measurable difference.
Run priority ablations single-seed first, then 3-seed the winners.

---

## Approximate Parameter Count

Lifting: 3 × 64 + 64 = 256
SpectralConv2d (layer 1): 2 × 64 × 64 × 12 × 8 × 2 = 1,572,864  (×2 for real/imag)
  (actually complex params, so: 2 × 64 × 64 × 12 × 8 = 786,432 complex = ~1.57M real)
Pointwise conv per layer: 64 × 64 + 64 = 4,160
× 4 layers total spectral: ~4.7M (varies with progressive reduction)
× 4 layers total pointwise: ~16.6k
Projection: (64×128 + 128) + (128×64 + 64) = 16,576
Pool: 0 (no parameters)
Param head: (1280×256+256) + (256×64+64) + (64×5+5) = 344,645
Total: ~5.1M parameters

This is larger than a typical MLP baseline (~200k–500k). For fair comparison,
scale MLP width to match parameter count OR report results at matched param count
AND matched compute budget.

---

## Resolution Invariance Checklist

Every component must accept variable (n_k, n_tau):
- [x] Lifting: Conv2d kernel=1 — works for any spatial size
- [x] SpectralConv2d: FFT defined for any size, mode clamping handles small grids
- [x] Pointwise conv: Conv2d kernel=1 — works for any spatial size
- [x] GELU: pointwise — works for any spatial size
- [x] Residual: addition — works for any matching size
- [x] Projection: Conv2d kernel=1 — works for any spatial size
- [x] AdaptiveAvgPool2d: designed to accept any spatial size
- [x] Flatten + Linear: fixed size after pooling — resolution-agnostic

The ONLY place where grid size matters is the coordinate channel construction
in the forward method. The k_grid and tau_grid must be passed as arguments
(not hardcoded) so different resolutions can be used at test time.
