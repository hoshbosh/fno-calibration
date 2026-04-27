# Phase 6: Paper Writing

## Context

I am writing a workshop paper (6–8 pages) on amortized Heston calibration via Fourier Neural Operators. All experiments are complete.

**The contribution in one sentence:** We train a Fourier Neural Operator that maps an implied volatility surface directly to Heston stochastic volatility parameters in a single forward pass, achieving resolution invariance that MLP-based surrogates lack.

**Target venues (in order of preference):**
1. ICAIF 2026 (ACM International Conference on AI in Finance) — deadline typically late August
2. NeurIPS 2026 ML4PS workshop (Machine Learning for Physical Sciences) — deadline typically late September
3. NeurIPS 2026 AI4Science workshop — similar deadline

**Key experimental results (fill in actual numbers from completed phases):**
- FNO parameter recovery: κ RMSE = ?, θ RMSE = ?, ξ RMSE = ?, ρ RMSE = ?, v₀ RMSE = ?
- MLP parameter recovery: same metrics for comparison
- Resolution invariance: FNO RMSE across grids (8×10, 16×20, 32×40, 64×80) vs MLP+interpolation
- Speed: FNO ~X ms vs LM ~Y seconds (Z× speedup)
- (If Phase 5 done) Rough Bergomi: FNO ~X ms vs LM+MC ~Y minutes

## Task 1: Paper skeleton in LaTeX

Create `paper/` directory with standard workshop paper structure. Use the ICAIF or NeurIPS workshop template.

```
paper/
├── main.tex
├── references.bib
├── figures/          # symlink or copy from ../figures/
│   ├── resolution_invariance.png    # Phase 3, Figure 1
│   ├── speed_accuracy_pareto.png    # Phase 4
│   ├── scatter_rho.png              # Phase 2
│   └── ...
└── tables/
    ├── comparison_table.tex         # Phase 2
    ├── resolution_table.tex         # Phase 3
    └── speed_benchmark.tex          # Phase 4
```

**Paper outline (6–8 pages):**

### 1. Introduction (~1 page)

**Paragraph 1:** The calibration problem. Stochastic volatility models (Heston, rough Bergomi) require calibration to market-observed implied volatility surfaces. Traditional calibration is iterative optimization with expensive forward model evaluations.

**Paragraph 2:** Existing neural approaches. Horvath et al. (2021) trained MLP surrogates to accelerate the forward evaluation. Limitation: fixed grid, no resolution invariance, still requires optimization loop.

**Paragraph 3:** Our contribution. We frame calibration as an operator learning problem — mapping from the space of IV surfaces to the space of model parameters. Using a Fourier Neural Operator (FNO), we learn this inverse map directly, eliminating the optimization loop. The FNO's spectral parameterization provides resolution invariance: the same trained operator accepts IV surfaces on any strike-maturity grid without retraining.

**Paragraph 4:** Summary of results. On synthetic Heston data, the FNO achieves [X]% median relative parameter error with [Y]× speedup over Levenberg-Marquardt, and generalizes to unseen grid resolutions where MLP baselines fail.

### 2. Background (~1 page)

**2.1 Heston Model.** The Heston stochastic volatility model:
```
dS_t = √V_t S_t dW¹_t
dV_t = κ(θ - V_t)dt + ξ√V_t dW²_t
d⟨W¹, W²⟩_t = ρ dt
```
Parameters: κ (mean reversion), θ (long-run variance), ξ (vol-of-vol), ρ (correlation), v₀ (initial variance). Semi-closed form via characteristic function (Heston 1993).

**2.2 Calibration as Inverse Problem.** Given observed IV surface Σ^mkt(k,τ), find θ* = argmin ||Σ^model(θ; k,τ) - Σ^mkt(k,τ)||². Standard approach: iterative optimization (LM) with forward evaluations. Cost: O(N_iter × C_forward).

**2.3 Fourier Neural Operator.** Brief description of FNO architecture: lifting → spectral convolution layers → projection. Key property: parameterization in Fourier space is discretization-invariant. Cite Li et al. (2021).

### 3. Method (~1.5 pages)

**3.1 Problem Formulation.** We learn the inverse operator G†: F(D) → Θ, where F(D) is the space of IV surfaces on domain D = [k_min, k_max] × [τ_min, τ_max] and Θ ⊂ R⁵ is the Heston parameter space.

**3.2 Architecture.** Describe the FNO calibrator:
- Input: IV surface Σ(k,τ) sampled on an arbitrary grid
- Lifting: pointwise mapping from 1 channel to 64
- 4 Fourier layers with spectral truncation at (12, 12) modes
- Adaptive average pooling to fixed spatial size → parameter head MLP
- Output: θ̂ = (κ̂, θ̂, ξ̂, ρ̂, v̂₀)

Explain why AdaptiveAvgPool2d enables resolution invariance while preserving enough spatial information for parameter recovery.

**3.3 Training.** Synthetic data from Heston model. 100k surfaces, Latin Hypercube sampling over parameter ranges. MSE loss on normalized parameters. OneCycleLR, AdamW.

### 4. Experiments (~2 pages)

**4.1 Parameter Recovery.** Table comparing FNO vs MLP on in-distribution test set. Per-parameter RMSE, relative error. Include scatter plot of predicted vs true ρ (hardest parameter).

**4.2 Resolution Invariance.** The key experiment. Figure showing RMSE vs grid resolution for FNO and MLP+interpolation. Table with full results across all grids.

Discuss: FNO trained on 16×20 and evaluated on 8×10, 32×40, 64×80 WITHOUT retraining. MLP requires either interpolation (degrades accuracy) or retraining (impractical).

**4.3 Calibration Speed.** Speed-accuracy Pareto plot. Table of timing results. Emphasize: FNO calibration is a single forward pass (~0.05ms GPU) vs LM iteration (50–200 forward evaluations × ~1ms each = ~0.1–1s for Heston).

**4.4 Ablations.** Key ablation results in a compact table: modes, depth, width, pooling, loss.

**(If Phase 5 done) 4.5 Extension to Rough Bergomi.** Same experiments on rBergomi. Speed comparison is 10⁵–10⁷× — put this number prominently.

### 5. Related Work (~0.5 pages)

- **Neural calibration:** Horvath-Muguruza-Tomas (2021) — MLP surrogate, fixed grid, forward map + optimization. Bayer-Stemper (2018) — deep calibration of rough vol. Hernandez (2016) — early neural calibration.
- **Neural operators in finance:** Wiedemann-Jacquier-Gonon (ICLR 2025) — GNO for IV smoothing. Different task (smoothing vs calibration), different architecture (GNO vs FNO).
- **Neural operators for PDEs:** Li et al. (2021) FNO, Lu et al. (2021) DeepONet, Kovachki et al. (2023) unified framework.
- **Constrained pricing:** Ackerer-Tagasovska-Vatter (NeurIPS 2020) — soft no-arbitrage constraints. Chataigner et al. (2022) — shape constraints. (Mention as future work direction for our approach.)

### 6. Conclusion (~0.5 pages)

Summarize. Limitations: synthetic data only, Heston-specific, no arbitrage constraints on outputs (parameters guarantee arbitrage-freeness by construction but model misspecification is not addressed). Future work: real market data validation, extension to multi-model calibration, hard-constraint architectures (claim (c) from the original project scope).

## Task 2: References

Populate `references.bib` with all load-bearing citations. Include AT MINIMUM:

```bibtex
% Pricing models
@article{heston1993,
  title={A Closed-Form Solution for Options with Stochastic Volatility},
  author={Heston, Steven L},
  journal={The Review of Financial Studies},
  volume={6}, number={2}, pages={327--343}, year={1993}}

@article{bayer2016rough,
  title={Pricing Under Rough Volatility},
  author={Bayer, Christian and Friz, Peter and Gatheral, Jim},
  journal={Quantitative Finance},
  volume={16}, number={6}, pages={887--904}, year={2016}}

% Neural operators
@inproceedings{li2021fno,
  title={Fourier Neural Operator for Parametric Partial Differential Equations},
  author={Li, Zongyi and Kovachki, Nikola and Azizzadenesheli, Kamyar and Liu, Burigede and Bhattacharya, Kaushik and Stuart, Andrew and Anandkumar, Anima},
  booktitle={International Conference on Learning Representations},
  year={2021}}

@article{lu2021deeponet,
  title={Learning Nonlinear Operators via DeepONet},
  author={Lu, Lu and Jin, Pengzhan and Pang, Guofei and Zhang, Zhongqiang and Karniadakis, George Em},
  journal={Nature Machine Intelligence},
  volume={3}, pages={218--229}, year={2021}}

@article{kovachki2023neural,
  title={Neural Operator: Learning Maps Between Function Spaces},
  author={Kovachki, Nikola and Li, Zongyi and Liu, Burigede and Azizzadenesheli, Kamyar and Bhattacharya, Kaushik and Stuart, Andrew and Anandkumar, Anima},
  journal={Journal of Machine Learning Research},
  volume={24}, number={89}, year={2023}}

% Neural calibration
@article{horvath2021deep,
  title={Deep Learning Volatility},
  author={Horvath, Blanka and Muguruza, Aitor and Tomas, Mehdi},
  journal={Quantitative Finance},
  volume={21}, number={1}, pages={11--27}, year={2021}}

% Neural operators in finance
@inproceedings{wiedemann2025operator,
  title={Operator Deep Smoothing for Implied Volatility},
  author={Wiedemann, Alexander and Jacquier, Antoine and Gonon, Lukas},
  booktitle={International Conference on Learning Representations},
  year={2025}}

% Simulation-based inference
@article{cranmer2020frontier,
  title={The Frontier of Simulation-Based Inference},
  author={Cranmer, Kyle and Brehmer, Johann and Louppe, Gilles},
  journal={Proceedings of the National Academy of Sciences},
  volume={117}, number={48}, pages={30055--30062}, year={2020}}

% Vol surface theory
@article{gatheral2014ssvi,
  title={Arbitrage-Free {SVI} Volatility Surfaces},
  author={Gatheral, Jim and Jacquier, Antoine},
  journal={Quantitative Finance},
  volume={14}, number={1}, pages={59--71}, year={2014}}

@article{carr2005sufficient,
  title={A Note on Sufficient Conditions for No Arbitrage},
  author={Carr, Peter and Madan, Dilip B},
  journal={Finance Research Letters},
  volume={2}, number={3}, pages={125--130}, year={2005}}
```

## Task 3: Figure preparation

All figures should be publication-quality:
- Font size ≥ 8pt in all labels and legends
- Vector format (PDF) preferred over raster (PNG)
- Consistent color scheme across all figures (use a colorblind-friendly palette)
- No titles (captions go in the LaTeX figure environment)
- Save to `paper/figures/`

**Key figures to prepare:**

1. **Architecture diagram** — create a clean schematic of the FNO calibrator: IV surface → lifting → Fourier layers → pooling → parameter head → θ. Use TikZ or a drawing tool.

2. **Resolution invariance plot** (from Phase 3) — the paper's most important figure.

3. **Speed-accuracy Pareto** (from Phase 4) — clean up axes, add method labels.

4. **Scatter plots** — predicted vs true for ρ and ξ (the hardest parameters), with identity line.

5. **Sample surfaces** — 2–3 example IV surfaces (heatmaps) showing typical Heston smiles.

## Task 4: Compile and proofread

```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Check:
- All figures render correctly
- All references resolve
- Page count is within workshop limits (6–8 pages)
- No orphaned citations or undefined references
- Abstract is ≤ 150 words
- The contribution is stated clearly in both abstract and introduction

## Task 5: GitHub repo cleanup

Prepare the public repo for submission:

```
README.md                  # Project overview, how to reproduce
requirements.txt           # Pinned versions
configs/                   # All config files used in the paper
data/generate_heston.py    # Data generation (don't include the data files — too large)
models/
train.py
eval.py
scripts/
  run_all_experiments.sh   # Reproduces all paper results
  plot_all_figures.sh      # Regenerates all figures
results/                   # CSV files with all numerical results
figures/                   # All figures
paper/                     # LaTeX source
LICENSE                    # MIT
.gitignore                 # Exclude: data/*.h5, checkpoints/*.pt, wandb/
```

Add a `REPRODUCE.md` with step-by-step instructions:
```markdown
# Reproducing results

## 1. Generate data
python data/generate_heston.py --n_samples 100000 --output data/heston_train_100k.h5
# (repeat for val, test, ood sets)

## 2. Train models
python train.py --config configs/fno_primary.yaml --seed 42
python train.py --config configs/mlp_baseline.yaml --seed 42

## 3. Run experiments
bash scripts/run_all_experiments.sh

## 4. Generate figures
bash scripts/plot_all_figures.sh

## Hardware
Experiments were run on [your setup]. Training takes ~X hours on a single A100.
Total compute cost: ~$Y.
```

## What success looks like at the end of Phase 6

- A compiled PDF that is submission-ready
- All figures are publication quality
- The contribution is clearly stated and differentiated from Horvath et al. and Wiedemann et al.
- The limitations section is honest (synthetic data, single model family)
- The GitHub repo is clean and reproducible
- You have identified the specific submission deadline and venue formatting requirements

The paper should be writable in 40–60 hours if all experimental results are clean. If results need touching up (a figure needs regenerating, an ablation is missing), budget an extra 10–20 hours for final experiments.
