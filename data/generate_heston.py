"""
Generates synthetic heston iv surfaces
"""

import argparse
import json
import os
from typing import Any, Sequence
import numpy as np
from numpy.typing import NDArray
import h5py
import yaml
import QuantLib as ql
from py_vollib_vectorized import vectorized_implied_volatility
from scipy.stats.qmc import LatinHypercube

from data.arbitrage import audit_dataset, is_arbitrage_free

def prices_to_iv(prices: NDArray[np.floating], s0: float, r: float, q: float,
                 strikes: Sequence[float], taus: Sequence[float],
                 flags: Sequence[str]) -> NDArray[np.floating]:
    '''
    Convert prices to implied volatility
    - Uses a Black-Scholes inversion
    '''
    n_k, n_tau = prices.shape
    K_grid, T_grid = np.meshgrid(strikes, taus, indexing="ij")
    flag_grid = np.broadcast_to(np.asarray(flags)[:, None], (n_k, n_tau))
    iv = vectorized_implied_volatility(
        prices.flatten(), s0, K_grid.flatten(), T_grid.flatten(), r,
        flag_grid.flatten(),
        q=q, model="black_scholes_merton", return_as="numpy",
    )
    return iv.reshape(n_k, n_tau)


def heston_option_prices(s0: float, r: float, q: float,
                         kappa: float, theta: float, xi: float,
                         rho: float, v0: float,
                         strikes: Sequence[float], taus: Sequence[float],
                         flags: Sequence[str]) -> NDArray[np.float64]:
    """Price European options on a (K, tau) grid. flags: 'c' or 'p' per strike."""
    today = ql.Date(1, 1, 2025)
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()

    spot = ql.QuoteHandle(ql.SimpleQuote(s0))
    # Yield structure risk free rate and divident yield respectively
    rTS = ql.YieldTermStructureHandle(ql.FlatForward(today, r, day_count))
    qTS = ql.YieldTermStructureHandle(ql.FlatForward(today, q, day_count))

    process = ql.HestonProcess(rTS, qTS, spot, v0, kappa, theta, xi, rho)
    model = ql.HestonModel(process)
    engine = ql.AnalyticHestonEngine(model)

    prices = np.zeros((len(strikes), len(taus)), dtype=np.float64)
    for j, tau in enumerate(taus):

        # Converts the maturity float tau into a date
        maturity = today + int(round(tau * 365))

        # When one can earn 
        exercise = ql.EuropeanExercise(maturity)
        for i, K in enumerate(strikes):
            opt_type = ql.Option.Call if flags[i] == "c" else ql.Option.Put

            # What one can jarn
            payoff = ql.PlainVanillaPayoff(opt_type, float(K))

            # In the 3D array, the option object lives
            option = ql.VanillaOption(payoff, exercise)
            option.setPricingEngine(engine)

            # Gracefully handle engine failures, populate with NaN
            try:
                prices[i, j] = option.NPV()
            except RuntimeError:
                prices[i, j] = np.nan
    return prices


def lhs_params(cfg: dict[str, Any], n: int, rng: np.random.Generator) -> NDArray[np.float64]:
    """
    Generate samples using Latin Hypercube sampling
    """
    h = cfg["heston"]

    sampler = LatinHypercube(d=5, rng=rng)

    u = sampler.random(n=n)

    lo = np.array([h["kappa"][0], h["theta"][0],h["xi"][0],h["rho"][0],h["v0"][0]])
    hi = np.array([h["kappa"][1], h["theta"][1],h["xi"][1],h["rho"][1],h["v0"][1]])

    return lo + u * (hi - lo)

def build_grid(cfg: dict[str, Any]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    g = cfg["grid"]

    # Log-moneyness is denser near ATM where the smile curve is the sharpest
    # Split the 16 points into 4 deep OTM puts, 8 near-ATM | 4 deep OTM calls
    # Slices at the ends are to prevent shared endpoints from appearing twice
    k = np.concatenate([
        np.linspace(g["k_min"], -g["k_atm"], g["n_k_wing"] + 1)[:-1],
        np.linspace(-g["k_atm"], g["k_atm"], g["n_k_atm"]),
        np.linspace(g["k_atm"], g["k_max"], g["n_k_wing"] + 1)[1:],
        ])
    tau = np.geomspace(g["tau_min"], g["tau_max"], g["n_tau"])
    return k, tau


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--n_samples", type=int, default=None)
    ap.add_argument("--output", default=None)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    n = args.n_samples or cfg["data"]["n_samples"]
    out = args.output or cfg["data"]["path"]
    seed = args.seed if args.seed is not None else cfg["data"]["seed"]
    rng = np.random.default_rng(seed)

    s0 = cfg["heston"]["s0"]
    r = cfg["heston"]["r"]
    q = cfg["heston"]["q"]

    k_grid, tau_grid = build_grid(cfg)
    flags = np.where(k_grid >= 0, "c", "p")  # OTM: call for k>=0, put for k<0

    surfaces = np.zeros((n, len(k_grid), len(tau_grid)), dtype=np.float32)
    params = np.zeros((n, 5), dtype=np.float32)

    accepted = 0
    attempts = 0
    feller_violations = 0  # flag-don't-drop: count parameter draws with 2*kappa*theta <= xi^2

    # LHS is a *batch* design — stratification is a property of the whole pool,
    # not individual draws. Pre-generate a pool sized for expected rejection rate;
    # refill if exhausted (each refill is its own LHS, locally stratified).
    BUFFER_FACTOR = 2
    pool = lhs_params(cfg, n * BUFFER_FACTOR, rng)
    pool_idx = 0

    # Generate a surface, if it is invalid via NaNs or arbitrage, reject the surface
    while accepted < n:
        if pool_idx >= len(pool):
            print(f" LHS pool exhausted at {accepted}/{n}; drawing fresh batch.")
            pool = lhs_params(cfg, n * BUFFER_FACTOR, rng)
            pool_idx = 0

        p = pool[pool_idx]
        pool_idx += 1
        attempts += 1

        kappa, theta, xi, rho, v0 = p
        # Feller condition: 2*kappa*theta > xi^2 ensures the variance process
        # stays strictly positive. Heston is still valid when violated (variance
        # just touches zero occasionally), so we keep these samples and only count.
        if 2 * kappa * theta <= xi ** 2:
            feller_violations += 1

        iv_surface = np.full((len(k_grid), len(tau_grid)), np.nan)
        ok = True

        for j, tau in enumerate(tau_grid):
            F = s0 * np.exp((r - q) * tau)
            strikes = F * np.exp(k_grid)
            prices = heston_option_prices(s0, r, q, kappa, theta, xi, rho, v0,
                                          strikes, [tau], flags)[:, 0]

            if np.any(~np.isfinite(prices)):
                ok = False
                break
            # Floor underflow / signed-zero noise; deep OTM truly has price < fp64 precision.
            prices = np.maximum(prices, 1e-12)

            iv_col = prices_to_iv(prices[:, None], s0, r, q, strikes, [tau], flags)[:, 0]
            if np.any(~np.isfinite(iv_col)) or np.any(iv_col <= 0):
                ok = False
                break
            iv_surface[:, j] = iv_col

        if not ok or not is_arbitrage_free(iv_surface, tau_grid, k_grid):
            continue

        surfaces[accepted] = iv_surface.astype(np.float32)
        params[accepted] = p.astype(np.float32)
        accepted += 1
        if accepted % 50 == 0:
            print(f" accepted {accepted}/{n} rejection rate {1 - accepted/attempts:.2%}")

    # Write to hdf5 file
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with h5py.File(out, "w") as f:
        f.create_dataset("surfaces", data=surfaces)
        f.create_dataset("parameters", data=params)
        f.create_dataset("grid_k", data=k_grid.astype(np.float32))
        f.create_dataset("grid_tau", data=tau_grid.astype(np.float32))
        f.attrs["param_names"] = ["kappa", "theta", "xi", "rho", "v0"]
        f.attrs["s0"] = s0
        f.attrs["r"] = r
        f.attrs["q"] = q

    # Post-generation audit. Should be ~0 for a Heston dataset since the
    # per-surface gate already rejected violations during generation. Non-zero
    # counts indicate numerical noise or a generator regression.
    audit = audit_dataset(surfaces, k_grid, tau_grid)
    audit_path = out.replace(".h5", ".audit.json")
    with open(audit_path, "w") as f:
        json.dump(audit, f, indent=2)

    print(f"wrote {n} surfaces to {out} ({attempts} attempts, "
          f"{1 - n/attempts:.2%} rejected, "
          f"{feller_violations}/{attempts} Feller violations)")
    print(f"audit: {audit['negative_iv']} negative-IV, "
          f"{audit['calendar_violations']} calendar, "
          f"{audit['butterfly_violations']} butterfly "
          f"(of {audit['n_total']}); written to {audit_path}")


if __name__ == "__main__":
    main()
