from __future__ import annotations
import argparse
import json
import os
import re
import numpy as np
import torch
import torch.nn as nn
import yaml
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from data.dataset import HestonSurfaceDataset, ParamNormalizer, load_datasets
from data.generate_heston import heston_option_prices, prices_to_iv, build_grid
from train import build_model

# Global constant, notated by the snake case spelling
PARAM_NAMES: list[str] = ["kappa", "theta", "xi", "rho", "v0"]

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--split", choices=["test", "ood"], default="test",
                    help="Which evaluation split to run against.")
    ap.add_argument("--results_dir", default="results",
                    help="Where to write the per-eval JSON + npz outputs.")
    ap.add_argument("--skip_iv", action="store_true",
                    help="Skip the slow QuantLib IV-reconstruction loop (useful "
                         "for quick metric-only sweeps during ablation hunting).")
    return ap.parse_args()


def parse_seed_from_path(path: str) -> int | None:
    """Extract seed N from a filename like 'fno_seed42_best.pt'."""
    m = re.search(r"seed(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else None

def load_from_checkpoint(path: str, cfg:dict, device:str) -> tuple[nn.Module, str, ParamNormalizer]:
    # weights_only=False because the checkpoint pickles numpy arrays
    # (normalizer mean/std). torch>=2.6 defaults to True and would refuse.
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model(cfg, ckpt["model_name"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    # Reuse the train-time normalizer (not recomputed from this split) so
    # denorm matches exactly what the model was trained against.
    normalizer = ParamNormalizer(
            mean=ckpt["normalizer_mean"],
            std=ckpt["normalizer_std"],
            )
    return model, ckpt["model_name"], normalizer

@torch.no_grad()
def predict_all(
        model: nn.Module,
        dataset: HestonSurfaceDataset,
        device: str,
        batch_size: int
        ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    preds_z: list[torch.Tensor] = []
    targets_z: list[torch.Tensor] = []
    for surface, target_z in loader:
        surface = surface.to(device, non_blocking=True)
        preds_z.append(model(surface).cpu())
        targets_z.append(target_z)
    return (
            torch.cat(preds_z).numpy().astype(np.float32),
            torch.cat(targets_z).numpy().astype(np.float32)
        )

def per_param_metrics(
        pred_raw: NDArray[np.floating],                                                                      
        true_raw: NDArray[np.floating]
        ) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    # axis=0 reduces over the sample dim, leaving one metric per Heston param.
    rmse = np.sqrt(((pred_raw - true_raw) ** 2).mean(axis=0))
    rel = np.abs(pred_raw - true_raw) / np.abs(true_raw)
    return rmse, np.median(rel, axis=0), np.percentile(rel, 95, axis=0)

def iv_reconstruction_rmse(                                                                              
    pred_raw: NDArray[np.floating],
    true_surfaces: NDArray[np.floating],                                                                 
    k_grid: NDArray[np.floating],
    tau_grid: NDArray[np.floating],                                                                      
    cfg: dict,                                                                                           
    ) -> NDArray[np.float32]:
    """For each test sample: re-price with predicted params, invert to IV, compare."""                   
    h = cfg["heston"]
    s0, r, q = h["s0"], h["r"], h["q"]
    # OTM convention from the generator: call wing for k>=0, put wing for k<0.
    # Pricing OTM avoids deep-ITM intrinsic-value dominance that swamps IV inversion.
    flags = np.where(k_grid >= 0, "c", "p")
    n_k, n_tau = len(k_grid), len(tau_grid)

    # Clip predicted params to training ranges before re-pricing. Models with
    # unbounded regression heads can overshoot (e.g., rho < -1), which QuantLib's
    # HestonModel rejects. Phase 0 expedient — proper fix is sigmoid output
    # scaling at the model head.
    lo = np.array([h["kappa"][0], h["theta"][0], h["xi"][0], h["rho"][0], h["v0"][0]])
    hi = np.array([h["kappa"][1], h["theta"][1], h["xi"][1], h["rho"][1], h["v0"][1]])
    pred_clipped = np.clip(pred_raw, lo, hi)
    n_clipped = int((pred_raw != pred_clipped).any(axis=1).sum())
    if n_clipped:
        print(f"[iv_reconstruction] {n_clipped}/{len(pred_raw)} predictions clipped to training range.")

    rmse = np.empty(len(pred_raw), dtype=np.float32)
    for i, (kappa, theta, xi, rho, v0) in enumerate(pred_clipped):
        iv_pred = np.full((n_k, n_tau), np.nan, dtype=np.float64)

        # Loop over maturities because the strike vector depends on tau:
        # K = F(tau) * exp(k), where F(tau) = s0 * exp((r-q)*tau) is the forward.
        # We can't precompute one strike grid for the whole surface.
        for j, tau in enumerate(tau_grid):
            F = s0 * np.exp((r - q) * tau)
            strikes = F * np.exp(k_grid)
            try:
                prices = heston_option_prices(
                    s0, r, q,
                    float(kappa), float(theta), float(xi), float(rho), float(v0),
                    strikes, [tau], flags,
                )[:, 0]
            except RuntimeError:
                # QuantLib rejected the parameter combination even after clipping
                # (rare boundary cases). Skip column; nanmean handles it.
                continue
            # If pricing fails for this maturity (engine NaN), skip the column.
            # Unlike the generator, eval doesn't reject — bad columns become NaN
            # and are excluded from the per-surface RMSE by nanmean.
            if np.any(~np.isfinite(prices)):
                continue
            # Floor underflow at deep OTM before BS inversion (mirrors generator).
            prices = np.maximum(prices, 1e-12)
            # prices_to_iv expects a 2D (n_k, n_tau) shape; we have one column.
            iv_col = prices_to_iv(
                prices[:, None], s0, r, q, strikes, [tau], flags,
            )[:, 0]
            iv_pred[:, j] = iv_col

        rmse[i] = float(np.sqrt(np.nanmean((iv_pred - true_surfaces[i]) ** 2)))
    return rmse

def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg: dict = yaml.safe_load(f)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
                                                                                                       
    # Dedicated test/OOD files — no fractional split, no leakage by construction.
    # Normalizer is fit on train (or loaded from sidecar) and shared across splits.
    _train_ds, _val_ds, test_ds, ood_ds, _normalizer = load_datasets(
        train_path=cfg["data"]["train_path"],
        val_path=cfg["data"].get("val_path"),
        test_path=cfg["data"].get("test_path"),
        ood_path=cfg["data"].get("ood_path"),
    )
    eval_ds = ood_ds if args.split == "ood" else test_ds
    if eval_ds is None:
        raise ValueError(f"Config has no {args.split}_path set.")

    model, model_name, normalizer = load_from_checkpoint(                                                
        args.checkpoint, cfg, device
    )

    preds_z, targets_z = predict_all(model, eval_ds, device, args.batch_size)
    preds_raw: NDArray[np.floating] = normalizer.decode(preds_z)
    targets_raw: NDArray[np.floating] = normalizer.decode(targets_z)

    rmse, rel_med, rel_p95 = per_param_metrics(preds_raw, targets_raw)

    iv_rmse: NDArray[np.floating] | None = None
    if not args.skip_iv:
        k_grid, tau_grid = build_grid(cfg)
        iv_rmse = iv_reconstruction_rmse(
            preds_raw, eval_ds.surfaces, k_grid, tau_grid, cfg
        )

    seed = parse_seed_from_path(args.checkpoint)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n=== {model_name} | seed={seed} | {args.split} | {len(eval_ds)} surfaces ===")
    print(f"{'param':<8} {'rmse':>10} {'rel_med':>10} {'rel_p95':>10}")
    for name, r, m, p in zip(PARAM_NAMES, rmse, rel_med, rel_p95):
        print(f"{name:<8} {r:>10.4f} {m:>10.2%} {p:>10.2%}")
    if iv_rmse is not None:
        print(
            f"\nIV reconstruction RMSE (vol points): "
            f"median={np.median(iv_rmse):.4f}  "
            f"mean={float(np.mean(iv_rmse)):.4f}  "
            f"p95={np.percentile(iv_rmse, 95):.4f}"
        )

    # Persist results so Task 5/6 (comparison table, scatter plots, worst-case
    # heatmaps) can be regenerated without re-running the slow IV loop.
    os.makedirs(args.results_dir, exist_ok=True)
    seed_tag = f"seed{seed}" if seed is not None else "seedNA"
    stem = f"{model_name}_{seed_tag}_{args.split}"

    summary = {
        "model": model_name,
        "seed": seed,
        "split": args.split,
        "checkpoint": os.path.abspath(args.checkpoint),
        "n_eval_samples": int(len(eval_ds)),
        "n_params": int(n_params),
        "per_param": {
            name: {
                "rmse": float(r),
                "rel_median": float(m),
                "rel_p95": float(p),
            }
            for name, r, m, p in zip(PARAM_NAMES, rmse, rel_med, rel_p95)
        },
        "iv_rmse": (
            None if iv_rmse is None else {
                "median": float(np.median(iv_rmse)),
                "mean": float(np.mean(iv_rmse)),
                "p95": float(np.percentile(iv_rmse, 95)),
            }
        ),
    }
    json_path = os.path.join(args.results_dir, f"{stem}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Raw arrays for downstream plotting (scatter, worst-case heatmaps).
    npz_path = os.path.join(args.results_dir, f"{stem}.npz")
    np.savez_compressed(
        npz_path,
        preds_raw=preds_raw.astype(np.float32),
        targets_raw=targets_raw.astype(np.float32),
        iv_rmse=(np.array([], dtype=np.float32) if iv_rmse is None
                 else iv_rmse.astype(np.float32)),
    )
    print(f"\nWrote {json_path} and {npz_path}")

if __name__ == "__main__":
    main()
