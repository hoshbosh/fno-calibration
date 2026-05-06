from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.nn as nn
import yaml
from numpy.typing import NDArray
from torch.utils.data import DataLoader

from data.dataset import HestonSurfaceDataset, ParamNormalizer, split_datasets
from data.generate_heston import heston_option_prices, prices_to_iv, build_grid
from train import build_model

# Global constant, notated by the snake case spelling
PARAM_NAMES: list[str] = ["kappa", "theta", "xi", "rho", "v0"]

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    return ap.parse_args()

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

    rmse = np.empty(len(pred_raw), dtype=np.float32)
    for i, (kappa, theta, xi, rho, v0) in enumerate(pred_raw):
        iv_pred = np.full((n_k, n_tau), np.nan, dtype=np.float64)

        # Loop over maturities because the strike vector depends on tau:
        # K = F(tau) * exp(k), where F(tau) = s0 * exp((r-q)*tau) is the forward.
        # We can't precompute one strike grid for the whole surface.
        for j, tau in enumerate(tau_grid):
            F = s0 * np.exp((r - q) * tau)
            strikes = F * np.exp(k_grid)
            prices = heston_option_prices(
                s0, r, q,
                float(kappa), float(theta), float(xi), float(rho), float(v0),
                strikes, [tau], flags,
            )[:, 0]
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
                                                                                                       
    # Same seed + fractions as train.py → identical test split, no leakage.
    _, _, test_ds, _ = split_datasets(
        h5_path=cfg["data"]["path"],                                                                     
        train_frac=cfg["data"]["train_frac"],                                                            
        val_frac=cfg["data"]["val_frac"],
        seed=cfg["data"]["seed"],                                                                        
    )                                                                                                    

    model, model_name, normalizer = load_from_checkpoint(                                                
        args.checkpoint, cfg, device
    )

    preds_z, targets_z = predict_all(model, test_ds, device, args.batch_size)                            
    preds_raw: NDArray[np.floating] = normalizer.decode(preds_z)
    targets_raw: NDArray[np.floating] = normalizer.decode(targets_z)                                     
                                                                                                       
    rmse, rel_med, rel_p95 = per_param_metrics(preds_raw, targets_raw)
                                                                                                       
    k_grid, tau_grid = build_grid(cfg)                                                                   
    iv_rmse = iv_reconstruction_rmse(
        preds_raw, test_ds.surfaces, k_grid, tau_grid, cfg                                               
    )                                                                                                    

    print(f"\n=== {model_name} | {len(test_ds)} test surfaces ===")                                      
    print(f"{'param':<8} {'rmse':>10} {'rel_med':>10} {'rel_p95':>10}")
    for name, r, m, p in zip(PARAM_NAMES, rmse, rel_med, rel_p95):                                       
        print(f"{name:<8} {r:>10.4f} {m:>10.2%} {p:>10.2%}")                                             
    print(                                                                                               
        f"\nIV reconstruction RMSE (vol points): "                                                       
        f"median={np.median(iv_rmse):.4f}  p95={np.percentile(iv_rmse, 95):.4f}"                         
    )

if __name__ == "__main__":
    main()
