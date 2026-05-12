from __future__ import annotations
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import yaml
import wandb
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader

from data.dataset import ParamNormalizer, load_datasets
from models.mlp_baseline import build_mlp
from data.generate_heston import build_grid
from models.fno import build_fno

PARAM_NAMES = ["kappa", "theta", "xi", "rho", "v0"]

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", choices=["mlp", "fno"], required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None,
                    help="Overrides cfg['data']['seed']. Used for torch/numpy seeding "
                         "and to namespace the checkpoint + W&B run name.")
    ap.add_argument("--wandb_mode",  default="offline", choices=["online", "offline", "disabled"])
    ap.add_argument("--out_dir", default="checkpoints")

    return ap.parse_args()

# Check back on this
def setup_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

def make_loaders(cfg: dict) -> tuple[DataLoader, DataLoader, ParamNormalizer]:
    train_ds, val_ds, _test_ds, _ood_ds, normalizer = load_datasets(
        train_path=cfg["data"]["train_path"],
        val_path=cfg["data"]["val_path"],
        test_path=cfg["data"].get("test_path"),
        ood_path=cfg["data"].get("ood_path"),
    )
    bs = cfg["train"]["batch_size"]
    nw = cfg["train"]["num_workers"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)

    return train_loader, val_loader, normalizer

# FNO wrapper that associates a grid to the IV channel 
class FNOWithGrid(nn.Module):
    def __init__(self, fno: nn.Module, k_grid: np.ndarray, tau_grid: np.ndarray) -> None:
        super().__init__()
        self.fno = fno
        # We use buffers here because buffers get sent to the device with the tensor for free
        self.register_buffer("k_grid", torch.as_tensor(k_grid, dtype=torch.float32))
        self.register_buffer("tau_grid", torch.as_tensor(tau_grid, dtype=torch.float32))

    def forward(self, surface: torch.Tensor) -> torch.Tensor:
        # The actual concating
        return self.fno(surface, self.k_grid, self.tau_grid)
            

def build_model(cfg: dict, name: str) -> nn.Module:
    if name == "mlp":
        return build_mlp(cfg)
    if name == "fno":
        fno = build_fno(cfg)
        k_grid, tau_grid = build_grid(cfg)
        return FNOWithGrid(fno, k_grid, tau_grid)
    raise ValueError("Unknown model given")

def train_one_epoch(model: nn.Module, loader: DataLoader,
                    loss_fn: nn.Module, optim: torch.optim.Optimizer,
                    device: str, scaler: torch.amp.GradScaler,
                    scheduler: torch.optim.lr_scheduler.OneCycleLR,
                    grad_clip: float, epoch: int,
                    log_to_wandb: bool) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_grad_norm = 0.0
    n_batches = 0
    use_amp = (device == "cuda")
    for surface, target_z in loader:
        surface = surface.to(device, non_blocking=True)
        target_z = target_z.to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            pred_z = model(surface)
            loss = loss_fn(pred_z, target_z)

        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optim)
        scaler.update()
        scheduler.step()

        loss_val = loss.item()
        gn_val = float(grad_norm)
        total_loss += loss_val
        total_grad_norm += gn_val
        n_batches += 1

        if log_to_wandb:
            wandb.log({
                "train/batch_loss": loss_val,
                "train/grad_norm": gn_val,
                "train/lr": scheduler.get_last_lr()[0],
                "epoch": epoch,
            })

    return total_loss / max(n_batches, 1), total_grad_norm / max(n_batches, 1)

@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module,
             device: str, normalizer: ParamNormalizer
             ) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (mean_loss, per_param_rmse_raw, per_param_rel_median, per_param_rel_p95)."""
    model.eval()
    total_loss = 0.0
    n_batches = 0
    use_amp = (device == "cuda")
    preds_z_all: list[torch.Tensor] = []
    targets_z_all: list[torch.Tensor] = []
    for surface, target_z in loader:
        surface = surface.to(device, non_blocking=True)
        target_z = target_z.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            pred_z = model(surface)
            loss = loss_fn(pred_z, target_z)

        total_loss += loss.item()
        n_batches += 1
        preds_z_all.append(pred_z.float().cpu())
        targets_z_all.append(target_z.float().cpu())

    mean_loss = total_loss / max(n_batches, 1)
    preds_z = torch.cat(preds_z_all).numpy()
    targets_z = torch.cat(targets_z_all).numpy()

    # Decode to original units before computing per-param metrics — RMSE in z-space
    # is uninformative because each param has a different scale.
    preds_raw = normalizer.decode(preds_z)
    targets_raw = normalizer.decode(targets_z)

    rmse_raw = np.sqrt(((preds_raw - targets_raw) ** 2).mean(axis=0))
    rel = np.abs(preds_raw - targets_raw) / np.maximum(np.abs(targets_raw), 1e-8)
    rel_median = np.median(rel, axis=0)
    rel_p95 = np.percentile(rel, 95, axis=0)
    return mean_loss, rmse_raw, rel_median, rel_p95

def save_checkpoint(path: str, model: nn.Module, model_name: str, cfg: dict,
                    normalizer: ParamNormalizer, epoch: int, val_loss: float) -> None:
    torch.save({
        "model_state": model.state_dict(),
        "model_name": model_name,
        "cfg": cfg,
        "normalizer_mean": normalizer.mean,
        "normalizer_std": normalizer.std,
        "epoch": epoch,
        "val_loss": val_loss,
        }, path)

def format_rmse_line(rmse_raw: np.ndarray) -> str:
    return " ".join(f"{name} = {v:.3f}" for name, v in zip(PARAM_NAMES, rmse_raw))

def main() -> None:
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    epochs = args.epochs if args.epochs is not None else cfg["train"]["epochs"]
    seed = args.seed if args.seed is not None else cfg["data"]["seed"]

    setup_seeds(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | model: {args.model} | seed: {seed} | epochs: {epochs}")

    train_loader, val_loader, normalizer = make_loaders(cfg)
    print(f"train batches: {len(train_loader)} | val batches {len(val_loader)}")

    model = build_model(cfg, args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{args.model} params: {n_params:,}")

    optim = AdamW(model.parameters(),
                  lr=cfg["train"]["lr"],
                  weight_decay=cfg["train"]["weight_decay"])
    # Schedule for setting the learning rate
    scheduler = OneCycleLR(
            optim,
            max_lr=cfg["train"]["lr"],
            epochs=epochs,
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy="cos",
            )
    loss_fn = nn.MSELoss()

    use_amp = (device=="cuda")
    # bf16 has the same exponent range as fp32, so gradient scaling isn't
    # needed (no underflow). Scaler kept in the call sites as a no-op for
    # API compatibility with the fp16 path.
    scaler = torch.amp.GradScaler("cuda", enabled=False)

    run_name = f"{args.model}-seed{seed}-{time.strftime('%Y%m%d-%H%M%S')}"
    wandb.init(
        project=cfg["wandb"]["project"],
        entity=cfg["wandb"]["entity"],
        mode=args.wandb_mode,
        config={**cfg, "model_arch": args.model, "n_params": n_params,
                "seed": seed, "device": device, "epochs_planned": epochs},
        name=run_name,
    )
    log_to_wandb = (args.wandb_mode != "disabled")

    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"{args.model}_seed{seed}_best.pt")
    best_val = float("inf")
    # Patience is used for stopping the training if validation loss does not improve, stopping early to prevent overfitting
    epochs_since_best = 0
    patience = cfg["train"]["early_stopping_patience"]

    grad_clip = cfg["train"]["grad_clip"]

    for epoch in range(epochs):
        t0 = time.time()
        train_loss, mean_grad_norm = train_one_epoch(
            model, train_loader, loss_fn, optim, device, scaler, scheduler,
            grad_clip, epoch, log_to_wandb,
        )
        val_loss, rmse_raw, rel_median, rel_p95 = validate(
            model, val_loader, loss_fn, device, normalizer,
        )
        epoch_time = time.time() - t0

        log = {
            "epoch": epoch,
            "train/epoch_loss": train_loss,
            "train/epoch_grad_norm": mean_grad_norm,
            "train/epoch_time_s": epoch_time,
            "val/loss": val_loss,
            "val/lr_at_epoch_end": scheduler.get_last_lr()[0],
        }
        for name, r, m, p in zip(PARAM_NAMES, rmse_raw, rel_median, rel_p95):
            log[f"val/rmse/{name}"] = float(r)
            log[f"val/rel_median/{name}"] = float(m)
            log[f"val/rel_p95/{name}"] = float(p)
        if log_to_wandb:
            wandb.log(log)

        print(f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} "
              f"| lr {scheduler.get_last_lr()[0]:.2e} | gn {mean_grad_norm:.2f} "
              f"| {epoch_time:.1f}s | rmse {format_rmse_line(rmse_raw)}")

        if val_loss < best_val:
            best_val = val_loss
            epochs_since_best = 0
            save_checkpoint(ckpt_path, model, args.model, cfg, normalizer, epoch, val_loss)
        else:
            epochs_since_best += 1

        if epochs_since_best >= patience:
            print(f"early stopping at epoch {epoch}")
            break

    print(f"best val loss: {best_val:.4f} | checkpoint: {ckpt_path}")
    if log_to_wandb:
        wandb.summary["best_val_loss"] = best_val
        wandb.finish()
                                                                                                                         
                  
if __name__ == "__main__":
    main()
