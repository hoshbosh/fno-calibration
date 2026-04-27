from __future__ import annotations
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from data.dataset import ParamNormalizer, split_datasets
from models.mlp_baseline import build_mlp

PARAM_NAMES = ["kappa", "theta", "xi", "rho", "v0"]

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--model", choices=["mlp", "fno"], required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--wandb_mode",  default="offline", choices=["online", "offline", "disabled"])
    ap.add_argument("--out_dir", default="checkpoints")

    return ap.parse_args()

# Check back on this
def setup_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

def make_loaders(cfg: dict) -> tuple[DataLoader, DataLoader, ParamNormalizer]:
    train_ds, val_ds, _test_ds, normalizer = split_datasets(
        h5_path=cfg["data"]["path"],
        train_frac=cfg["data"]["train_frac"],
        val_frac=cfg["data"]["val_frac"],
        seed=cfg["data"]["seed"]
    )
    bs = cfg["train"]["batch_size"]
    nw = cfg["train"]["num_workers"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=nw)

    return train_loader, val_loader, normalizer

def build_model(cfg: dict, name: str) -> nn.Module:
    if name == "mlp":
        return build_mlp(cfg)
    if name == "fno":
        pass
        # return build_fno(cfg)
    raise ValueError("Unknown model given")

def train_one_epoch(model: nn.Module, loader: DataLoader,
                    loss_fn: nn.Module, optim: torch.optim.Optimizer,
                    device: str) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    for surface, target_z in loader:
        surface = surface.to(device, non_blocking=True)
        target_z = target_z.to(device, non_blocking=True)

        optim.zero_grad()
        pred_z = model(surface)
        loss = loss_fn(pred_z, target_z)
        loss.backward()
        optim.step()

        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)

@torch.no_grad()
def validate(model: nn.Module, loader: DataLoader, loss_fn: nn.Module,
             device: str, normalizer: ParamNormalizer) -> tuple[float, np.ndarray]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    sq_err_sum = torch.zeros(5, device = device)
    n_seen = 0
    for surface, target_z in loader:
        surface = surface.to(device, non_blocking=True)
        target_z = target_z.to(device, non_blocking=True)

        pred_z = model(surface)
        loss = loss_fn(pred_z, target_z)

        total_loss += loss.item()
        n_batches += 1

        sq_err_sum += ((pred_z - target_z) ** 2).sum(dim=0)
        n_seen += target_z.shape[0]

    mean_loss = total_loss / max(n_batches, 1)
    mse_z = (sq_err_sum / max(n_seen, 1)).cpu().numpy()
    rmse_z = np.sqrt(mse_z)
    rmse_raw = rmse_z * normalizer.std
    return mean_loss, rmse_raw

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

    setup_seeds(cfg["data"]["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} | model: {args.model} epoch: {epochs}")

    train_loader, val_loader, normalizer = make_loaders(cfg)
    print(f"train batches: {len(train_loader)} | val batches {len(val_loader)}")

    model = build_model(cfg, args.model).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{args.model} params: {n_params:,}")

    optim = AdamW(model.parameters(),
                  lr=cfg["train"]["lr"],
                  weight_decay=cfg["train"]["weight_decay"])
    scheduler = CosineAnnealingLR(optim, T_max=epochs)
    loss_fn = nn.MSELoss()

# run_name = f"{args.model}-{time.strftime('%Y%m%d-%H%M%S')}"
#           wandb.init( 
#           project=cfg["wandb"]["project"],                                                                               
#           entity=cfg["wandb"]["entity"],                                                                                 
#           mode=args.wandb_mode,
#           config=cfg,                                                                                                    
#           name=run_name,                                                                                                 

#       )
    os.makedirs(args.out_dir, exist_ok=True)
    ckpt_path = os.path.join(args.out_dir, f"{args.model}_best.pt")
    best_val = float("inf")                                                                                            
   
    for epoch in range(epochs):                                                                                        
        train_loss = train_one_epoch(model, train_loader, loss_fn, optim, device)
        val_loss, rmse_raw = validate(model, val_loader, loss_fn, device, normalizer)                                  
        scheduler.step()
                                                                                                                     
        log = { 
          "epoch": epoch,                                                                                            
          "train_loss": train_loss,
          "val_loss": val_loss,
          "lr": scheduler.get_last_lr()[0],
        }                                                                                                              
        for name, v in zip(PARAM_NAMES, rmse_raw):
          log[f"rmse/{name}"] = float(v)                                                                             
        # wandb.log(log)                                                                                                 

        print(f"epoch {epoch:3d} | train {train_loss:.4f} | val {val_loss:.4f} "                                       
            f"| lr {scheduler.get_last_lr()[0]:.2e} | rmse {format_rmse_line(rmse_raw)}")
                                                                                                                     
        if val_loss < best_val:
          best_val = val_loss                                                                                        
          save_checkpoint(ckpt_path, model, args.model, cfg, normalizer, epoch, val_loss)
                                                                                                                     
    print(f"best val loss: {best_val:.4f} | checkpoint: {ckpt_path}")                                                  
  # wandb.finish()                                                                                                     
                                                                                                                         
                  
if __name__ == "__main__":
    main()
