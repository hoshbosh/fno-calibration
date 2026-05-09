from __future__ import annotations
import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import yaml
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
                    grad_clip: float) -> float:
    model.train()
    total_loss = 0.0
    n_batches = 0
    use_amp = (device == "cuda")
    for surface, target_z in loader:
        surface = surface.to(device, non_blocking=True)
        target_z = target_z.to(device, non_blocking=True)

        optim.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            pred_z = model(surface)
            loss = loss_fn(pred_z, target_z)
        # loss.backward()
        # optim.step()
        scaler.scale(loss).backward()
        scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optim)
        scaler.update()
        scheduler.step()

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
    use_amp = (device == "cuda")
    for surface, target_z in loader:
        surface = surface.to(device, non_blocking=True)
        target_z = target_z.to(device, non_blocking=True)
        
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            pred_z = model(surface)
            loss = loss_fn(pred_z, target_z)

        total_loss += loss.item()
        n_batches += 1

        sq_err_sum += ((pred_z - target_z) ** 2).sum(dim=0).float()
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
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

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
    # Patience is used for stopping the training if validation loss does not improve, stopping early to prevent overfitting
    epochs_since_best = 0
    patience = cfg["train"]["early_stopping_patience"]

    grad_clip = cfg["train"]["grad_clip"]
   
    for epoch in range(epochs):                                                                                        
        train_loss = train_one_epoch(model, train_loader, loss_fn, optim, device, scaler, scheduler, grad_clip)
        val_loss, rmse_raw = validate(model, val_loader, loss_fn, device, normalizer)                                  
                                                                                                                     
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
          epochs_since_best = 0
          save_checkpoint(ckpt_path, model, args.model, cfg, normalizer, epoch, val_loss)
        else:
            # we have not improved for another epoch
            epochs_since_best += 1

        if epochs_since_best >= patience:
            print(f"early stopping at epoch {epoch}")
            break
                                                                                                                     
    print(f"best val loss: {best_val:.4f} | checkpoint: {ckpt_path}")                                                  

    # useful for the future
    # state = torch.load(ckpt_path, map_location=device)
    # model.load_state_dict(state["model_state"])
  # wandb.finish()                                                                                                     
                                                                                                                         
                  
if __name__ == "__main__":
    main()
