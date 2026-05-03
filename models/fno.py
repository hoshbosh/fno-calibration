from __future__ import annotations
from typing import Sequence
import torch
import torch.nn as nn

class FNO2d(nn.Module):
    def __init__(
            self, 
            in_channel: int, 
            hidden_channels: Sequence[int], 
            modes1: int,
            modes2: int,
            n_blocks: int,
            proj_channels: int,
            pool_size: Sequence[int],
            head_hidden: Sequence[int],
            out_dim: int,
    ) -> None:
        super().__init__()
        
        # Lifting layer, convolutional transformation into 64 channels
        self.lifting = nn.Conv2d(in_channel, hidden_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Shape of x is [batch, in_channel, n_k, n_tau]
        x = self.lifting(x) #[batch, in_channel, n_k, n_tau] -> [batch, hidden_channels, n_k, n_tau]
        return x
    
def build_fno(cfg : dict) -> FNO2d:
    f= cfg["fno"]
    return FNO2d(
            in_channel=f["in_channel"],
            hidden_channels=f["hidden_channels"],
            modes1=f["modes1"],
            modes2=f["modes2"],
            n_blocks=f["n_blocks"],
            proj_channels=f["proj_channels"],
            pool_size=f["pool_size"],
            head_hidden=f["head_hidden"],
            out_dim=f["out_dim"],
        )

# Validation
if __name__ == "__main__":
    import yaml
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)
