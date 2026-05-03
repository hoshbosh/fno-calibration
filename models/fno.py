from __future__ import annotations
from typing import Sequence
import torch
import torch.nn as nn

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )
        self.weights2 = nn.Parameter(
            scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_ft = torch.fft.rfft2(x)

        m1 = min(self.modes1, x.size(-2))
        m2 = min(self.modes2, x.size(-1) // 2 + 1)

        out_ft = torch.zeros(
            x.shape[0], self.out_channels, x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )

        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bixy, ioxy -> boxy",
            x_ft[:, :, :m1, :m2],
            self.weights1[:, :, :m1, :m2],
        )

        if x.size(-2) > m1:
            out_ft[:, :, :m1, :m2] = torch.einsum(
                "bixy, ioxy -> boxy",
                x_ft[:, :, :m1, :m2],
                self.weights2[:, :, :m1, :m2],
            )

        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))

class FourerBlock(nn.Module):
    def __init__(self, channels: int, modes1: int, modes2: int):
        super().__init__()
        self.spectral_conv = SpectralConv2d(channels, channels, modes1, modes2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spectral_conv(x)

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
