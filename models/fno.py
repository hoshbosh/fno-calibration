from __future__ import annotations
from typing import Sequence
import torch
import torch.nn as nn

class SpectralConv2d(nn.Module):
    """
    Global Convolution parameterized in fourier space
    """
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int) -> None:
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
        batch, _, n_k, n_tau = x.shape

        x_ft = torch.fft.rfft2(x)

        m1 = min(self.modes1, n_k)
        m2 = min(self.modes2, n_tau // 2 + 1)

        out_ft = torch.zeros(
                batch, self.out_channels, n_k, n_tau // 2 + 1,
                dtype=torch.cfloat, device=x.device
        )
        out_ft[:, :, :m1, :m2] = torch.einsum(
            "bixy, ioxy -> boxy",                                                                                                            
            x_ft[:, :, :m1, :m2],                                                                                                            
            self.weights1[:, :, :m1, :m2],                                                                                                   
        )                                                                                                                                    

        if n_k > m1:
            out_ft[:, :, -m1:, :m2] = torch.einsum(                                                                                          
                "bixy, ioxy -> boxy",                                                                                                        
                x_ft[:, :, -m1:, :m2],
                self.weights2[:, :, :m1, :m2],                                                                                               
            )                                                                                                                                

        return torch.fft.irfft2(out_ft, s=(n_k, n_tau))

class FourierBlock(nn.Module):
    def __init__(self, channels: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.spectral_conv = SpectralConv2d(channels, channels, modes1, modes2)
        self.pointwise_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.activation(self.spectral_conv(x) + self.pointwise_conv(x))




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
        self.fourier_blocks = nn.ModuleList([
            FourierBlock(hidden_channels, modes1, modes2) for _ in range(n_blocks)
        ])
        # Expands channels 64->128, applies GELU nonlinearity, then compresses back 128->64
        self.projection = nn.Sequential(
            nn.Conv2d(hidden_channels, proj_channels, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(proj_channels, hidden_channels, kernel_size=1)
        )
        # Pools spatial grid to (4,5) bins by averaging; adaptive average allows FNO to have resolution invariance
        self.pool = nn.AdaptiveAvgPool2d((4,5))

        # Expands channels 128->256, applies GELU, and compresses 256->64, GELU, Dropout(0.1)
        head_layers = []
        head_in = hidden_channels * pool_size[0] * pool_size[1]  # 64 * 4 * 5 = 1280
        prev = head_in
        for h in head_hidden:
            head_layers += [nn.Linear(prev, h), nn.GELU(), nn.Dropout(0.1)]
            prev = h

        # Linear(64->5)
        head_layers.append(nn.Linear(prev, out_dim))
        self.head = nn.Sequential(*head_layers)

        # # Optional auxilary head
        # self.aux_head = nn.Sequential(
        #     nn.Linear(head_in, 64),
        #     nn.GELU(),
        #     nn.Linear(64, 3)
        # )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
    # Shape of x is [batch, in_channel, n_k, n_tau]
        x = self.lifting(x) #[batch, in_channel, n_k, n_tau] -> [batch, hidden_channels, n_k, n_tau]
        for block in self.fourier_blocks:
            x = block(x)
        x = self.projection(x)
        x_pooled = self.pool(x).flatten(1)
        x = self.head(x_pooled)
        # aux = self.aux_head(x_pooled) # Optional auxilary head
        
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

    model = build_fno(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    x = torch.randn(8, cfg["fno"]["in_channel"], cfg["grid"]["n_k"], cfg["grid"]["n_tau"])                                                   
    y = model(x)                                                                                                                             
    print(f"FNO2d: {n_params:,} params | input {tuple(x.shape)} -> output {tuple(y.shape)}")                                                 
                                                                                                                                           
    # Resolution invariance check: same model, different grid
    x2 = torch.randn(2, cfg["fno"]["in_channel"], 32, 40)                                                                                    
    y2 = model(x2)                                                                                                                           
    print(f"  resolution check: {tuple(x2.shape)} -> {tuple(y2.shape)}")
