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
            layer_modes: Sequence[tuple[int, int]],
            proj_channels: int,
            pool_size: Sequence[int],
            head_hidden: Sequence[int],
            out_dim: int,
    ) -> None:
        super().__init__()
        
        # Lifting layer, convolutional transformation into 64 channels
        self.lifting = nn.Conv2d(in_channel, hidden_channels, kernel_size=1)
        self.fourier_blocks = nn.ModuleList([
            FourierBlock(hidden_channels, m1, m2) for m1, m2 in layer_modes
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

    def forward(
            self,
            iv_surface: torch.Tensor,
            k_grid: torch.Tensor,
            tau_grid: torch.Tensor,
    ) -> torch.Tensor:
        # iv_surface: (batch, 1, n_k, n_tau)
        # k_grid:     (n_k,)
        # tau_grid:   (n_tau,)
        batch = iv_surface.shape[0]

        # Build coordinate channels on-the-fly from the (possibly resolution-varying) grid.
        # meshgrid with indexing='ij' so K varies along axis 0 (n_k) and T along axis 1 (n_tau).
        K, T = torch.meshgrid(
            k_grid.to(iv_surface.device),
            tau_grid.to(iv_surface.device),
            indexing="ij",
        )
        # Stack into (2, n_k, n_tau), add batch dim, expand (no copy) to (batch, 2, n_k, n_tau).
        coords = torch.stack([K, T], dim=0).unsqueeze(0).expand(batch, -1, -1, -1)

        # Concatenate IV with coord channels -> (batch, 3, n_k, n_tau)
        x = torch.cat([iv_surface, coords], dim=1)

        x = self.lifting(x)              # -> (batch, hidden_channels, n_k, n_tau)
        for block in self.fourier_blocks:
<<<<<<< jlabasbas/fno-input-refactor
            x = block(x)                 # shape preserved
=======
            x = block(x)
        x = self.projection(x)
        x_pooled = self.pool(x).flatten(1)
        x = self.head(x_pooled)
        # aux = self.aux_head(x_pooled) # Optional auxilary head
        
>>>>>>> master
        return x
    
def build_fno(cfg : dict) -> FNO2d:
    f= cfg["fno"]
    return FNO2d(
            in_channel=f["in_channel"],
            hidden_channels=f["hidden_channels"],
            layer_modes = f['layer_modes'],
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

    # Native grid from config
    n_k, n_tau = cfg["grid"]["n_k"], cfg["grid"]["n_tau"]
    k_grid = torch.linspace(cfg["grid"]["k_min"], cfg["grid"]["k_max"], n_k)
    tau_grid = torch.tensor(
        [cfg["grid"]["tau_min"] * (cfg["grid"]["tau_max"] / cfg["grid"]["tau_min"]) ** (i / (n_tau - 1))
         for i in range(n_tau)],
        dtype=torch.float32,
    )
    iv = torch.randn(8, 1, n_k, n_tau)
    y = model(iv, k_grid, tau_grid)
    print(f"FNO2d: {n_params:,} params | input {tuple(iv.shape)} -> output {tuple(y.shape)}")

    # Resolution invariance check: same model, different grid
    n_k2, n_tau2 = 32, 40
    k_grid2 = torch.linspace(cfg["grid"]["k_min"], cfg["grid"]["k_max"], n_k2)
    tau_grid2 = torch.tensor(
        [cfg["grid"]["tau_min"] * (cfg["grid"]["tau_max"] / cfg["grid"]["tau_min"]) ** (i / (n_tau2 - 1))
         for i in range(n_tau2)],
        dtype=torch.float32,
    )
    iv2 = torch.randn(2, 1, n_k2, n_tau2)
    y2 = model(iv2, k_grid2, tau_grid2)
    print(f"  resolution check: {tuple(iv2.shape)} -> {tuple(y2.shape)}")
