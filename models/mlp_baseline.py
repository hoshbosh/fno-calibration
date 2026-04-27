from __future__ import annotations
from typing import Sequence
import torch
import torch.nn as nn

class MLPBaseline(nn.Module):
    def __init__(self, in_dim: int, hidden: Sequence[int], out_dim: int) -> None:
        '''
        in_dim: the number of input dimensions
        hidden: an array of dimensions for each hidden layer
        out_dim: the number of output dimensions
        '''
        super().__init__()
        layers: list[nn.Module] = [nn.Flatten()]
        prev = in_dim

        # For each hideen layer, create a linear activation function from the previous layer to the 
        # current hidden layer, then plug that into a GELU activation function
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.GELU()]
            prev = h

        layers.append(nn.Linear(prev, out_dim))

        # The actual net is a "sequential" object comprised of the layers
        self.net = nn.Sequential(*layers)

    # Called at each forward step of the model
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

# Build the mlp from the yaml config
def build_mlp(cfg: dict) -> MLPBaseline:
    m = cfg["mlp"]
    return MLPBaseline(m["in_dim"], hidden = m["hidden"], out_dim=m["out_dim"])

# Validation
if __name__ == "__main__":
    import yaml
    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    model = build_mlp(cfg)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    x = torch.randn(8, 1, cfg["grid"]["n_k"], cfg["grid"]["n_tau"])
    y = model(x)
    print(f"MLPBaseline: {n_params:,} params | input {tuple(x.shape)} -> output {tuple(y.shape)}")
