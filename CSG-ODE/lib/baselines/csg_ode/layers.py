"""Small shared neural-network building blocks."""

from __future__ import annotations

from torch import nn


def activation_module(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "elu":
        return nn.ELU()
    if name == "softplus":
        return nn.Softplus()
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {name}")


class MLP(nn.Sequential):
    """MLP where ``depth`` is the number of hidden layers."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        width: int,
        depth: int = 1,
        activation: str = "tanh",
        dropout: float = 0.0,
    ) -> None:
        if depth < 0:
            raise ValueError("depth must be non-negative")
        if depth == 0:
            layers: list[nn.Module] = [nn.Linear(input_dim, output_dim)]
        else:
            layers = [nn.Linear(input_dim, width), activation_module(activation)]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            for _ in range(depth - 1):
                layers.extend([nn.Linear(width, width), activation_module(activation)])
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(width, output_dim))
        super().__init__(*layers)
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
