# Adapted from https://github.com/nicklashansen/tdmpc2 (tdmpc2/common/layers.py
# and tdmpc2/common/init.py), MIT license. Changes: `cfg` parameters replaced by
# explicit arguments; conv/pixel encoders removed (state observations only).
"""Shared network building blocks: LayerNorm+Mish MLPs, SimNorm, vmapped ensembles."""
from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import from_modules


class Ensemble(nn.Module):
    """Vectorized ensemble of identically-shaped modules (single vmapped forward)."""

    def __init__(self, modules_list: list[nn.Module], **kwargs) -> None:
        super().__init__()
        # combine_state_for_ensemble causes graph breaks under torch.compile
        self.params = from_modules(*modules_list, as_module=True)
        with self.params[0].data.to("meta").to_module(modules_list[0]):
            self.module = deepcopy(modules_list[0])
        self._repr = str(modules_list[0])
        self._n = len(modules_list)

    def __len__(self) -> int:
        return self._n

    def _call(self, params, *args, **kwargs):
        with params.to_module(self.module):
            return self.module(*args, **kwargs)

    def forward(self, *args, **kwargs):
        return torch.vmap(self._call, (0, None), randomness="different")(
            self.params, *args, **kwargs
        )

    def __repr__(self) -> str:
        return f"Vectorized {len(self)}x " + self._repr


class SimNorm(nn.Module):
    """Simplicial normalization: softmax over groups of ``dim`` latent entries.

    Adapted from https://arxiv.org/abs/2204.00616.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = x.shape
        x = x.view(*shp[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.view(*shp)

    def __repr__(self) -> str:
        return f"SimNorm(dim={self.dim})"


class NormedLinear(nn.Linear):
    """Linear layer with LayerNorm, activation, and optional dropout."""

    def __init__(self, *args, dropout: float = 0.0, act: nn.Module | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        if act is None:
            act = nn.Mish(inplace=False)
        self.act = act
        self.dropout = nn.Dropout(dropout, inplace=False) if dropout else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        if self.dropout:
            x = self.dropout(x)
        return self.act(self.ln(x))

    def __repr__(self) -> str:
        repr_dropout = f", dropout={self.dropout.p}" if self.dropout else ""
        return (
            f"NormedLinear(in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"bias={self.bias is not None}{repr_dropout}, "
            f"act={self.act.__class__.__name__})"
        )


def mlp(
    in_dim: int,
    mlp_dims: int | list[int],
    out_dim: int,
    act: nn.Module | None = None,
    dropout: float = 0.0,
) -> nn.Sequential:
    """MLP with LayerNorm + Mish layers; basic building block of TD-MPC2."""
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + list(mlp_dims) + [out_dim]
    layers = nn.ModuleList()
    for i in range(len(dims) - 2):
        layers.append(NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0)))
    layers.append(
        NormedLinear(dims[-2], dims[-1], act=act) if act else nn.Linear(dims[-2], dims[-1])
    )
    return nn.Sequential(*layers)


def weight_init(m: nn.Module) -> None:
    """Custom weight initialization for TD-MPC2."""
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Embedding):
        nn.init.uniform_(m.weight, -0.02, 0.02)
    elif isinstance(m, nn.ParameterList):
        for i, p in enumerate(m):
            if p.dim() == 3:  # Linear
                nn.init.trunc_normal_(p, std=0.02)  # Weight
                nn.init.constant_(m[i + 1], 0)  # Bias


def zero_(params) -> None:
    """Initialize parameters to zero."""
    for p in params:
        p.data.fill_(0)
