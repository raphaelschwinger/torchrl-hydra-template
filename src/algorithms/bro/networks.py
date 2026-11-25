# Adapted from https://github.com/naumix/BiggerRegularizedOptimistic
# (jaxrl/networks/common.py: BroNet, jaxrl/networks/policies.py: DualTanhPolicy,
# jaxrl/bro/temperature.py: Adjustment), MIT license.
# Changes: ported from flax to PyTorch; the depth-unrolled BroNet is a loop.
"""BroNet, the optimistic actor and BRO's bounded dual variables.

The SAC heads (double critic, tanh-Gaussian actor) are reused from
``src.algorithms.sac.networks`` with BroNet as their trunk.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.algorithms.sac.networks import orthogonal_linear

# flax.linen.LayerNorm default epsilon (torch's default is 1e-5).
LN_EPS = 1e-6


class BroNet(nn.Module):
    """Dense -> LN -> ReLU, then ``depth`` residual blocks of
    (Dense -> LN -> ReLU -> Dense -> LN) + skip; optional linear output.

    Same call signature as the SAC ``MLP`` trunk: ``(in_dim, hidden_dim, depth, out_dim=None)``.
    """

    def __init__(self, in_dim: int, hidden_dim: int, depth: int, out_dim: int | None = None) -> None:
        super().__init__()
        self.inp = orthogonal_linear(in_dim, hidden_dim)
        self.inp_ln = nn.LayerNorm(hidden_dim, eps=LN_EPS)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                orthogonal_linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim, eps=LN_EPS),
                nn.ReLU(),
                orthogonal_linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim, eps=LN_EPS),
            )
            for _ in range(depth)
        )
        self.out = orthogonal_linear(hidden_dim, out_dim) if out_dim is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.inp_ln(self.inp(x)))
        for block in self.blocks:
            x = x + block(x)
        return self.out(x) if self.out is not None else x


class DualTanhActor(nn.Module):
    """Optimistic exploration actor: shifts the conservative mean, scales its std.

    ``mean_o = mean_c + W BroNet([obs, mean_c])`` (no bias, orthogonal init x0.01),
    ``std_o = std_c * std_multiplier``.
    """

    def __init__(self, obs_dim: int, action_dim: int, *, hidden_dim: int, depth: int, scale_means: float = 0.01) -> None:
        super().__init__()
        self.trunk = BroNet(obs_dim + action_dim, hidden_dim, depth)
        self.shift = orthogonal_linear(hidden_dim, action_dim, scale=scale_means, bias=False)

    def forward(
        self, obs: torch.Tensor, mean_c: torch.Tensor, std_c: torch.Tensor, std_multiplier: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(torch.cat([obs, mean_c], dim=-1))
        return mean_c + self.shift(h), std_c * std_multiplier


class BoundedLogParam(nn.Module):
    """Scalar ``exp(lo + (hi - lo) * 0.5 * (1 + tanh(p)))`` (BRO's ``Adjustment``),
    with ``p`` initialised so the output equals ``init_value``."""

    def __init__(self, init_value: float, lo: float = -10.0, hi: float = 7.5) -> None:
        super().__init__()
        self.lo, self.hi = lo, hi
        p0 = math.atanh((math.log(init_value) - lo) / ((hi - lo) * 0.5) - 1)
        self.p = nn.Parameter(torch.tensor(p0))

    def forward(self) -> torch.Tensor:
        return torch.exp(self.lo + (self.hi - self.lo) * 0.5 * (1 + torch.tanh(self.p)))
