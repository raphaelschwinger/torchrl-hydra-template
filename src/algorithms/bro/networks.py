# Adapted from https://github.com/naumix/BiggerRegularizedOptimistic
# (jaxrl/networks/common.py: BroNet, jaxrl/networks/critic_net.py,
# jaxrl/networks/policies.py: NormalTanhPolicy, DualTanhPolicy), MIT license.
# Changes: ported from flax to PyTorch; depth-unrolled BroNet written as a loop;
# DoubleCritic kept as two independent modules (vmappable later via the
# runtime ablation).
"""BroNet and the BRO actor/critic heads."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# flax.linen.LayerNorm default epsilon (torch's default is 1e-5).
LN_EPS = 1e-6


def _orthogonal_linear(in_dim: int, out_dim: int, *, scale: float = math.sqrt(2), bias: bool = True) -> nn.Linear:
    """Dense layer with flax's ``orthogonal(scale)`` kernel init and zero bias."""
    layer = nn.Linear(in_dim, out_dim, bias=bias)
    nn.init.orthogonal_(layer.weight, gain=scale)
    if bias:
        nn.init.zeros_(layer.bias)
    return layer


class BroNet(nn.Module):
    """Dense -> LN -> ReLU, then ``depth`` residual blocks of
    (Dense -> LN -> ReLU -> Dense -> LN) + skip. No activation after the skip.
    """

    def __init__(self, in_dim: int, hidden_dim: int, depth: int, out_dim: int | None = None) -> None:
        super().__init__()
        self.inp = _orthogonal_linear(in_dim, hidden_dim)
        self.inp_ln = nn.LayerNorm(hidden_dim, eps=LN_EPS)
        self.blocks = nn.ModuleList(
            nn.Sequential(
                _orthogonal_linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim, eps=LN_EPS),
                nn.ReLU(),
                _orthogonal_linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim, eps=LN_EPS),
            )
            for _ in range(depth)
        )
        self.out = _orthogonal_linear(hidden_dim, out_dim) if out_dim is not None else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.inp_ln(self.inp(x)))
        for block in self.blocks:
            x = x + block(x)
        return self.out(x) if self.out is not None else x


class DoubleCritic(nn.Module):
    """Two independent BroNet critics on ``[obs, action]``.

    Returns ``(q1, q2)`` of shape ``[B, n_quantiles]`` (``[B]`` if 1 output).
    """

    def __init__(self, obs_dim: int, action_dim: int, *, hidden_dim: int, depth: int, n_outputs: int) -> None:
        super().__init__()
        self.n_outputs = n_outputs
        self.q1 = BroNet(obs_dim + action_dim, hidden_dim, depth, n_outputs)
        self.q2 = BroNet(obs_dim + action_dim, hidden_dim, depth, n_outputs)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        q1, q2 = self.q1(x), self.q2(x)
        if self.n_outputs == 1:
            q1, q2 = q1.squeeze(-1), q2.squeeze(-1)
        return q1, q2


def tanh_normal_sample(mean: torch.Tensor, std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reparameterised sample from tanh(N(mean, std)) and its log-prob (summed over dims).

    Uses the numerically stable ``log(1 - tanh(u)^2) = 2 (log 2 - u - softplus(-2u))``
    that tfp's ``Tanh`` bijector uses.
    """
    u = mean + std * torch.randn_like(mean)
    log_prob = (
        -0.5 * ((u - mean) / std) ** 2 - torch.log(std) - 0.5 * math.log(2 * math.pi)
    ).sum(-1)
    log_prob = log_prob - (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(-1)
    return torch.tanh(u), log_prob


class NormalTanhActor(nn.Module):
    """Pessimistic (conservative) actor: BroNet trunk -> mean, tanh-squashed log-std."""

    def __init__(
        self, obs_dim: int, action_dim: int, *, hidden_dim: int, depth: int,
        log_std_min: float = -10.0, log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.trunk = BroNet(obs_dim, hidden_dim, depth)
        self.mean = _orthogonal_linear(hidden_dim, action_dim)
        self.log_std = _orthogonal_linear(hidden_dim, action_dim, scale=1.0)
        self.log_std_min, self.log_std_max = log_std_min, log_std_max

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        log_std = self.log_std(h)
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (1 + torch.tanh(log_std))
        return self.mean(h), log_std.exp()


class DualTanhActor(nn.Module):
    """Optimistic exploration actor: shifts the conservative mean, scales its std.

    ``mean_o = mean_c + W [obs, mean_c]`` (no bias, orthogonal init x0.01),
    ``std_o = std_c * std_multiplier``.
    """

    def __init__(
        self, obs_dim: int, action_dim: int, *, hidden_dim: int, depth: int, scale_means: float = 0.01,
    ) -> None:
        super().__init__()
        self.trunk = BroNet(obs_dim + action_dim, hidden_dim, depth)
        self.shift = _orthogonal_linear(hidden_dim, action_dim, scale=scale_means, bias=False)

    def forward(
        self, obs: torch.Tensor, mean_c: torch.Tensor, std_c: torch.Tensor, std_multiplier: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(torch.cat([obs, mean_c], dim=-1))
        return mean_c + self.shift(h), std_c * std_multiplier


class BoundedLogParam(nn.Module):
    """Scalar ``exp(lo + (hi - lo) * 0.5 * (1 + tanh(p)))`` (BRO's ``Adjustment``).

    ``p`` is initialised so the output equals ``init_value``.
    """

    def __init__(self, init_value: float, lo: float = -10.0, hi: float = 7.5) -> None:
        super().__init__()
        self.lo, self.hi = lo, hi
        p0 = math.atanh((math.log(init_value) - lo) / ((hi - lo) * 0.5) - 1)
        self.p = nn.Parameter(torch.tensor(p0))

    def forward(self) -> torch.Tensor:
        return torch.exp(self.lo + (self.hi - self.lo) * 0.5 * (1 + torch.tanh(self.p)))


class LogParam(nn.Module):
    """Scalar ``exp(p)`` (SAC temperature)."""

    def __init__(self, init_value: float) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(math.log(init_value)))

    def forward(self) -> torch.Tensor:
        return self.p.exp()
