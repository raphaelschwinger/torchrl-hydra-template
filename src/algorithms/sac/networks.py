# Adapted from https://github.com/ikostrikov/jaxrl (jaxrl/networks/common.py:
# MLP, critic_net.py: DoubleCritic, policies.py: NormalTanhPolicy), MIT license,
# in the form BRO's official code (github.com/naumix/BiggerRegularizedOptimistic)
# keeps them. Changes: ported from flax to PyTorch; the trunk is a constructor
# argument so BRO can swap in BroNet without re-implementing the heads.
"""SAC actor/critic heads, a plain MLP trunk and the tanh-Gaussian sampler."""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Builds a trunk: ``trunk(in_dim, hidden_dim, depth, out_dim=None) -> nn.Module``
#: whose output width is ``hidden_dim`` (or ``out_dim`` when given).
Trunk = Callable[..., nn.Module]


def orthogonal_linear(in_dim: int, out_dim: int, *, scale: float = math.sqrt(2), bias: bool = True) -> nn.Linear:
    """Dense layer with flax's ``orthogonal(scale)`` kernel init and zero bias (jaxrl's default)."""
    layer = nn.Linear(in_dim, out_dim, bias=bias)
    nn.init.orthogonal_(layer.weight, gain=scale)
    if bias:
        nn.init.zeros_(layer.bias)
    return layer


class MLP(nn.Module):
    """``depth`` ReLU hidden layers of width ``hidden_dim``, optional linear output."""

    def __init__(self, in_dim: int, hidden_dim: int, depth: int, out_dim: int | None = None) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(depth):
            layers += [orthogonal_linear(in_dim if i == 0 else hidden_dim, hidden_dim), nn.ReLU()]
        if out_dim is not None:
            layers.append(orthogonal_linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DoubleCritic(nn.Module):
    """Two independent critics on ``[obs, action]``; ``(q1, q2)`` of shape [B] or [B, n_outputs]."""

    def __init__(
        self, obs_dim: int, action_dim: int, *, trunk: Trunk, hidden_dim: int, depth: int, n_outputs: int = 1,
    ) -> None:
        super().__init__()
        self.n_outputs = n_outputs
        self.q1 = trunk(obs_dim + action_dim, hidden_dim, depth, n_outputs)
        self.q2 = trunk(obs_dim + action_dim, hidden_dim, depth, n_outputs)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([obs, action], dim=-1)
        q1, q2 = self.q1(x), self.q2(x)
        if self.n_outputs == 1:
            q1, q2 = q1.squeeze(-1), q2.squeeze(-1)
        return q1, q2


class NormalTanhActor(nn.Module):
    """Trunk -> mean and a tanh-squashed log-std in ``[log_std_min, log_std_max]``."""

    def __init__(
        self, obs_dim: int, action_dim: int, *, trunk: Trunk, hidden_dim: int, depth: int,
        log_std_min: float = -10.0, log_std_max: float = 2.0,
    ) -> None:
        super().__init__()
        self.trunk = trunk(obs_dim, hidden_dim, depth)
        self.mean = orthogonal_linear(hidden_dim, action_dim)
        self.log_std = orthogonal_linear(hidden_dim, action_dim, scale=1.0)
        self.log_std_min, self.log_std_max = log_std_min, log_std_max

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        log_std = self.log_std(h)
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (1 + torch.tanh(log_std))
        return self.mean(h), log_std.exp()


class LogParam(nn.Module):
    """Scalar ``exp(p)``, e.g. the SAC temperature."""

    def __init__(self, init_value: float) -> None:
        super().__init__()
        self.p = nn.Parameter(torch.tensor(math.log(init_value)))

    def forward(self) -> torch.Tensor:
        return self.p.exp()


def tanh_normal_sample(mean: torch.Tensor, std: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Reparameterised sample of ``tanh(N(mean, std))`` and its log-prob (summed over dims).

    Uses the numerically stable ``log(1 - tanh(u)^2) = 2 (log 2 - u - softplus(-2u))``
    of tfp's ``Tanh`` bijector.
    """
    u = mean + std * torch.randn_like(mean)
    log_prob = (-0.5 * ((u - mean) / std) ** 2 - torch.log(std) - 0.5 * math.log(2 * math.pi)).sum(-1)
    log_prob = log_prob - (2 * (math.log(2) - u - F.softplus(-2 * u))).sum(-1)
    return torch.tanh(u), log_prob
