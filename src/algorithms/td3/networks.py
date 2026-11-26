# Adapted from https://github.com/sfujim/TD3 (TD3.py: Actor, Critic), MIT
# license. Changes: the critic is SAC's DoubleCritic with this file's trunk, so
# TD3 reuses the SAC heads; the layers keep PyTorch's default initialisation
# like the reference (SAC's MLP uses jaxrl's orthogonal init instead).
"""TD3 deterministic actor and the default-initialised MLP trunk."""
from __future__ import annotations

import torch
import torch.nn as nn


class TorchMLP(nn.Module):
    """``depth`` ReLU hidden layers of width ``hidden_dim``, optional linear output.

    Same signature as ``src.algorithms.sac.networks.MLP`` (the SAC trunk
    protocol), but with ``nn.Linear``'s default init, as in Fujimoto's TD3.
    """

    def __init__(self, in_dim: int, hidden_dim: int, depth: int, out_dim: int | None = None) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for i in range(depth):
            layers += [nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim), nn.ReLU()]
        if out_dim is not None:
            layers.append(nn.Linear(hidden_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DeterministicActor(nn.Module):
    """``tanh(trunk(s))``: actions in ``[-1, 1]`` (DMC's action bounds)."""

    def __init__(self, obs_dim: int, action_dim: int, *, trunk, hidden_dim: int, depth: int) -> None:
        super().__init__()
        self.net = trunk(obs_dim, hidden_dim, depth, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.net(obs))
