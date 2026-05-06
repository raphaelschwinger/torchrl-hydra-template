"""Atari CNN encoder used by image-based DQN/PPO experiments.

Mirrors the architecture from Mnih et al. (2015) "Human-level control through
deep reinforcement learning": three conv layers (32/64/64 channels with kernels
8/4/3 and strides 4/2/1) followed by a 512-unit linear projection and a final
linear head with ``out_features`` outputs.
"""

from __future__ import annotations

import torch
from torch import nn
from torchrl.modules import ConvNet, MLP


class AtariCNN(nn.Module):
    def __init__(self, *, out_features: int, hidden_features: int = 512) -> None:
        super().__init__()
        self.out_features = out_features
        self.conv = ConvNet(
            num_cells=[32, 64, 64],
            kernel_sizes=[8, 4, 3],
            strides=[4, 2, 1],
            activation_class=nn.ReLU,
        )
        self.head = MLP(
            in_features=None,
            out_features=out_features,
            num_cells=[hidden_features],
            activation_class=nn.ReLU,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))
