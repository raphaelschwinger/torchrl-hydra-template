"""Shared exploration modules used by Rainbow/DER and BBF."""
from __future__ import annotations

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase


class FixedEpsilonGreedy(TensorDictModuleBase):
    """Tiny fixed-epsilon eval policy.

    Unlike ``EGreedyModule`` it also acts under ``ExplorationType.MODE``
    (used by ``BaseTrainer.evaluate``). The tiny epsilon matters on Atari: a
    fully deterministic policy can freeze (e.g. never pressing FIRE to launch
    the Breakout ball).
    """

    def __init__(self, action_spec, eps: float) -> None:
        self.in_keys = ["action"]
        self.out_keys = ["action"]
        super().__init__()
        self.action_spec = action_spec
        self.eps = eps

    def forward(self, tensordict: TensorDict) -> TensorDict:
        if self.eps > 0 and float(torch.rand(())) < self.eps:
            random_action = self.action_spec.rand().to(tensordict["action"].device)
            tensordict.set("action", random_action)
        return tensordict
