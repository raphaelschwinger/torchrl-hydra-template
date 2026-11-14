"""Regression tests for two mid-run crashes of the Rainbow/DER Atari-100k sweeps.

1. Tied Q-values made torchrl's Q-value actors emit a two-hot action, which
   crashed ``DistributionalDQNLoss`` once sampled from the replay buffer.
2. torchrl's ``NoopResetEnv`` raised when every retry of its random no-ops
   ended in a life loss (Breakout).
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
import torch
import torch.nn as nn
from tensordict import TensorDict
from torchrl.data import OneHot
from torchrl.envs import GymWrapper, TransformedEnv
from torchrl.envs import NoopResetEnv as TorchRLNoopResetEnv
from torchrl.modules import DistributionalQValueActor, QValueActor

from src.algorithms.rainbow.rainbow import strict_one_hot_
from src.environments.atari_wrappers import NoopResetEnv

ATOMS, ACTIONS = 5, 4


class _TiedDistribution(nn.Module):
    """C51 logits whose actions 1 and 2 have identical, best distributions."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = torch.zeros(x.shape[0], ATOMS, ACTIONS)
        logits[:, -1, 1] = logits[:, -1, 2] = 5.0
        return logits


class _TiedQ(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor([[0.0, 2.0, 2.0, 1.0]]).expand(x.shape[0], -1)


def _act(actor) -> torch.Tensor:
    return actor(TensorDict(obs=torch.zeros(3, 2), batch_size=[3]))["action"]


def test_distributional_actor_tie_is_two_hot_without_fix():
    """Documents the torchrl behaviour the fix exists for."""
    actor = DistributionalQValueActor(
        _TiedDistribution(), support=torch.linspace(-1, 1, ATOMS),
        spec=OneHot(ACTIONS), in_keys=["obs"],
    )
    assert (_act(actor).sum(-1) == 2).all()


@pytest.mark.parametrize("distributional", [True, False])
def test_strict_one_hot_picks_first_max(distributional):
    if distributional:
        actor = DistributionalQValueActor(
            _TiedDistribution(), support=torch.linspace(-1, 1, ATOMS),
            spec=OneHot(ACTIONS), in_keys=["obs"],
        )
    else:
        actor = QValueActor(_TiedQ(), spec=OneHot(ACTIONS), in_keys=["obs"])
    action = _act(strict_one_hot_(actor))
    # Exactly one hot, on the first maximum: the action OneHot.to_numpy executes.
    assert action.tolist() == [[0, 1, 0, 0]] * 3


class _CounterEnv(gym.Env):
    """Observation = steps since reset; terminates every step if `always_done`."""

    def __init__(self, always_done: bool) -> None:
        self.always_done = always_done
        self.observation_space = gym.spaces.Box(0, 1e6, (1,), np.float32)
        self.action_space = gym.spaces.Discrete(3)
        self.t = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        return np.array([0.0], np.float32), {}

    def step(self, action):
        self.t += 1
        return np.array([self.t], np.float32), 0.0, self.always_done, False, {}


def _env(transform_cls, always_done: bool) -> TransformedEnv:
    return TransformedEnv(
        GymWrapper(_CounterEnv(always_done)), transform_cls(noops=5, random=False)
    )


def test_noop_reset_survives_env_that_is_always_done():
    with pytest.raises(RuntimeError, match="repeatedly done"):
        _env(TorchRLNoopResetEnv, always_done=True).reset()
    td = _env(NoopResetEnv, always_done=True).reset()
    assert td["observation"].item() == 0.0  # the state right after the last reset


def test_noop_reset_matches_torchrl_when_it_does_not_raise():
    results = []
    for cls in (TorchRLNoopResetEnv, NoopResetEnv):
        env = TransformedEnv(GymWrapper(_CounterEnv(False)), cls(noops=30, random=True))
        torch.manual_seed(0)
        results.append([env.reset()["observation"].item() for _ in range(20)])
    assert results[0] == results[1]
    assert len(set(results[0])) > 1  # the random no-op count actually varies
