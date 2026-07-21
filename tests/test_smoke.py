"""Smoke test: one full training cycle of DQN on CartPole.

Loads the experiment config, applies minimal-frame overrides so the run
finishes in a few seconds, and asserts that ``_train()`` returns a non-empty
metrics dict without raising.

Run with:
    pytest tests/test_smoke.py -v
"""
from __future__ import annotations

import pytest

from tests.conftest import load_experiment_cfg


BASE_OVERRIDES = [
    "logger=[]",
    "trainer.accelerator=cpu",
    "trainer.devices=[0]",
    "hydra.run.dir=/tmp/hydra_smoke_tests",
]


def _dqn_overrides() -> list[str]:
    # 600 frames in 100-frame batches: 1 warm-up batch then 5 update batches.
    # batch_size=8 keeps sampling cheap while ensuring buffer >= batch_size after batch 1.
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=600",
        "trainer.log_every_n_steps=100",
        "algorithm.frames_per_batch=100",
        "algorithm.init_random_frames=100",
        "algorithm.batch_size=8",
        "algorithm.num_updates=2",
        "algorithm.annealing_frames=600",
    ]


def test_smoke_dqn_cartpole():
    """DQN on CartPole-v1: discrete actions, MLP Q-network, replay buffer."""
    cfg = load_experiment_cfg("dqn/cartpole", _dqn_overrides())
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def _dqn_pong_overrides() -> list[str]:
    # Same shape as the cartpole overrides: 600 frames in 100-frame batches,
    # init_random_frames=100 so we hit the gradient path. Shrinks the 1M
    # replay buffer to 500 to keep memory bounded during the smoke run.
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=600",
        "trainer.log_every_n_steps=100",
        "algorithm.frames_per_batch=100",
        "algorithm.init_random_frames=100",
        "algorithm.batch_size=8",
        "algorithm.num_updates=2",
        "algorithm.annealing_frames=600",
        "algorithm.replay_buffer.storage.max_size=500",
    ]


def test_smoke_dqn_pong():
    """DQN on ALE/Pong-v5: pixel obs, NatureDQN CNN, eval-env split."""
    pytest.importorskip("ale_py")  # ALE is an optional system dep
    cfg = load_experiment_cfg("dqn/pong", _dqn_pong_overrides())
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def _ddpg_overrides() -> list[str]:
    # 600 frames in 100-frame batches: 1 warm-up batch then 5 update batches.
    # batch_size=8 keeps sampling cheap while ensuring buffer >= batch_size after batch 1.
    # Shrink the 1M replay buffer to 500 to keep memory bounded during the smoke run.
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=600",
        "trainer.log_every_n_steps=100",
        "algorithm.frames_per_batch=100",
        "algorithm.init_random_frames=100",
        "algorithm.batch_size=8",
        "algorithm.num_updates=2",
        "algorithm.replay_buffer.storage.max_size=500",
        "algorithm.exploration_noise.annealing_num_steps=600",
    ]


def test_smoke_ddpg_halfcheetah():
    """DDPG on HalfCheetah-v4: continuous actions, MLP actor/critic, OU noise."""
    pytest.importorskip("mujoco")  # MuJoCo is an optional system dep
    cfg = load_experiment_cfg("ddpg/halfcheetah", _ddpg_overrides())
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def _a2c_overrides() -> list[str]:
    # 600 frames in 120-frame rollouts: 5 collections, 6 mini-batches each
    # (mini_batch_size=20). On-policy: no replay buffer, no warm-up.
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=600",
        "trainer.log_every_n_steps=100",
        "algorithm.frames_per_batch=120",
        "algorithm.mini_batch_size=20",
    ]


def test_smoke_a2c_halfcheetah():
    """A2C on HalfCheetah-v4: continuous actions, stochastic actor + GAE."""
    pytest.importorskip("mujoco")  # MuJoCo is an optional system dep
    cfg = load_experiment_cfg("a2c/halfcheetah", _a2c_overrides())
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def _dreamer_overrides() -> list[str]:
    # Constraint: batch_size * batch_length >= train_ratio (128) so that
    # frames_per_batch = (batch_size*batch_length/train_ratio) >= 1.
    # batch_size=16, batch_length=8 → product=128, frames_per_batch=1.
    # First update fires after (batch_length+1)*action_repeat = 36 game frames
    # = 9 collector steps. total_frames=20 gives ~11 updates on a tiny model.
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=20",
        "trainer.log_every_n_steps=10",
        "trainer.num_envs=1",
        # Tiny model so CPU completes well within the 300s timeout
        "model.deter=64",
        "model.hidden=64",
        "model.discrete=8",
        "model.depth=8",
        "model.units=64",
        # Short sequences keep RSSM cheap; product must stay >= train_ratio
        "algorithm.buffer_config.batch_size=16",
        "algorithm.buffer_config.batch_length=8",
        "algorithm.buffer_config.max_size=500",
        # Disable compile — torch.compile on CPU takes minutes on first call
        "algorithm.dreamer_config.compile=false",
        # Short imagination horizon to reduce per-update cost
        "algorithm.dreamer_config.imag_horizon=3",
    ]


def test_smoke_dreamer_atari100k():
    """DreamerV3 on Atari100k (default env): pixel obs, RSSM world model, actor-critic."""
    pytest.importorskip("ale_py")
    cfg = load_experiment_cfg("dreamer/atari100k", _dreamer_overrides())
    from src.train import _train

    metrics = _train(cfg)
    # DreamerAlgorithm.step() always returns {} — metrics are accumulated
    # internally and flushed via pop_train_metrics() to logger callbacks.
    # Just verify the run completed without raising.
    assert isinstance(metrics, dict)
