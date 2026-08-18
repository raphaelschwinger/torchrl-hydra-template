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
    "eval=false",  # skip the post-training evaluation stage; keeps runs to seconds
    # The compose API can't resolve ${hydra:runtime.output_dir}; checkpointing
    # is on by default (save_last), so point it at a literal path to keep the
    # default checkpoint path exercised.
    "checkpoint.save_dir=/tmp/hydra_smoke_tests/checkpoints",
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
    cfg = load_experiment_cfg("dqn/gym", _dqn_overrides())
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
    cfg = load_experiment_cfg("dqn/ale", _dqn_pong_overrides())
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
    cfg = load_experiment_cfg("ddpg/gym", _ddpg_overrides())
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def _ppo_overrides() -> list[str]:
    # 256 frames in 64-frame rollouts: 4 collections, 2 epochs x 2 mini-batches
    # each (mini_batch_size=32). On-policy: no replay buffer, no warm-up.
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=256",
        "trainer.log_every_n_steps=64",
        "algorithm.frames_per_batch=64",
        "algorithm.mini_batch_size=32",
        "algorithm.num_epochs=2",
        "algorithm.anneal_frames=256",
    ]


def test_smoke_ppo_dmc_cheetah():
    """PPO on DMC cheetah-run: dm_control backend, Normal + clip policy."""
    pytest.importorskip("dm_control")  # DMC is an optional system dep
    cfg = load_experiment_cfg("ppo/dmc", _ppo_overrides())
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def test_smoke_ppo_jamesbond():
    """PPO on ALE/Jamesbond-v5: pixel obs, shared CNN trunk, eval-env split."""
    pytest.importorskip("ale_py")  # ALE is an optional system dep
    cfg = load_experiment_cfg("ppo/ale", [*_ppo_overrides(), "trainer.num_envs=1"])
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def _tdmpc2_overrides() -> list[str]:
    # 120 frames in 40-frame batches: 1 warm-up batch, then a 1-update pretrain
    # burst and 1-update batches. Tiny model (dims divisible by simnorm_dim=8)
    # and a shrunk MPPI keep the run to seconds on CPU. 40 seed frames within a
    # single 500-step trajectory guarantee horizon-3 slices exist.
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=120",
        "trainer.log_every_n_steps=40",
        "algorithm.compile=false",
        "algorithm.frames_per_batch=40",
        "algorithm.init_random_frames=40",
        "algorithm.pretrain_updates=1",
        "algorithm.num_updates=1",
        "algorithm.batch_size=4",
        "algorithm.buffer_size=1000",
        "algorithm.latent_dim=64",
        "algorithm.enc_dim=32",
        "algorithm.mlp_dim=32",
        "algorithm.num_q=2",
        "algorithm.num_samples=32",
        "algorithm.num_elites=4",
        "algorithm.num_pi_trajs=2",
        "algorithm.iterations=1",
        "checkpoint.enabled=false",
    ]


def test_smoke_tdmpc2_cheetah_run():
    """TD-MPC2 on dm_control cheetah-run: world model, MPPI planning, slice buffer."""
    pytest.importorskip("dm_control")  # dm_control is an optional system dep
    cfg = load_experiment_cfg("tdmpc2/dmc", _tdmpc2_overrides())
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
    cfg = load_experiment_cfg("a2c/gym", _a2c_overrides())
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def _der_overrides() -> list[str]:
    # 40 frames in 4-frame batches: 2 warm-up batches then 8 update batches.
    # batch_size=4/num_updates=1 keeps sampling cheap; replay_capacity=200 and
    # n_steps=3 keep MultiStepTransform's internal per-episode buffer small.
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=40",
        "trainer.log_every_n_steps=8",
        "algorithm.frames_per_batch=4",
        "algorithm.init_random_frames=8",
        "algorithm.batch_size=4",
        "algorithm.num_updates=1",
        "algorithm.replay_capacity=200",
        "algorithm.n_steps=3",
    ]


def test_smoke_der_jamesbond():
    """DER on ALE/Jamesbond-v5 (Atari-100k): C51 + noisy nets + prioritized replay."""
    pytest.importorskip("ale_py")  # ALE is an optional system dep
    cfg = load_experiment_cfg("rainbow/atari100k", _der_overrides())
    from src.train import _train

    metrics = _train(cfg)
    assert isinstance(metrics, dict)
    assert len(metrics) > 0


def _bbf_overrides() -> list[str]:
    # 40 frames in 1-frame batches: 8 warm-up frames then 32 update steps
    # (replay_ratio=1). Tiny Impala (width_scale=1, hidden_dim=64), short
    # window (max_update_horizon=3, spr_depth=2 -> slice_len=4) and a 200-step
    # ring keep it to seconds on CPU. reset_interval=12 exercises the
    # shrink-and-perturb path (~2 resets) without dominating the runtime.
    return [
        *BASE_OVERRIDES,
        "trainer.total_frames=40",
        "trainer.log_every_n_steps=8",
        "algorithm.min_replay_history=8",
        "algorithm.batch_size=2",
        "algorithm.replay_ratio=1",
        "algorithm.replay_capacity=200",
        "algorithm.max_update_horizon=3",
        "algorithm.min_update_horizon=1",
        "algorithm.spr_depth=2",
        "algorithm.width_scale=1",
        "algorithm.hidden_dim=64",
        "algorithm.reset_interval=12",
        "algorithm.eps_annealing_frames=8",
    ]


def test_smoke_bbf_atari100k():
    """BBF on ALE/Jamesbond-v5 (Atari-100k): hand-written C51 + SPR + resets +
    annealed n-step/discount over a torchrl PrioritizedSliceSampler buffer."""
    pytest.importorskip("ale_py")  # ALE is an optional system dep
    cfg = load_experiment_cfg("bbf/atari100k", _bbf_overrides())
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


def test_smoke_dreamer_jamesbond():
    """DreamerV3 on ALE/Jamesbond-v5: pixel obs, RSSM world model, actor-critic."""
    pytest.importorskip("ale_py")
    cfg = load_experiment_cfg("dreamer/atari100k", _dreamer_overrides())
    from src.train import _train

    metrics = _train(cfg)
    # DreamerAlgorithm.step() always returns {} — metrics are accumulated
    # internally and flushed via pop_train_metrics() to logger callbacks.
    # Just verify the run completed without raising.
    assert isinstance(metrics, dict)
