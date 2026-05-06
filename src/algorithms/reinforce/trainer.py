"""REINFORCE trainer factory.

TorchRL ships per-algorithm trainers for DQN, PPO, SAC, TD3, DDPG, CQL, IQL,
but not REINFORCE. This module shows how to wire a new algorithm into the
template: write a factory that returns a configured ``torchrl.trainers.Trainer``
and register a matching config dataclass with Hydra's ConfigStore (see
``src.algorithms.reinforce.configs``).

REINFORCE is on-policy Monte-Carlo policy gradient: collect a batch of
trajectories, compute returns, take one gradient step, discard the batch.
"""

from __future__ import annotations

from typing import Any

import torch
from torchrl.collectors.collectors import DataCollectorBase
from torchrl.objectives import LossModule
from torchrl.trainers import Trainer


def make_reinforce_trainer(
    *,
    collector: Any,
    total_frames: int | None,
    loss_module: Any,
    optimizer: Any,
    logger: Any,
    actor_network: Any,
    gamma: float = 0.99,
    frame_skip: int = 1,
    optim_steps_per_batch: int | None = 1,
    clip_grad_norm: bool = True,
    clip_norm: float | None = None,
    progress_bar: bool = True,
    seed: int | None = None,
    save_trainer_interval: int = 10_000,
    log_interval: int = 10_000,
    save_trainer_file: str | None = None,
    create_env_fn: Any = None,  # noqa: ARG001 (referenced by ${collector} only)
) -> Trainer:
    """Build a TorchRL ``Trainer`` configured for REINFORCE.

    Each argument may be a fully-instantiated object or a Hydra ``_partial_``
    that we finalize here (mirrors the pattern in
    ``torchrl.trainers.algorithms.configs.trainers._make_ppo_trainer``).
    """

    if actor_network is not None and not isinstance(actor_network, torch.nn.Module):
        actor_network = actor_network()

    if not isinstance(collector, DataCollectorBase):
        collector = collector()

    if not isinstance(loss_module, LossModule):
        loss_module = loss_module(actor_network=actor_network)
    loss_module.make_value_estimator(gamma=gamma)

    if not isinstance(optimizer, torch.optim.Optimizer):
        optimizer = optimizer(params=loss_module.parameters())

    if total_frames is None:
        total_frames = collector.total_frames

    return Trainer(
        collector=collector,
        total_frames=total_frames,
        frame_skip=frame_skip,
        optim_steps_per_batch=optim_steps_per_batch,
        loss_module=loss_module,
        optimizer=optimizer,
        logger=logger,
        clip_grad_norm=clip_grad_norm,
        clip_norm=clip_norm,
        progress_bar=progress_bar,
        seed=seed,
        save_trainer_interval=save_trainer_interval,
        log_interval=log_interval,
        save_trainer_file=save_trainer_file,
    )
