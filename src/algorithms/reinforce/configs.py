"""Hydra ConfigStore registration for REINFORCE.

Importing this module registers a structured config under
``trainer/reinforce`` so YAML configs can use::

    defaults:
      - /trainer@trainer: reinforce

The actual factory lives in :mod:`src.algorithms.reinforce.trainer`. We also
register a matching ``loss/reinforce`` so the loss YAML can attach a schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from torchrl.trainers.algorithms.configs.common import ConfigBase
from torchrl.trainers.algorithms.configs.objectives import LossConfig
from torchrl.trainers.algorithms.configs.trainers import TrainerConfig


@dataclass
class REINFORCETrainerConfig(TrainerConfig):
    """Configuration for the REINFORCE trainer factory."""

    collector: Any = MISSING
    total_frames: int | None = None
    optim_steps_per_batch: int | None = 1
    loss_module: Any = MISSING
    optimizer: Any = MISSING
    logger: Any = None
    save_trainer_file: Any = None
    frame_skip: int = 1
    clip_grad_norm: bool = True
    clip_norm: float | None = None
    progress_bar: bool = True
    seed: int | None = None
    save_trainer_interval: int = 10_000
    log_interval: int = 10_000
    create_env_fn: Any = None
    actor_network: Any = MISSING
    gamma: float = 0.99
    _target_: str = "src.algorithms.reinforce.trainer.make_reinforce_trainer"


@dataclass
class ReinforceLossConfig(LossConfig):
    """Configuration for ``torchrl.objectives.ReinforceLoss``."""

    actor_network: Any = MISSING
    critic_network: Any = None
    delay_value: bool = False
    loss_critic_type: str = "smooth_l1"
    gamma: float | None = None
    advantage_key: str | None = None
    value_target_key: str | None = None
    separate_losses: bool = False
    functional: bool = True
    reduction: str | None = None
    clip_value: float | None = None
    _partial_: bool = True
    _target_: str = "torchrl.objectives.ReinforceLoss"


cs = ConfigStore.instance()
cs.store(group="trainer", name="reinforce", node=REINFORCETrainerConfig)
cs.store(group="loss", name="reinforce", node=ReinforceLossConfig)


__all__ = ["REINFORCETrainerConfig", "ReinforceLossConfig"]
