"""Base algorithm interface.

Every RL algorithm inherits from :class:`BaseAlgorithm` and implements the
required methods.  The Trainer drives the training loop and calls these
methods — the algorithm never manages the loop itself.

Lifecycle (called by Trainer):
    1. ``__init__(cfg, device)`` — store config
    2. ``setup(make_env)`` — build networks, loss, optimizer (call ``make_env()`` for specs)
    3. Training loop — Trainer calls ``step(batch)`` repeatedly
    4. Checkpointing — ``save_checkpoint`` / ``load_checkpoint``
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs import EnvBase


@dataclass
class TrainingState:
    """Full snapshot of algorithm state for checkpointing and resuming."""

    step: int
    policy_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    extra: dict[str, Any] | None = field(default=None)


@dataclass
class CollectorConfig:
    """Parameters the algorithm provides to the Trainer for collector creation.

    Only used by :class:`StepTrainer`.  :class:`EpisodicTrainer` ignores this.
    """

    frames_per_batch: int
    init_random_frames: int = 0
    max_frames_per_traj: int = -1 # -1 means no limit


class BaseAlgorithm(ABC):
    """Abstract base class for all RL algorithms.

    Args:
        cfg: full Hydra config (algorithm params via ``cfg.algorithm``,
             environment params via ``cfg.environment``, etc.)
        device: resolved ``torch.device`` (set by the Trainer)
    """

    def __init__(self, device: torch.device) -> None:
        self.device = device

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @abstractmethod
    def setup(self, make_env: Callable[[], EnvBase]) -> None:
        """Build networks, loss module, and optimizer.

        Called once by the Trainer.  ``make_env`` is a zero-argument factory
        that returns a fresh environment; the algorithm calls it to read
        ``action_spec``, ``observation_spec``, etc.  The algorithm should
        **not** keep a long-lived reference to the env — the Trainer owns
        the env used for collection and evaluation.
        """

    # ------------------------------------------------------------------
    # Policy access (Trainer needs these for collection / eval)
    # ------------------------------------------------------------------

    @abstractmethod
    def get_policy(self) -> TensorDictModule:
        """Return the greedy / deterministic policy for evaluation."""

    @abstractmethod
    def get_explore_policy(self) -> TensorDictModule:
        """Return the exploration policy for data collection.

        For stochastic actors (REINFORCE, PPO) this is typically the same
        object as ``get_policy()``.  For DQN it wraps the Q-actor with
        an epsilon-greedy module.
        """

    # ------------------------------------------------------------------
    # Collector configuration (StepTrainer only)
    # ------------------------------------------------------------------

    def get_collector_config(self) -> CollectorConfig:
        """Return parameters the Trainer needs to create ``SyncDataCollector``.

        Only called by :class:`StepTrainer`.  Override in algorithms that use
        step-based collection (DQN, PPO).
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_collector_config() "
            "to be used with StepTrainer."
        )

    # ------------------------------------------------------------------
    # Primary training interface
    # ------------------------------------------------------------------

    @abstractmethod
    def step(self, batch: TensorDict) -> dict[str, float]:
        """Process one batch of collected data and return metrics.

        This is the main method the Trainer calls each iteration.

        For DQN:  sample from replay buffer → TD loss update.
        For PPO:  compute GAE → multi-epoch minibatch updates.
        For REINFORCE:  compute returns → policy gradient update.

        Args:
            batch: ``TensorDict`` from the collector (StepTrainer) or
                   ``env.rollout()`` (EpisodicTrainer).

        Returns:
            dict of scalar metrics (losses, Q-values, etc.)
        """

    # ------------------------------------------------------------------
    # Optional hooks — override for custom behaviour
    # ------------------------------------------------------------------

    def on_batch_collected(self, batch: TensorDict) -> TensorDict:
        """Called right after the Trainer collects a batch.

        Default: return batch unchanged.
        DQN overrides this to store transitions in its replay buffer.
        """
        return batch

    def should_skip_update(self, frames_collected: int) -> bool:
        """Return ``True`` to skip ``step()`` this iteration.

        Called by ``StepTrainer`` before ``step()``.
        DQN overrides this to skip during warmup.
        """
        return False

    def on_step_complete(self, frames_collected: int) -> None:
        """Called after ``step()`` completes.  For periodic maintenance.

        DQN overrides this for epsilon decay and target network updates.
        """

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: Path, step: int) -> None:
        """Serialize the current training state to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = self._get_training_state()
        state.step = step
        torch.save(state, path)

    def load_checkpoint(self, path: Path) -> int:
        """Restore training state from a checkpoint file.

        Returns:
            The step count stored in the checkpoint.
        """
        state: TrainingState = torch.load(
            path, map_location=self.device, weights_only=False
        )
        self._load_training_state(state)
        return state.step

    @abstractmethod
    def _get_training_state(self) -> TrainingState:
        """Collect current state dicts into a ``TrainingState``."""

    @abstractmethod
    def _load_training_state(self, state: TrainingState) -> None:
        """Restore network/optimizer/buffer state from a loaded ``TrainingState``."""
