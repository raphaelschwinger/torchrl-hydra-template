from __future__ import annotations

from typing import TYPE_CHECKING

from hydra.utils import instantiate
from omegaconf import DictConfig, ListConfig

if TYPE_CHECKING:
    from src.trainers import BaseTrainer


def build_loggers(logger_cfgs: ListConfig | list) -> list:
    """Instantiate all logger callbacks from a Hydra list config.

    Args:
        logger_cfgs: list of logger DictConfigs, each with a _target_ key.
                     An empty list means no logging.

    Returns:
        list of instantiated logger objects
    """
    return [instantiate(cfg) for cfg in logger_cfgs]


def build_callbacks(
    trainer_cfg: DictConfig,
    checkpoint_cfg: DictConfig,
    trainer: BaseTrainer,
    loggers: list,
) -> list:
    """Assemble the full callback list for a training run.

    Always includes ProgressCallback and CheckpointCallback.
    Logger callbacks are appended after.

    Args:
        trainer_cfg: trainer sub-config (contains total_frames, log_every_n_steps)
        checkpoint_cfg: checkpoint sub-config (save_dir, save_every_n_steps, save_last)
        trainer: the trainer instance (injected into CheckpointCallback)
        loggers: pre-instantiated logger callback objects

    Returns:
        ordered list of callbacks
    """
    from src.callbacks.checkpoint import CheckpointCallback
    from src.callbacks.progress import ProgressCallback

    checkpoint_cb = CheckpointCallback(
        save_dir=checkpoint_cfg.save_dir,
        save_every_n_steps=checkpoint_cfg.save_every_n_steps,
        save_last=checkpoint_cfg.save_last,
    )
    checkpoint_cb.set_trainer(trainer)

    return [
        ProgressCallback(total_steps=trainer_cfg.total_frames),
        checkpoint_cb,
        *loggers,
    ]
