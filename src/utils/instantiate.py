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


def build_environment(env_cfg: DictConfig):
    """Build an ``Environment`` from a flat env config.

    Environments stay ``to_container`` + ``**kwargs`` (unlike algorithms, which
    go through ``instantiate`` so their nested ``_partial_`` factories become
    real callables).
    """
    from omegaconf import OmegaConf

    from src.environments import Environment

    kwargs = {
        k: v
        for k, v in OmegaConf.to_container(env_cfg, resolve=True).items()
        if k != "_target_"
    }
    return Environment(**kwargs)


def build_trainer(cfg: DictConfig) -> BaseTrainer:
    """Compose algorithm, environments, loggers and callbacks into a trainer.

    Shared by ``src/train.py`` and ``src/eval.py`` so the two entry points
    cannot drift apart — they differ only in which stages they run.
    """
    from hydra.utils import get_class, instantiate

    environment = build_environment(cfg.environment)

    # The evaluation component owns the eval env stack; ``null`` there means
    # "measure on the training env config" (a fresh instance of it).
    eval_env_cfg = (cfg.get("evaluation") or {}).get("eval_environment")
    eval_environment = (
        build_environment(eval_env_cfg) if eval_env_cfg is not None else None
    )

    algorithm = instantiate(cfg.algorithm, device=None)  # Trainer sets device
    loggers = build_loggers(cfg.get("logger") or [])

    TrainerClass = get_class(cfg.trainer._target_)
    trainer = TrainerClass(
        cfg=cfg,
        algorithm=algorithm,
        environment=environment,
        eval_environment=eval_environment,
    )
    trainer.callbacks = build_callbacks(
        cfg.trainer, cfg.get("checkpoint") or {}, trainer, loggers
    )
    return trainer


def build_callbacks(
    trainer_cfg: DictConfig,
    checkpoint_cfg: DictConfig,
    trainer: BaseTrainer,
    loggers: list,
) -> list:
    """Assemble the full callback list for a training run.

    Always includes ProgressCallback. CheckpointCallback is added only when
    ``checkpoint.enabled`` is true. Logger callbacks are appended after.

    Args:
        trainer_cfg: trainer sub-config (contains total_frames, log_every_n_steps)
        checkpoint_cfg: checkpoint sub-config (enabled, save_dir, save_every_n_steps, save_last)
        trainer: the trainer instance (injected into CheckpointCallback)
        loggers: pre-instantiated logger callback objects

    Returns:
        ordered list of callbacks
    """
    from src.callbacks.checkpoint import CheckpointCallback
    from src.callbacks.progress import ProgressCallback

    callbacks: list = [
        ProgressCallback(total_steps=trainer_cfg.total_frames),
    ]

    if checkpoint_cfg.get("enabled", False):
        checkpoint_cb = CheckpointCallback(
            save_dir=checkpoint_cfg.save_dir,
            save_every_n_steps=checkpoint_cfg.save_every_n_steps,
            save_last=checkpoint_cfg.save_last,
        )
        checkpoint_cb.set_trainer(trainer)
        callbacks.append(checkpoint_cb)

    callbacks.extend(loggers)
    return callbacks
