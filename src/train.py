"""Training entry point.

Usage:
    python src/train.py experiment=reinforce/cartpole
    python src/train.py experiment=dqn/cartpole logger=[wandb,tensorboard]
    python src/train.py experiment=dqn/atari_breakout trainer.accelerator=gpu trainer.devices=[0]
    python src/train.py experiment=ppo/dmc_humanoid trainer.accelerator=gpu
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def train(cfg: DictConfig) -> None:
    _train(cfg)


def _train(cfg: DictConfig) -> dict[str, float]:
    """Separated from the Hydra decorator for testability.

    Args:
        cfg: fully composed Hydra config

    Returns:
        dict of final training metrics
    """
    from hydra.utils import get_class, instantiate
    from omegaconf import OmegaConf

    from src.environments.environment import Environment
    from src.utils.instantiate import build_callbacks, build_loggers
    from src.utils.seeding import seed_everything

    # Must run before seed_everything() (the first CUDA call in the process)
    # — see perf_flags.configure_env's docstring for why this can't wait
    # until the Dreamer model is constructed.
    algo_target = str(cfg.algorithm.get("_target_", ""))
    if algo_target.startswith("src.algorithms.dreamer."):
        from src.algorithms.dreamer.perf_flags import configure_env

        configure_env()

    seed_everything(int(cfg.trainer.seed))

    # Build components
    env_kwargs = {k: v for k, v in OmegaConf.to_container(cfg.environment, resolve=True).items()
                  if k != "_target_"}
    environment = Environment(**env_kwargs)

    eval_env_cfg = cfg.get("eval_environment")
    eval_environment = None
    if eval_env_cfg is not None:
        eval_kwargs = {k: v for k, v in OmegaConf.to_container(eval_env_cfg, resolve=True).items()
                       if k != "_target_"}
        eval_environment = Environment(**eval_kwargs)

    algorithm = instantiate(cfg.algorithm, device=None)  # Trainer sets device

    loggers = build_loggers(cfg.get("logger") or [])

    # Select and create trainer
    TrainerClass = get_class(cfg.trainer._target_)
    trainer = TrainerClass(
        cfg=cfg,
        algorithm=algorithm,
        environment=environment,
        eval_environment=eval_environment,
    )

    # Build callbacks (references trainer for checkpointing)
    callbacks = build_callbacks(cfg.trainer, cfg.checkpoint, trainer, loggers)
    trainer.callbacks = callbacks

    # Setup (creates env, builds networks, creates collector if StepTrainer)
    trainer.setup()

    # Optionally resume from a checkpoint
    if cfg.checkpoint.get("resume_from") is not None:
        trainer.load_checkpoint(cfg.checkpoint.resume_from)

    return trainer.fit()


if __name__ == "__main__":
    train()
