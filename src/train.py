"""Training entry point.

Usage:
    python src/train.py experiment=reinforce/cartpole
    python src/train.py experiment=dqn/gym logger=[wandb,tensorboard]
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
        dict of final training metrics (plus ``eval/*`` when ``eval: true``)
    """
    from src.utils.instantiate import build_trainer
    from src.utils.seeding import seed_everything

    seed_everything(int(cfg.trainer.seed))

    trainer = build_trainer(cfg)

    # Setup (creates env, builds networks, creates collector if StepTrainer)
    trainer.setup()

    # Optionally resume from a checkpoint
    if cfg.checkpoint.get("resume_from") is not None:
        trainer.load_checkpoint(cfg.checkpoint.resume_from)

    return trainer.run(
        train=bool(cfg.get("train", True)),
        evaluate=bool(cfg.get("eval", True)),
    )


if __name__ == "__main__":
    train()
