import hydra
from torchrl.trainers.algorithms.configs import *  # noqa: F401,F403  (registers Hydra ConfigStore entries)

from src.algorithms.reinforce.configs import *  # noqa: F401,F403  (registers REINFORCETrainerConfig)
from src.utils.seeding import seed_everything


def _register_atari_envs() -> None:
    """Register ALE Atari environments with Gymnasium.

    Gymnasium 0.29+ no longer auto-registers Atari, so envs like
    ``PongNoFrameskip-v4`` are unknown until ``ale_py`` is imported and its
    envs registered explicitly.
    """
    try:
        import ale_py
        import gymnasium as gym

        gym.register_envs(ale_py)
    except ImportError:
        pass


_register_atari_envs()


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg):
    if cfg.get("seed") is not None:
        seed_everything(int(cfg.seed))
    trainer = hydra.utils.instantiate(cfg.trainer)
    trainer.train()


if __name__ == "__main__":
    main()
