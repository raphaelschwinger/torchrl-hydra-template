"""Evaluation entry point.

Usage:
    python src/eval.py experiment=dqn/cartpole checkpoint.resume_from=logs/.../last.pt
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="eval", version_base="1.3")
def evaluate(cfg: DictConfig) -> None:
    results = _evaluate(cfg)
    print("\nEvaluation results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")


def _evaluate(cfg: DictConfig) -> dict[str, float]:
    from hydra.utils import get_class
    from omegaconf import OmegaConf

    from src.environments.environment import Environment
    from src.utils.seeding import seed_everything

    seed_everything(int(cfg.trainer.seed))

    env_kwargs = {k: v for k, v in OmegaConf.to_container(cfg.environment, resolve=True).items()
                  if k != "_target_"}
    environment = Environment(**env_kwargs)

    AlgClass = get_class(cfg.algorithm._target_)
    alg_kwargs = {k: v for k, v in OmegaConf.to_container(cfg.algorithm, resolve=True).items()
                  if k != "_target_"}
    algorithm = AlgClass(device=None, **alg_kwargs)

    TrainerClass = get_class(cfg.trainer._target_)
    trainer = TrainerClass(
        cfg=cfg,
        algorithm=algorithm,
        environment=environment,
    )

    trainer.setup()
    trainer.load_checkpoint(cfg.checkpoint.resume_from)

    return trainer.evaluate(num_episodes=int(cfg.trainer.num_eval_episodes))


if __name__ == "__main__":
    evaluate()
