"""Eval entry point: load a checkpoint and run greedy rollouts.

Usage::

    python -m src.eval experiment=dqn/cartpole \
        checkpoint=outputs/<run>/trainer_<step>.pt num_episodes=10

Picks the policy out of the trainer's loss module (DQN: ``value_network``;
PPO/REINFORCE: ``actor_network``) so this works for any algorithm.
"""

from __future__ import annotations

import statistics
from pathlib import Path

import hydra
from omegaconf import DictConfig
from torchrl.trainers.algorithms.configs import *  # noqa: F401,F403  (registers ConfigStore entries)

from src.algorithms.reinforce.configs import *  # noqa: F401,F403
from src.train import _register_atari_envs
from src.utils.seeding import seed_everything

_register_atari_envs()


def _resolve_policy(trainer):
    loss_module = trainer.loss_module
    for name in ("actor_network", "value_network"):
        net = getattr(loss_module, name, None)
        if net is not None:
            return net
    raise RuntimeError(
        "Could not find an actor or value network on the loss module — "
        "this eval helper only supports algorithms that expose one of those."
    )


@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    if cfg.get("seed") is not None:
        seed_everything(int(cfg.seed))

    checkpoint = cfg.get("checkpoint")
    num_episodes = int(cfg.get("num_episodes", 10))
    if checkpoint is None:
        raise SystemExit("Pass `checkpoint=<path>` to load a saved trainer file.")
    if not Path(checkpoint).exists():
        raise SystemExit(f"Checkpoint not found: {checkpoint}")

    trainer = hydra.utils.instantiate(cfg.trainer)
    # Use strict=False to tolerate small state-dict drift (e.g. value-estimator
    # buffers added/removed across TorchRL versions).
    import torch

    state = torch.load(checkpoint, weights_only=False)
    if "loss_module" in state:
        trainer.loss_module.load_state_dict(state["loss_module"], strict=False)
    elif isinstance(state, dict) and "model_state_dict" in state:
        trainer.loss_module.load_state_dict(state["model_state_dict"], strict=False)
    policy = _resolve_policy(trainer)

    eval_env = hydra.utils.instantiate(cfg.training_env)
    returns: list[float] = []
    for _ in range(num_episodes):
        rollout = eval_env.rollout(max_steps=10_000, policy=policy, auto_reset=True)
        returns.append(float(rollout["next", "reward"].sum()))

    print(
        f"episodes={num_episodes} "
        f"mean={statistics.mean(returns):.3f} "
        f"std={statistics.pstdev(returns):.3f} "
        f"min={min(returns):.3f} max={max(returns):.3f}"
    )


if __name__ == "__main__":
    main()
