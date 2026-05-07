"""Proximal Policy Optimization (PPO) algorithm.

Compatible environments: continuous-action dm_control environments (humanoid-walk).

Architecture:
  - Actor: MLP → Normal distribution via NormalParamExtractor + TanhNormal
  - Critic: separate MLP → scalar value estimate
  - Loss: ClipPPOLoss (clipped surrogate + value + entropy)
  - Advantage: GAE (Generalized Advantage Estimation)
  - Trainer: StepTrainer (SyncDataCollector over ParallelEnv)
  - No replay buffer (on-policy)
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.envs import EnvBase

from src.algorithms.base import BaseAlgorithm, CollectorConfig, TrainingState
from src.networks.factory import make_network


class PPOAlgorithm(BaseAlgorithm):
    """PPO with clipped surrogate objective, GAE, and parallel environment collection.

    Defaults are tuned for continuous-action dm_control environments (e.g. humanoid-walk).
    Override any parameter in the algorithm YAML or experiment config.

    Args:
        cfg: Full Hydra config (trainer, logger, environment sections).
        device: Resolved torch.device; set by the Trainer.
        frames_per_batch: Total frames per rollout across all envs.
        epochs_per_batch: Number of gradient passes over each collected batch.
        minibatch_size: Mini-batch size within each epoch.
        clip_epsilon: Clipping parameter for the surrogate objective.
        entropy_coef: Entropy bonus coefficient.
        critic_coef: Value loss coefficient relative to policy loss.
        gamma: Discount factor for future rewards.
        lmbda: GAE lambda for advantage estimation.
        lr: Adam optimizer learning rate.
        max_grad_norm: Gradient clipping threshold (L2 norm).
        network: Dict with keys ``architecture``, ``hidden_sizes``, ``activation``,
            ``layer_norm``. Shared backbone for actor and critic.
    """

    def __init__(
        self,
        cfg: DictConfig,
        device: torch.device | None = None,
        *,
        frames_per_batch: int = 2_048,
        epochs_per_batch: int = 10,
        minibatch_size: int = 64,
        clip_epsilon: float = 0.2,
        entropy_coef: float = 0.01,
        critic_coef: float = 0.5,
        gamma: float = 0.99,
        lmbda: float = 0.95,
        lr: float = 3e-4,
        max_grad_norm: float = 0.5,
        network: dict | None = None,
    ) -> None:
        super().__init__(cfg, device)
        self.frames_per_batch = frames_per_batch
        self.epochs_per_batch = epochs_per_batch
        self.minibatch_size = minibatch_size
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.critic_coef = critic_coef
        self.gamma = gamma
        self.lmbda = lmbda
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self._network_cfg = network or {
            "architecture": "mlp", "hidden_sizes": [256, 256],
            "activation": "tanh", "layer_norm": False,
        }

    def setup(self, make_env: Callable[[], EnvBase]) -> None:
        from torchrl.modules import (
            NormalParamExtractor,
            ProbabilisticActor,
            TanhNormal,
            ValueOperator,
        )
        from torchrl.objectives import ClipPPOLoss
        from torchrl.objectives.value import GAE

        env = make_env()
        ecfg = self.cfg.environment
        obs_shape = tuple(ecfg.obs_shape)
        num_actions = int(ecfg.num_actions)
        net_cfg = OmegaConf.create(self._network_cfg)

        # --- Actor backbone → outputs 2 * num_actions (mean + log_std) ---
        actor_net = make_network(net_cfg, obs_shape, num_actions * 2).to(self.device)
        actor_module = TensorDictModule(
            nn.Sequential(actor_net, NormalParamExtractor()),
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        )

        action_spec = env.action_spec
        self.actor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=["action"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": action_spec.space.low,
                "high": action_spec.space.high,
            },
            return_log_prob=True,
        ).to(self.device)

        # --- Critic ---
        critic_net = make_network(net_cfg, obs_shape, 1).to(self.device)
        self.critic = ValueOperator(
            module=critic_net,
            in_keys=["observation"],
        ).to(self.device)

        # --- Advantage module (GAE) ---
        self.gae = GAE(
            gamma=self.gamma,
            lmbda=self.lmbda,
            value_network=self.critic,
            average_gae=False,
        )

        # --- PPO loss ---
        self.loss_module = ClipPPOLoss(
            actor_network=self.actor,
            critic_network=self.critic,
            clip_epsilon=self.clip_epsilon,
            entropy_bonus=True,
            entropy_coeff=self.entropy_coef,
            critic_coeff=self.critic_coef,
            normalize_advantage=True,
            loss_critic_type="smooth_l1",
        ).to(self.device)

        # --- Single optimizer for actor + critic ---
        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.lr,
        )

    # ------------------------------------------------------------------
    # Policy access
    # ------------------------------------------------------------------

    def get_policy(self) -> TensorDictModule:
        return self.actor

    def get_explore_policy(self) -> TensorDictModule:
        return self.actor  # stochastic actor is the exploration policy

    # ------------------------------------------------------------------
    # Collector configuration
    # ------------------------------------------------------------------

    def get_collector_config(self) -> CollectorConfig:
        return CollectorConfig(
            frames_per_batch=self.frames_per_batch,
            total_frames=int(self.cfg.trainer.total_frames),
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def step(self, batch: TensorDict) -> dict[str, float]:
        """Compute GAE, then multi-epoch minibatch PPO updates."""
        # Compute GAE advantages in-place
        with torch.no_grad():
            self.gae(batch)

        # Flatten (num_envs x time) → single batch dimension
        data = batch.reshape(-1)
        batch_size = data.batch_size[0]

        metrics: dict[str, float] = {}
        for _ in range(self.epochs_per_batch):
            perm = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, self.minibatch_size):
                idx = perm[start : start + self.minibatch_size]
                if len(idx) < 2:
                    continue
                mb = data[idx]

                loss_td = self.loss_module(mb)
                loss = (
                    loss_td["loss_objective"]
                    + loss_td["loss_critic"]
                    + loss_td["loss_entropy"]
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()

                metrics = {
                    "loss/total": loss.item(),
                    "loss/policy": loss_td["loss_objective"].item(),
                    "loss/value": loss_td["loss_critic"].item(),
                    "loss/entropy": loss_td["loss_entropy"].item(),
                }

        return metrics

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _get_training_state(self) -> TrainingState:
        return TrainingState(
            step=0,
            policy_state_dict={
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
            },
            optimizer_state_dict=self.optimizer.state_dict(),
        )

    def _load_training_state(self, state: TrainingState) -> None:
        self.actor.load_state_dict(state.policy_state_dict["actor"])
        self.critic.load_state_dict(state.policy_state_dict["critic"])
        self.optimizer.load_state_dict(state.optimizer_state_dict)
