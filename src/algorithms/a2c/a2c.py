"""Synchronous Advantage Actor-Critic (A2C).

Mnih et al. (2016), "Asynchronous Methods for Deep Reinforcement Learning."
https://arxiv.org/abs/1602.01783 (this is the synchronous variant.)

Pseudocode:
    Initialise policy pi(.|s; theta) and value V(s; phi)
    For each iteration:
        Roll out N steps with pi -> trajectory (s_t, a_t, r_t, s_{t+1})
        Compute GAE advantages A_t and value targets V^target_t
        For each minibatch (single epoch, sampled without replacement):
            Maximise log pi(a_t|s_t) * A_t + beta * H[pi(.|s_t)]
            Minimise (V(s_t; phi) - V^target_t)^2
            One backward + step with summed (actor + critic) loss

Defaults match the torchrl SOTA reference for HalfCheetah-v4
(https://github.com/pytorch/rl/blob/main/sota-implementations/a2c/a2c_mujoco.py).
"""
from __future__ import annotations

import functools
from typing import Callable

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import NormalParamExtractor, TensorDictModule
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import A2CLoss
from torchrl.objectives.value import GAE

from src.algorithms.base import BaseAlgorithm, CollectorConfig, TrainingState
from src.components.networks import make_mlp_a2c_actor, make_mlp_a2c_value


class A2CAlgorithm(BaseAlgorithm):
    """Synchronous Advantage Actor-Critic for continuous control.

    On-policy: each collected rollout is consumed in a single epoch of
    mini-batch updates (sampled without replacement) and discarded. No
    long-term replay buffer, no target networks, no warm-up phase.

    Defaults are tuned for HalfCheetah-v4 and mirror the torchrl SOTA reference
    (sota-implementations/a2c/config_mujoco.yaml).
    """

    def __init__(
        self,
        device: torch.device | None = None,
        *,
        # --- Design choices: factories injected as Callables ---------------
        # Factories called as ``factory(obs_shape, action_dim)`` in ``setup``.
        # The actor body outputs ``2 * action_dim`` features, split by
        # ``NormalParamExtractor`` into ``loc`` and ``scale`` for TanhNormal.
        actor_network: Callable[[tuple[int, ...], int], nn.Module] = functools.partial(
            make_mlp_a2c_actor,
            num_cells=[64, 64],
            activation_class=nn.Tanh,
        ),
        value_network: Callable[[tuple[int, ...], int], nn.Module] = functools.partial(
            make_mlp_a2c_value,
            num_cells=[64, 64],
            activation_class=nn.Tanh,
        ),
        obs_key: str = "observation",
        # --- Optimisation --------------------------------------------------
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        entropy_coeff: float = 0.0,
        critic_coeff: float = 0.25,
        loss_critic_type: str = "l2",
        max_grad_norm: float = 1.0,
        mini_batch_size: int = 64,
        # --- Data collection ----------------------------------------------
        frames_per_batch: int = 640,
        max_frames_per_traj: int = -1,
    ) -> None:
        super().__init__(device)
        self._make_actor_network = actor_network
        self._make_value_network = value_network
        self.obs_key = obs_key
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coeff = entropy_coeff
        self.critic_coeff = critic_coeff
        self.loss_critic_type = loss_critic_type
        self.max_grad_norm = max_grad_norm
        self.mini_batch_size = mini_batch_size
        self.frames_per_batch = frames_per_batch
        self.max_frames_per_traj = max_frames_per_traj

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, make_env: Callable[[], EnvBase]) -> None:
        # Read env specs from a short-lived proof environment.
        proof_env = make_env()
        obs_shape = tuple(proof_env.observation_spec[self.obs_key].shape)
        action_spec = proof_env.action_spec
        action_dim = int(action_spec.shape[-1])

        # 1. Stochastic actor: MLP -> NormalParamExtractor -> TanhNormal.
        actor_mlp = self._make_actor_network(obs_shape, action_dim).to(self.device)
        actor_module = TensorDictModule(
            nn.Sequential(actor_mlp, NormalParamExtractor()),
            in_keys=[self.obs_key],
            out_keys=["loc", "scale"],
        )
        self.actor = ProbabilisticActor(
            module=actor_module,
            spec=action_spec,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            distribution_kwargs={
                "low": action_spec.space.low,
                "high": action_spec.space.high,
                "tanh_loc": False,
            },
            return_log_prob=True,
            default_interaction_type=ExplorationType.RANDOM,
        ).to(self.device)

        # 2. State-value critic. ``ValueOperator`` writes ``state_value``,
        #    the key both ``GAE`` and ``A2CLoss`` consume by default.
        value_mlp = self._make_value_network(obs_shape, action_dim).to(self.device)
        self.critic = ValueOperator(
            module=value_mlp,
            in_keys=[self.obs_key],
        ).to(self.device)

        # 3. A2C loss (policy-gradient + entropy bonus + critic MSE).
        self.loss_module = A2CLoss(
            actor_network=self.actor,
            critic_network=self.critic,
            loss_critic_type=self.loss_critic_type,
            entropy_coeff=self.entropy_coeff,
            critic_coeff=self.critic_coeff,
        )

        # 4. GAE advantage estimator. Applied to the rollout once per
        #    iteration (under no_grad) so its outputs feed the loss as fixed
        #    targets, not gradient-tracked tensors.
        self.adv_module = GAE(
            gamma=self.gamma,
            lmbda=self.gae_lambda,
            value_network=self.critic,
            average_gae=False,
            device=self.device,
        )

        # 5. On-policy mini-batch buffer: holds exactly one rollout, sampled
        #    without replacement so a single epoch covers each sample once.
        self.data_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(self.frames_per_batch, device=self.device),
            sampler=SamplerWithoutReplacement(),
            batch_size=self.mini_batch_size,
        )

        # 6. Single Adam over actor + critic params (disjoint sets).
        self.optimizer = torch.optim.Adam(
            self.loss_module.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    def get_collector_config(self) -> CollectorConfig:
        return CollectorConfig(
            frames_per_batch=self.frames_per_batch,
            init_random_frames=0,
            max_frames_per_traj=self.max_frames_per_traj,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def step(self, batch: TensorDict) -> dict[str, float]:
        """One collector iteration: GAE -> single epoch of mini-batch updates."""
        batch = batch.reshape(-1)

        # Compute GAE advantages and value targets in-place on the rollout.
        with torch.no_grad():
            batch = self.adv_module(batch)

        # Reset and load the on-policy buffer for this iteration only.
        self.data_buffer.empty()
        self.data_buffer.extend(batch)

        n = max(1, batch.numel() // self.mini_batch_size)
        crit = torch.zeros(n, device=self.device)
        obj = torch.zeros(n, device=self.device)
        ent = torch.zeros(n, device=self.device)

        for j, mb in enumerate(self.data_buffer):
            mb = mb.to(self.device)
            loss_td = self.loss_module(mb)
            loss = loss_td["loss_objective"] + loss_td["loss_critic"]
            if "loss_entropy" in loss_td.keys():
                loss = loss + loss_td["loss_entropy"]

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(self.loss_module.parameters(), self.max_grad_norm)
            self.optimizer.step()

            if j < n:
                crit[j] = loss_td["loss_critic"].detach()
                obj[j] = loss_td["loss_objective"].detach()
                if "loss_entropy" in loss_td.keys():
                    ent[j] = loss_td["loss_entropy"].detach()

        return {
            "train/loss_critic": crit.mean().item(),
            "train/loss_objective": obj.mean().item(),
            "train/loss_entropy": ent.mean().item(),
        }

    # ------------------------------------------------------------------
    # Policy access
    # ------------------------------------------------------------------

    def get_policy(self) -> TensorDictModule:
        # The trainer wraps eval rollouts in ``set_exploration_type(MODE)``,
        # which makes the same actor return the distribution mode (deterministic).
        return self.actor

    def get_explore_policy(self) -> TensorDictModule:
        # Default interaction type is RANDOM, so collection samples stochastically.
        return self.actor

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _get_training_state(self) -> TrainingState:
        return TrainingState(
            step=0,
            policy_state_dict=self.loss_module.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
        )

    def _load_training_state(self, state: TrainingState) -> None:
        self.loss_module.load_state_dict(state.policy_state_dict)
        self.optimizer.load_state_dict(state.optimizer_state_dict)
