"""Deep Deterministic Policy Gradient (DDPG).

Lillicrap et al. (2016), "Continuous control with deep reinforcement learning."
https://arxiv.org/abs/1509.02971

Pseudocode:
    Initialise replay buffer D
    Initialise online actor mu(s; theta_mu) and critic Q(s, a; theta_Q)
    Initialise target nets: theta_mu_target = theta_mu, theta_Q_target = theta_Q
    For each step:
        Select action a = mu(s; theta_mu) + N (exploration noise)
        Execute a, observe r and s'
        Store (s, a, r, s', done) in D
        Sample minibatch from D
        y = r + gamma * (1 - done) * Q_target(s', mu_target(s'))
        Update critic:  minimise (y - Q(s, a; theta_Q))^2
        Update actor:   maximise Q(s, mu(s; theta_mu); theta_Q)
        Polyak target update: theta_target <- tau * theta_online + (1 - tau) * theta_target

Defaults match the torchrl SOTA reference for HalfCheetah-v4
(https://github.com/pytorch/rl/blob/main/sota-implementations/ddpg/ddpg.py).
"""
from __future__ import annotations

import functools
from typing import Callable

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.data import LazyTensorStorage, ReplayBuffer, TensorDictReplayBuffer
from torchrl.envs import EnvBase
from torchrl.modules import OrnsteinUhlenbeckProcessModule, TanhModule, ValueOperator
from torchrl.objectives import DDPGLoss, SoftUpdate, group_optimizers

from src.algorithms.base import BaseAlgorithm, CollectorConfig, TrainingState
from src.components.networks import make_mlp_ddpg_actor, make_mlp_ddpg_critic


class DDPGAlgorithm(BaseAlgorithm):
    """DDPG with replay buffer, target nets and additive exploration noise.

    Defaults are tuned for HalfCheetah-v4 and mirror the torchrl SOTA reference
    (sota-implementations/ddpg/config.yaml).
    """

    def __init__(
        self,
        device: torch.device | None = None,
        *,
        # --- Design choices: factories injected as Callables ---------------
        replay_buffer: Callable[[], ReplayBuffer] = lambda: TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=1_000_000, device="cpu"),
        ),
        # Factories called as ``factory(obs_shape, action_dim)`` in ``setup``.
        actor_network: Callable[[tuple[int, ...], int], nn.Module] = functools.partial(
            make_mlp_ddpg_actor,
            num_cells=[256, 256],
            activation_class=nn.ReLU,
        ),
        value_network: Callable[[tuple[int, ...], int], nn.Module] = functools.partial(
            make_mlp_ddpg_critic,
            num_cells=[256, 256],
            activation_class=nn.ReLU,
        ),
        # Exploration noise factory; called as ``factory(spec=action_spec)``.
        exploration_noise: Callable[..., nn.Module] = functools.partial(
            OrnsteinUhlenbeckProcessModule,
            annealing_num_steps=1_000_000,
            safe=False,
        ),
        # Observation tensordict key (``"observation"`` for vector obs).
        obs_key: str = "observation",
        # --- Optimisation --------------------------------------------------
        lr_actor: float = 3e-4,
        lr_value: float = 3e-4,
        weight_decay: float = 0.0,
        gamma: float = 0.99,
        tau: float = 0.005,        # SoftUpdate uses ``tau`` directly (Polyak step size)
        batch_size: int = 256,
        max_grad_norm: float = 1.0,
        # --- Data collection ----------------------------------------------
        frames_per_batch: int = 1_000,
        init_random_frames: int = 25_000,
        max_frames_per_traj: int = -1,
        # --- Learning schedule --------------------------------------------
        num_updates: int = 1_000,  # gradient updates per collector batch (utd=1.0)
    ) -> None:
        super().__init__(device)
        self._make_replay_buffer = replay_buffer
        self._make_actor_network = actor_network
        self._make_value_network = value_network
        self._make_exploration_noise = exploration_noise
        self.obs_key = obs_key
        self.lr_actor = lr_actor
        self.lr_value = lr_value
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.frames_per_batch = frames_per_batch
        self.init_random_frames = init_random_frames
        self.max_frames_per_traj = max_frames_per_traj
        self.num_updates = num_updates
        self._collected_frames = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, make_env: Callable[[], EnvBase]) -> None:
        # Read env specs from a short-lived proof environment.
        proof_env = make_env()
        obs_shape = tuple(proof_env.observation_spec[self.obs_key].shape)
        action_spec = proof_env.action_spec
        action_dim = int(action_spec.shape[-1])

        # 1. Actor: MLP -> TanhModule that rescales to action_spec bounds.
        actor_mlp = self._make_actor_network(obs_shape, action_dim).to(self.device)
        self.actor = TensorDictSequential(
            TensorDictModule(actor_mlp, in_keys=[self.obs_key], out_keys=["param"]),
            TanhModule(in_keys=["param"], out_keys=["action"], spec=action_spec),
        ).to(self.device)

        # 2. Critic: ValueOperator concatenates [obs, action] along last dim
        #    and writes ``state_action_value`` (the key DDPGLoss reads).
        critic_mlp = self._make_value_network(obs_shape, action_dim).to(self.device)
        self.critic = ValueOperator(
            module=critic_mlp,
            in_keys=[self.obs_key, "action"],
        ).to(self.device)

        # 3. Exploration policy: actor + additive noise (OU by default).
        #    The noise module reads ``is_init`` to reset its state at episode
        #    boundaries, so the env must include an InitTracker transform.
        self.noise_module = self._make_exploration_noise(spec=action_spec).to(self.device)
        self._explore_policy = TensorDictSequential(self.actor, self.noise_module)

        # 4. Replay buffer.
        self.replay_buffer = self._make_replay_buffer()

        # 5. DDPG loss with delayed actor + critic target nets.
        self.loss_module = DDPGLoss(
            actor_network=self.actor,
            value_network=self.critic,
            loss_function="l2",
            delay_actor=True,
            delay_value=True,
        )
        self.loss_module.make_value_estimator(gamma=self.gamma)
        self.loss_module = self.loss_module.to(self.device)

        # 6. Soft target update with Polyak step ``tau``.
        self.target_updater = SoftUpdate(self.loss_module, tau=self.tau)

        # 7. Separate Adam optimisers (different learning rates allowed) grouped
        #    via ``group_optimizers`` so a single ``.zero_grad()`` / ``.step()``
        #    drives both. Mirrors the SOTA reference.
        self.optimizer_actor = torch.optim.Adam(
            self.loss_module.actor_network_params.values(True, True),
            lr=self.lr_actor,
            weight_decay=self.weight_decay,
        )
        self.optimizer_critic = torch.optim.Adam(
            self.loss_module.value_network_params.values(True, True),
            lr=self.lr_value,
            weight_decay=self.weight_decay,
        )
        self.optimizer = group_optimizers(self.optimizer_actor, self.optimizer_critic)

    def get_collector_config(self) -> CollectorConfig:
        return CollectorConfig(
            frames_per_batch=self.frames_per_batch,
            init_random_frames=self.init_random_frames,
            max_frames_per_traj=self.max_frames_per_traj,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def step(self, batch: TensorDict) -> dict[str, float]:
        """One collector iteration: anneal noise, store, optimise."""
        # Always: anneal exploration noise and store transitions.
        batch = batch.reshape(-1)
        self.noise_module.step(batch.numel())
        self.replay_buffer.extend(batch)
        self._collected_frames += batch.numel()

        # Warm-up: collect random transitions before any gradient update.
        if self._collected_frames < self.init_random_frames:
            return {}

        actor_losses = torch.zeros(self.num_updates, device=self.device)
        value_losses = torch.zeros(self.num_updates, device=self.device)
        pred_values = torch.zeros(self.num_updates, device=self.device)
        target_values = torch.zeros(self.num_updates, device=self.device)

        for j in range(self.num_updates):
            sample = self.replay_buffer.sample(self.batch_size).to(self.device)
            self.optimizer.zero_grad(set_to_none=True)
            loss_td = self.loss_module(sample)
            (loss_td["loss_actor"] + loss_td["loss_value"]).backward()
            nn.utils.clip_grad_norm_(
                list(self.loss_module.actor_network_params.values(True, True))
                + list(self.loss_module.value_network_params.values(True, True)),
                self.max_grad_norm,
            )
            self.optimizer.step()
            self.target_updater.step()

            actor_losses[j] = loss_td["loss_actor"].detach()
            value_losses[j] = loss_td["loss_value"].detach()
            pred_values[j] = loss_td["pred_value"].detach().mean()
            target_values[j] = loss_td["target_value"].detach().mean()

        return {
            "train/loss_actor": actor_losses.mean().item(),
            "train/loss_value": value_losses.mean().item(),
            "train/pred_value": pred_values.mean().item(),
            "train/target_value": target_values.mean().item(),
        }

    # ------------------------------------------------------------------
    # Policy access
    # ------------------------------------------------------------------

    def get_policy(self) -> TensorDictModule:
        return self.actor

    def get_explore_policy(self) -> TensorDictModule:
        return self._explore_policy

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _get_training_state(self) -> TrainingState:
        return TrainingState(
            step=0,
            policy_state_dict=self.loss_module.state_dict(),
            optimizer_state_dict={
                "actor": self.optimizer_actor.state_dict(),
                "critic": self.optimizer_critic.state_dict(),
            },
            extra={"collected_frames": self._collected_frames},
        )

    def _load_training_state(self, state: TrainingState) -> None:
        self.loss_module.load_state_dict(state.policy_state_dict)
        opt_state = state.optimizer_state_dict
        self.optimizer_actor.load_state_dict(opt_state["actor"])
        self.optimizer_critic.load_state_dict(opt_state["critic"])
        if state.extra and "collected_frames" in state.extra:
            self._collected_frames = int(state.extra["collected_frames"])
