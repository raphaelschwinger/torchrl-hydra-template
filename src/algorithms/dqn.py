"""Deep Q-Network (DQN).

Mnih et al. (2015), "Human-level control through deep reinforcement learning."
https://www.nature.com/articles/nature14236

Pseudocode:
    Initialise replay buffer D
    Initialise Q-network with random weights theta
    Initialise target Q-network with weights theta_target = theta
    For each step:
        With probability epsilon select random action, else a = argmax_a Q(s, a; theta)
        Execute a, observe r and s'
        Store (s, a, r, s', done) in D
        Sample minibatch from D
        y = r + gamma * (1 - done) * max_a' Q(s', a'; theta_target)
        Update theta by minimising (y - Q(s, a; theta))^2
        Every C gradient steps: theta_target <- theta

Defaults match the torchrl SOTA reference for CartPole-v1
(https://github.com/pytorch/rl/blob/main/sota-implementations/dqn/dqn_cartpole.py).
"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.data import LazyTensorStorage, ReplayBuffer, TensorDictReplayBuffer
from torchrl.envs import EnvBase
from torchrl.modules import EGreedyModule, MLP, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate

from src.algorithms.base import BaseAlgorithm, CollectorConfig, TrainingState


def default_replay_buffer() -> ReplayBuffer:
    """Default replay buffer: in-memory, 10k transitions (matches SOTA reference)."""
    return TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=10_000, device="cpu"),
    )


def default_network(input_shape: tuple[int, ...], num_outputs: int) -> nn.Module:
    """Default Q-network: MLP [120, 84] with ReLU (matches SOTA reference)."""
    return MLP(
        in_features=int(math.prod(input_shape)),
        activation_class=nn.ReLU,
        out_features=num_outputs,
        num_cells=[120, 84],
    )


class DQNAlgorithm(BaseAlgorithm):
    """DQN with experience replay, target network and epsilon-greedy exploration.

    Defaults are tuned for CartPole-v1 and mirror the torchrl SOTA reference
    (sota-implementations/dqn/config_cartpole.yaml).
    """

    def __init__(
        self,
        device: torch.device | None = None,
        *,
        # --- Design choices: factories injected as Callables ---------------
        replay_buffer: Callable[[], ReplayBuffer] = default_replay_buffer,
        network: Callable[[tuple[int, ...], int], nn.Module] = default_network,
        # --- Optimisation --------------------------------------------------
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        batch_size: int = 128,
        max_grad_norm: float = 10.0,
        # --- Exploration (epsilon-greedy, linear anneal in frames) ---------
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        annealing_frames: int = 250_000,
        # --- Data collection ----------------------------------------------
        frames_per_batch: int = 1_000,
        init_random_frames: int = 10_000,
        max_frames_per_traj: int = -1,
        # --- Learning schedule --------------------------------------------
        num_updates: int = 100,           # gradient updates per collector batch
        hard_update_freq: int = 50,       # target net <- online every N grad steps
    ) -> None:
        super().__init__(device)
        self._make_replay_buffer = replay_buffer
        self._make_network = network
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.annealing_frames = annealing_frames
        self.frames_per_batch = frames_per_batch
        self.init_random_frames = init_random_frames
        self.max_frames_per_traj = max_frames_per_traj
        self.num_updates = num_updates
        self.hard_update_freq = hard_update_freq
        self._collected_frames = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, make_env: Callable[[], EnvBase]) -> None:
        # Read env specs from a short-lived proof environment.
        proof_env = make_env()
        obs_shape = tuple(proof_env.observation_spec["observation"].shape)
        action_spec = proof_env.action_spec
        num_actions = int(action_spec.space.n)

        # 1. Q-network -> QValueActor (action_value head + greedy argmax).
        q_net = self._make_network(obs_shape, num_actions).to(self.device)
        self.q_actor = QValueActor(
            module=q_net,
            spec=action_spec,
            in_keys=["observation"],
        ).to(self.device)

        # 2. Epsilon-greedy module wrapping the actor for exploration.
        self.greedy_module = EGreedyModule(
            spec=action_spec,
            eps_init=self.eps_start,
            eps_end=self.eps_end,
            annealing_num_steps=self.annealing_frames,
            device=self.device,
        )
        self._explore_policy = TensorDictSequential(self.q_actor, self.greedy_module)

        # 3. Replay buffer.
        self.replay_buffer = self._make_replay_buffer()

        # 4. DQN loss with delayed target network + HardUpdate scheduler.
        self.loss_module = DQNLoss(
            value_network=self.q_actor,
            loss_function="l2",
            delay_value=True,
        )
        self.loss_module.make_value_estimator(gamma=self.gamma)
        self.loss_module = self.loss_module.to(self.device)
        self.target_updater = HardUpdate(
            self.loss_module, value_network_update_interval=self.hard_update_freq
        )

        # 5. Optimiser on the online Q-network parameters.
        self.optimizer = torch.optim.Adam(self.q_actor.parameters(), lr=self.lr)

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
        """One collector iteration: anneal epsilon, store, optimise."""
        # Always: anneal epsilon and store transitions, even during warm-up.
        batch = batch.reshape(-1)
        self.greedy_module.step(batch.numel())
        self.replay_buffer.extend(batch)
        self._collected_frames += batch.numel()

        # Warm-up: collect random transitions before any gradient update.
        if self._collected_frames < self.init_random_frames:
            return {"epsilon": float(self.greedy_module.eps)}

        losses = torch.zeros(self.num_updates, device=self.device)
        for j in range(self.num_updates):
            sample = self.replay_buffer.sample(self.batch_size).to(self.device)
            loss = self.loss_module(sample)["loss"]

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_actor.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.target_updater.step()
            losses[j] = loss.detach()

        return {
            "loss/td": losses.mean().item(),
            "epsilon": float(self.greedy_module.eps),
        }

    # ------------------------------------------------------------------
    # Policy access
    # ------------------------------------------------------------------

    def get_policy(self) -> TensorDictModule:
        return self.q_actor

    def get_explore_policy(self) -> TensorDictModule:
        return self._explore_policy

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _get_training_state(self) -> TrainingState:
        return TrainingState(
            step=0,
            policy_state_dict=self.q_actor.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            extra={"collected_frames": self._collected_frames},
        )

    def _load_training_state(self, state: TrainingState) -> None:
        self.q_actor.load_state_dict(state.policy_state_dict)
        self.optimizer.load_state_dict(state.optimizer_state_dict)
        if state.extra and "collected_frames" in state.extra:
            self._collected_frames = int(state.extra["collected_frames"])
