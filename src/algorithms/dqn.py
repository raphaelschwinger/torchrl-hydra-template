"""Deep Q-Network (DQN) algorithm.

Compatible environments:
  - Discrete-action Gym environments (CartPole-v1, MLP network)
  - Atari pixel environments (ALE/Breakout-v5, CNN network)

Architecture:
  - Q-network: MLP or AtariCNN → action values
  - Policy: QValueActor with epsilon-greedy exploration (EGreedyModule)
  - Loss: DQNLoss with double-DQN target network
  - Target update: HardUpdate every N frames
  - Buffer: ReplayBuffer with LazyMemmapStorage
  - Trainer: StepTrainer (SyncDataCollector, step-based)
"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.envs import EnvBase

from src.algorithms.base import BaseAlgorithm, CollectorConfig, TrainingState

from torchrl.data import Composite, LazyTensorStorage, ReplayBuffer, TensorDictReplayBuffer
from torchrl.modules import EGreedyModule, MLP, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate, SoftUpdate


class DQNAlgorithm(BaseAlgorithm):
    """Deep Q-learning with Experience Replay by Mnih et al. (2015).
    https://www.nature.com/articles/nature14236
    
    Defaults are tuned for CartPole-v1 (MLP network).
   """

    def __init__(
        self,
        device: torch.device | None = None,
        *,
        replay_buffer: Callable[[], ReplayBuffer] = lambda: TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size=10_000,device='cpu'),
            batch_size=128,
        ),
        network: Callable[[tuple[int, ...], int], nn.Module] = lambda input_shape, num_outputs: MLP(
            in_features=input_shape,
            activation_class=torch.nn.ReLU,
            out_features=num_outputs,
            num_cells=[120, 84],
     
        ),
        annealing_frames: int = 50_000,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        gamma: float = 0.99,
        hard_update_freq: int = 1000,
        lr: float = 1e-4,
        frames_per_batch: int = 200,
        init_random_frames: int = 5_000,
        max_frames_per_traj: int = -1,
        num_updates: int = 1,
        batch_size: int = 128,

    ) -> None:
        super().__init__(device)
        self.replay_buffer = replay_buffer
        self.network = network
        self.annealing_frames = annealing_frames
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.gamma = gamma
        self.hard_update_freq = hard_update_freq
        self.lr = lr
        self.frames_per_batch = frames_per_batch
        self.init_random_frames = init_random_frames
        self.max_frames_per_traj = max_frames_per_traj
        self.num_updates = num_updates
        self.batch_size = batch_size


    def setup(self, make_env: Callable[[], EnvBase]) -> None:
        # Initialize replay buffer
        self.replay_buffer = self.replay_buffer()

        # Initialite action-value function Q with random weights

        proof_environment = make_env()
        # Define input shape
        input_shape = proof_environment.observation_spec["observation"].shape[-1]
        env_specs = proof_environment.specs
        num_outputs = env_specs["input_spec", "full_action_spec", "action"].space.n
        action_spec = env_specs["input_spec", "full_action_spec", "action"]

        network = network(in_features=int(math.prod(input_shape)), out_features=num_outputs, device=self.device)

        self.q_actor = QValueActor(
            module=network,
            spec=Composite(action=action_spec).to(self.device),
            in_keys=["observation"],
            out_keys=["action_value"],
        )

        greedy_module = EGreedyModule(
            annealing_num_steps=self.annealing_frames,
            eps_init=self.eps_start,
            eps_end=self.eps_end,
            spec=self.q_actor.spec,
            device=self.device,
            )

        self._explore_policy = TensorDictSequential(
            self.q_actor,
            greedy_module,
        )

        # create the loss module
        loss_module = DQNLoss(
            value_network=self.q_actor,
            loss_function="l2",
            delay_value=True,
        )
        loss_module.make_value_estimator(gamma=self.gamma, device=self.device)
        loss_module = loss_module.to(self.device)
        self.target_net_updater = HardUpdate(
            loss_module, value_network_update_interval=self.hard_update_freq
        )

        # Create the optimizer
        self.optimizer = torch.optim.Adam(loss_module.parameters(), lr=self.lr)

    def get_collector_config(self) -> CollectorConfig:
        return CollectorConfig(
            frames_per_batch=self.frames_per_batch,
            init_random_frames=self.init_random_frames,
            max_frames_per_traj=self.max_frames_per_traj,
            batch_size=self.batch_size,
        )

        

    def step(self, batch: TensorDict) -> dict[str, float]:
        # anneal epsilon
        self.greedy_policy.step(batch.numel())

        # Store transition in the replay buffer
        self.replay_buffer.extend(batch)

        # Optimize the Q-network
        
        for j in range(self.num_updates):
            sampled_tensordict = self.replay_buffer.sample(self.batch_size)
            sampled_tensordict = sampled_tensordict.to(self.device)
            loss_td = self.loss_module(sampled_tensordict)
            q_loss = loss_td["loss"]
            q_loss.backward()
            self.optimizer.step()
            self.target_net_updater.step()


        return {
            "loss/td": q_loss.item(),
            "q/mean": sampled_tensordict["action_value"].mean().item(),
       
        }
  
    # ------------------------------------------------------------------
    # Policy access
    # ------------------------------------------------------------------

    def get_policy(self) -> TensorDictModule:
        return self.q_actor

    def get_explore_policy(self) -> TensorDictModule:
        return self._explore_policy


    # ------------------------------------------------------------------
    # Training hooks
    # ------------------------------------------------------------------



    def should_skip_update(self, frames_collected: int) -> bool:
        return frames_collected < self.init_random_frames



    def on_step_complete(self, frames_collected: int) -> None:
        """Decay epsilon."""
        if frames_collected >= self.init_random_frames:
            self.eps_module.step(self.frames_per_batch)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _get_training_state(self) -> TrainingState:
        return TrainingState(
            step=0,
            policy_state_dict=self.q_actor.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            extra={"storage_path": str(getattr(self.replay_buffer._storage, "scratch_dir", "in-memory"))},
        )

    def _load_training_state(self, state: TrainingState) -> None:
        self.q_actor.load_state_dict(state.policy_state_dict)
        self.optimizer.load_state_dict(state.optimizer_state_dict)
