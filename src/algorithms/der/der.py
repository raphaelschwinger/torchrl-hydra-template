"""Data Efficient Rainbow (DER) agent core in PyTorch."""

from __future__ import annotations

import copy
import dataclasses
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.algorithms.common.pixel_networks import RainbowDQNNetwork
from src.algorithms.common.rl import categorical_target
from src.algorithms.common.rl import linearly_decaying_epsilon
from src.algorithms.common.rl import select_actions


@dataclasses.dataclass
class DERConfig:
  num_actions: int
  observation_shape: tuple[int, int] = (84, 84)
  stack_size: int = 4
  num_atoms: int = 51
  v_min: float = -10.0
  v_max: float = 10.0
  gamma: float = 0.99
  update_horizon: int = 10
  learning_rate: float = 1e-4
  adam_eps: float = 1.5e-4
  weight_decay: float = 0.0
  noisy: bool = True
  dueling: bool = True
  double_dqn: bool = True
  distributional: bool = True
  replay_ratio: int = 32
  batch_size: int = 32
  target_update_period: int = 8000
  target_update_tau: float = 1.0
  epsilon_train: float = 0.01
  epsilon_eval: float = 0.001
  epsilon_decay_period: int = 2000
  min_replay_history: int = 1600
  encoder_type: str = "dqn"
  hidden_dim: int = 512
  width_scale: int = 1
  renormalize_output: bool = False
  data_augmentation: bool = False
  batches_to_group: int = 1
  eval_noise: bool = True
  target_eval_mode: bool = False
  device: str = "cpu"


class DERAgent:
  """Minimal DER learning core.

  This class intentionally stops at the algorithm core: network construction,
  action selection, C51 targets, optimizer updates, target synchronization, and
  priority values. Environment runners live one layer above this.
  """

  def __init__(self, config: DERConfig, seed: int = 0) -> None:
    self.config = config
    self.device = torch.device(config.device)
    self.training_steps = 0
    self.gradient_steps = 0
    self.generator = torch.Generator(device=self.device).manual_seed(seed)
    self.support = torch.linspace(
        config.v_min,
        config.v_max,
        config.num_atoms,
        device=self.device,
    )
    self.online_network = self._make_network().to(self.device)
    self.target_network = self._make_network().to(self.device)
    self._build_lazy_modules()
    self.target_network.load_state_dict(self.online_network.state_dict())
    self.target_network.eval()
    self.optimizer = self._make_optimizer()

  def _make_network(self) -> RainbowDQNNetwork:
    return RainbowDQNNetwork(
        num_actions=self.config.num_actions,
        num_atoms=self.config.num_atoms,
        noisy=self.config.noisy,
        dueling=self.config.dueling,
        distributional=self.config.distributional,
        encoder_type=self.config.encoder_type,  # type: ignore[arg-type]
        hidden_dim=self.config.hidden_dim,
        width_scale=self.config.width_scale,
        renormalize_output=self.config.renormalize_output,
        input_channels=self.config.stack_size,
    )

  def _make_optimizer(self) -> torch.optim.Optimizer:
    decay_params = []
    no_decay_params = []
    for parameter in self.online_network.parameters():
      if parameter.ndim == 1:
        no_decay_params.append(parameter)
      else:
        decay_params.append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=self.config.learning_rate,
        eps=self.config.adam_eps,
    )

  def _build_lazy_modules(self) -> None:
    height, width = self.config.observation_shape
    dummy = torch.zeros(
        (1, self.config.stack_size, height, width),
        dtype=torch.uint8,
        device=self.device,
    )
    with torch.no_grad():
      self.online_network(dummy, self.support, eval_mode=True)
      self.target_network(dummy, self.support, eval_mode=True)

  def select_action(self, state: np.ndarray | torch.Tensor, *, eval_mode: bool = False) -> torch.Tensor:
    noisy_eval_mode = eval_mode and not self.config.eval_noise
    self.online_network.eval() if eval_mode else self.online_network.train()
    state_tensor = self._state_to_tensor(state)
    with torch.inference_mode():
      q_values = self.online_network(
          state_tensor,
          self.support,
          eval_mode=noisy_eval_mode,
      ).q_values
      return select_actions(
          q_values,
          eval_mode=eval_mode,
          epsilon_eval=self.config.epsilon_eval,
          epsilon_train=self.config.epsilon_train,
          epsilon_decay_period=self.config.epsilon_decay_period,
          training_steps=self.training_steps,
          min_replay_history=self.config.min_replay_history,
          epsilon_fn=linearly_decaying_epsilon,
          generator=self.generator,
      )

  def train_step(self, batch: dict[str, Any]) -> dict[str, float | np.ndarray]:
    self.online_network.train()
    states = self._batch_tensor(batch["state"])[:, 0]
    next_states = self._batch_tensor(batch["next_state"])[:, 0]
    actions = self._batch_tensor(batch["action"]).long()[:, 0]
    rewards = self._batch_tensor(batch["return"]).float()[:, 0]
    terminals = self._batch_tensor(batch["terminal"]).float()[:, 0]
    discounts = self._batch_tensor(batch["discount"]).float()[:, 0]
    loss_weights = self._loss_weights(batch, states.shape[0])

    with torch.no_grad():
      next_online = self.online_network(next_states, self.support, eval_mode=False)
      next_target = self.target_network(
          next_states,
          self.support,
          eval_mode=self.config.target_eval_mode,
      )
      assert next_target.probabilities is not None
      target = categorical_target(
          rewards,
          terminals,
          self.support,
          discounts,
          next_online.q_values,
          next_target.q_values,
          next_target.probabilities,
          double_dqn=self.config.double_dqn,
      )

    output = self.online_network(states, self.support, eval_mode=False)
    assert output.logits is not None
    chosen_logits = output.logits[torch.arange(actions.shape[0], device=self.device), actions]
    per_sample_loss = -(target * F.log_softmax(chosen_logits, dim=-1)).sum(dim=-1)
    loss = (loss_weights * per_sample_loss).mean()

    self.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(self.online_network.parameters(), max_norm=10.0)
    self.optimizer.step()
    self._maybe_update_target()
    self.gradient_steps += 1

    priorities = torch.sqrt(per_sample_loss.detach() + 1e-10).cpu().numpy()
    return {
        "TotalLoss": float(loss.detach().cpu()),
        "DQNLoss": float(per_sample_loss.mean().detach().cpu()),
        "GradNorm": float(torch.as_tensor(grad_norm).detach().cpu()),
        "priorities": priorities,
    }

  def _maybe_update_target(self) -> None:
    if self.gradient_steps % self.config.target_update_period != 0:
      return
    tau = self._current_target_update_tau()
    if tau >= 1.0:
      self.target_network.load_state_dict(self.online_network.state_dict())
      return
    with torch.no_grad():
      for target_param, online_param in zip(
          self.target_network.parameters(),
          self.online_network.parameters(),
      ):
        target_param.mul_(1.0 - tau)
        target_param.add_(online_param, alpha=tau)

  def _current_target_update_tau(self) -> float:
    return float(self.config.target_update_tau)

  def _state_to_tensor(self, state: np.ndarray | torch.Tensor) -> torch.Tensor:
    if not isinstance(state, torch.Tensor):
      state = torch.from_numpy(np.asarray(state))
    if state.ndim == 3:
      state = state.unsqueeze(0)
    return state.to(self.device)

  def _batch_tensor(self, value: np.ndarray | torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor):
      value = torch.from_numpy(np.asarray(value))
    non_blocking = self.device.type == "cuda" and value.device.type == "cpu"
    if non_blocking and not value.is_pinned():
      value = value.pin_memory()
    return value.to(self.device, non_blocking=non_blocking)

  def _loss_weights(self, batch: dict[str, Any], batch_size: int) -> torch.Tensor:
    if "sampling_probabilities" not in batch:
      return torch.ones(batch_size, device=self.device)
    probs = self._batch_tensor(batch["sampling_probabilities"]).float()
    weights = 1.0 / torch.sqrt(probs + 1e-10)
    return weights / weights.max().clamp_min(1e-10)

  def state_dict(self) -> dict[str, Any]:
    return {
        "config": dataclasses.asdict(self.config),
        "online_network": self.online_network.state_dict(),
        "target_network": self.target_network.state_dict(),
        "optimizer": self.optimizer.state_dict(),
        "training_steps": self.training_steps,
        "gradient_steps": self.gradient_steps,
    }

  def load_state_dict(self, state: dict[str, Any]) -> None:
    self.online_network.load_state_dict(state["online_network"])
    self.target_network.load_state_dict(state["target_network"])
    self.optimizer.load_state_dict(state["optimizer"])
    self.training_steps = int(state.get("training_steps", 0))
    self.gradient_steps = int(state.get("gradient_steps", 0))

  def clone_for_eval(self) -> "DERAgent":
    clone = DERAgent(copy.deepcopy(self.config))
    clone.load_state_dict(self.state_dict())
    clone.online_network.eval()
    clone.target_network.eval()
    return clone
