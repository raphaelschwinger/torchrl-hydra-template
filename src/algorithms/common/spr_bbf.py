"""Shared SPR/BBF agent core in PyTorch."""

from __future__ import annotations

import dataclasses
import math
import copy
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.algorithms.der.der import DERAgent
from src.algorithms.der.der import DERConfig
from src.algorithms.common.pixel_networks import RainbowDQNNetwork
from src.algorithms.common.rl import categorical_target
from src.algorithms.common.rl import linearly_decaying_epsilon
from src.algorithms.common.rl import select_actions


@dataclasses.dataclass
class SPRBBFConfig(DERConfig):
  gamma: float = 0.997
  update_horizon: int = 3
  max_update_horizon: int = 10
  min_gamma: float = 0.97
  cycle_steps: int = 10_000
  learning_rate: float = 1e-4
  weight_decay: float = 0.1
  noisy: bool = False
  dueling: bool = True
  double_dqn: bool = True
  replay_ratio: int = 64
  target_update_period: int = 1
  target_update_tau: float = 0.005
  max_target_update_tau: float | None = None
  epsilon_train: float = 0.0
  epsilon_eval: float = 0.001
  epsilon_decay_period: int = 2001
  min_replay_history: int = 2000
  encoder_type: str = "impala"
  hidden_dim: int = 2048
  width_scale: int = 4
  renormalize_output: bool = True
  data_augmentation: bool = True
  batches_to_group: int = 2
  spr_weight: float = 5.0
  jumps: int = 5
  reset_every: int = 20_000
  reset_offset: int = 1
  no_resets_after: int = 100_000
  reset_priorities: bool = False
  reset_interval_scaling: float | str | None = None
  offline_update_frac: float = 0.0
  shrink_perturb_keys: tuple[str, ...] = ("encoder", "transition_model")
  shrink_factor: float = 0.5
  perturb_factor: float = 0.5
  reset_projection: bool = False
  reset_encoder: bool = False
  reset_noise: bool = False
  reset_head: bool = True
  reset_target: bool = True
  target_action_selection: bool = True
  match_online_target_rngs: bool = True


class SPRBBFAgent(DERAgent):
  """Shared learning core extending DER with SPR and optional periodic resets."""

  config: SPRBBFConfig

  def __init__(self, config: SPRBBFConfig, seed: int = 0) -> None:
    super().__init__(config, seed=seed)
    self.cumulative_resets = 0
    self.next_reset = config.reset_every + config.reset_offset
    self.cycle_gradient_steps = 0
    self.reset_priorities_requested = False

  def current_update_horizon(self) -> int:
    if self.config.max_update_horizon <= self.config.update_horizon:
      return int(self.config.update_horizon)
    ratio = self._exponential_schedule(
        self.config.cycle_steps,
        initial_value=1.0,
        final_value=self.config.update_horizon / self.config.max_update_horizon,
        step=self.cycle_gradient_steps,
    )
    return max(1, int(round(ratio * self.config.max_update_horizon)))

  def current_gamma(self) -> float:
    if self.config.cycle_steps <= 1:
      return float(self.config.gamma)
    return self._exponential_schedule(
        self.config.cycle_steps,
        initial_value=self.config.min_gamma,
        final_value=self.config.gamma,
        step=self.cycle_gradient_steps,
        reverse=True,
    )

  def _current_target_update_tau(self) -> float:
    if self.config.max_target_update_tau is None:
      return float(self.config.target_update_tau)
    return self._exponential_schedule(
        self.config.cycle_steps,
        initial_value=float(self.config.max_target_update_tau),
        final_value=float(self.config.target_update_tau),
        step=self.cycle_gradient_steps,
    )

  def replay_update_horizon(self) -> int:
    return int(self.config.max_update_horizon)

  def train_step(self, batch: dict[str, Any]) -> dict[str, float | np.ndarray]:
    if self._should_reset():
      self.reset_weights()
    mini_batches = self._split_grouped_batch(batch)
    metrics = [self._train_one_minibatch(mini_batch) for mini_batch in mini_batches]
    return self._merge_group_metrics(metrics)

  def _train_one_minibatch(self, batch: dict[str, Any]) -> dict[str, float | np.ndarray]:
    self.online_network.train()
    states = self._batch_tensor(batch["state"])
    next_states = self._batch_tensor(batch["next_state"])
    actions = self._batch_tensor(batch["action"]).long()
    rewards = self._batch_tensor(batch["return"]).float()[:, 0]
    terminals = self._batch_tensor(batch["terminal"]).float()[:, 0]
    discounts = self._batch_tensor(batch["discount"]).float()[:, 0]
    same_trajectory = self._same_trajectory(batch, states)
    loss_weights = self._loss_weights(batch, states.shape[0])

    current_states = states[:, 0]
    backup_next_states = next_states[:, 0]
    current_actions = actions[:, 0]
    rollout_actions = actions[:, :-1]
    current_latent = None
    backup_next_processed = None
    future_processed = None
    if self.config.data_augmentation:
      states_processed = self._preprocess_state_sequence(states)
      current_processed = states_processed[:, 0]
      current_latent = self.online_network.encode_processed(current_processed)
      backup_next_processed = self._preprocess_state_batch(backup_next_states)
      if states.shape[1] > 1:
        future_processed = states_processed[:, 1:]

    with torch.no_grad():
      if backup_next_processed is None:
        next_online = self.online_network(
            backup_next_states,
            self.support,
            eval_mode=False,
            data_augmentation=self.config.data_augmentation,
        )
        next_target = self.target_network(
            backup_next_states,
            self.support,
            eval_mode=self.config.target_eval_mode,
            data_augmentation=self.config.data_augmentation,
        )
      else:
        backup_next_online_latent = self.online_network.encode_processed(backup_next_processed)
        backup_next_target_processed = (
            backup_next_processed
            if self.config.match_online_target_rngs
            else self._preprocess_state_batch(backup_next_states)
        )
        backup_next_target_latent = self.target_network.encode_processed(backup_next_target_processed)
        next_online = self.online_network.forward_from_latent(
            backup_next_online_latent,
            self.support,
            eval_mode=False,
        )
        next_target = self.target_network.forward_from_latent(
            backup_next_target_latent,
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
      spr_targets = self._spr_targets(states[:, 1:], future_processed=future_processed)

    if current_latent is None:
      output = self.online_network(
          current_states,
          self.support,
          actions=rollout_actions,
          do_rollout=self.config.spr_weight > 0 and rollout_actions.shape[1] > 0,
          eval_mode=False,
          data_augmentation=self.config.data_augmentation,
      )
    else:
      output = self.online_network.forward_from_latent(
          current_latent,
          self.support,
          actions=rollout_actions,
          do_rollout=self.config.spr_weight > 0 and rollout_actions.shape[1] > 0,
          eval_mode=False,
      )
    assert output.logits is not None
    chosen_logits = output.logits[torch.arange(current_actions.shape[0], device=self.device), current_actions]
    dqn_loss = -(target * F.log_softmax(chosen_logits, dim=-1)).sum(dim=-1)
    spr_loss = self._spr_loss(output.latent, spr_targets, same_trajectory)
    per_sample_loss = dqn_loss + self.config.spr_weight * spr_loss
    loss = (loss_weights * per_sample_loss).mean()

    self.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    grad_norm = nn.utils.clip_grad_norm_(self.online_network.parameters(), max_norm=10.0)
    self.optimizer.step()
    self._maybe_update_target()
    self.gradient_steps += 1
    self.cycle_gradient_steps += 1

    priorities = torch.sqrt(dqn_loss.detach() + 1e-10).cpu().numpy()
    return {
        "TotalLoss": float(loss.detach().cpu()),
        "DQNLoss": float(dqn_loss.mean().detach().cpu()),
        "SPRLoss": float(spr_loss.mean().detach().cpu()),
        "GradNorm": float(torch.as_tensor(grad_norm).detach().cpu()),
        "priorities": priorities,
    }

  def _split_grouped_batch(self, batch: dict[str, Any]) -> list[dict[str, Any]]:
    first = next(iter(batch.values()))
    batch_size = int(np.asarray(first).shape[0])
    if batch_size <= self.config.batch_size:
      return [batch]
    if batch_size % self.config.batch_size != 0:
      raise ValueError(
          f"Grouped batch size {batch_size} must be divisible by "
          f"mini-batch size {self.config.batch_size}."
      )
    mini_batches = []
    for start in range(0, batch_size, self.config.batch_size):
      end = start + self.config.batch_size
      mini_batches.append({
          key: self._slice_batch_value(value, start, end)
          for key, value in batch.items()
      })
    return mini_batches

  def _slice_batch_value(self, value: Any, start: int, end: int) -> Any:
    if isinstance(value, np.ndarray):
      return value[start:end]
    if isinstance(value, torch.Tensor):
      return value[start:end]
    return value

  def _merge_group_metrics(self, metrics: list[dict[str, float | np.ndarray]]) -> dict[str, float | np.ndarray]:
    if len(metrics) == 1:
      return metrics[0]
    merged: dict[str, float | np.ndarray] = {}
    keys = set().union(*(metric.keys() for metric in metrics))
    for key in keys:
      values = [metric[key] for metric in metrics if key in metric]
      first = values[0]
      if key == "priorities":
        merged[key] = np.concatenate([np.asarray(value) for value in values], axis=0)
      elif isinstance(first, np.ndarray):
        merged[key] = np.mean(np.stack([np.asarray(value) for value in values], axis=0), axis=0)
      else:
        merged[key] = float(np.mean([float(value) for value in values]))
    return merged

  def _should_reset(self) -> bool:
    if self.config.reset_every <= 0:
      return False
    if self.training_steps <= self.next_reset:
      return False
    if self.next_reset > self.config.no_resets_after + self.config.reset_offset:
      return False
    return True

  def reset_weights(self) -> None:
    self.cumulative_resets += 1
    interval = self._next_reset_interval()
    self.next_reset = int(interval) + self.training_steps
    if self.next_reset > self.config.no_resets_after + self.config.reset_offset:
      return

    old_optimizer_state = self._optimizer_state_by_name()
    random_online = self._make_network().to(self.device)
    random_target = self._make_network().to(self.device)
    self._build_specific_network(random_online)
    self._build_specific_network(random_target)
    self._reset_network(self.online_network, random_online)
    if self.config.reset_target:
      self._reset_network(self.target_network, random_target)
    else:
      self.target_network.load_state_dict(self.online_network.state_dict())
    self.optimizer = self._make_optimizer()
    self._restore_optimizer_state_for_kept_keys(old_optimizer_state)
    self.cycle_gradient_steps = 0
    self.reset_priorities_requested = self.config.reset_priorities

  def _next_reset_interval(self) -> int:
    scaling = self.config.reset_interval_scaling
    if scaling is None or scaling is False or scaling == "":
      return int(self.config.reset_every)
    if str(scaling).lower() == "linear":
      return int(self.config.reset_every * (1 + self.cumulative_resets))
    if "epoch" in str(scaling).lower():
      raise NotImplementedError(
          "reset_interval_scaling='epochs:<n>' needs replay-size access and "
          "is intentionally not approximated in the standalone agent."
      )
    if isinstance(scaling, (float, int)) or "." in str(scaling):
      return int(self.config.reset_every * float(scaling) ** self.cumulative_resets)
    raise NotImplementedError(f"Unsupported reset_interval_scaling={scaling!r}")

  def _optimizer_state_by_name(self) -> dict[str, dict[str, Any]]:
    state_by_name = {}
    for name, parameter in self.online_network.named_parameters():
      state = self.optimizer.state.get(parameter)
      if state:
        state_by_name[name] = copy.deepcopy(state)
    return state_by_name

  def _restore_optimizer_state_for_kept_keys(self, old_state: dict[str, dict[str, Any]]) -> None:
    keep_prefixes = self._keys_to_copy()
    for name, parameter in self.online_network.named_parameters():
      if name not in old_state or not self._matches_prefix(name, keep_prefixes):
        continue
      self.optimizer.state[parameter] = copy.deepcopy(old_state[name])

  def _exponential_schedule(
      self,
      decay_period: int,
      *,
      initial_value: float,
      final_value: float,
      step: int,
      reverse: bool = False,
  ) -> float:
    if decay_period == 0:
      return initial_value if step < 0 else final_value
    if reverse:
      initial_value = 1.0 - initial_value
      final_value = 1.0 - final_value
    start = math.log(initial_value)
    end = math.log(final_value)
    bonus = np.clip((decay_period - step) / decay_period, 0.0, 1.0)
    value = math.exp(float(bonus) * (start - end) + end)
    if reverse:
      value = 1.0 - value
    return value

  def _reset_network(self, network: RainbowDQNNetwork, random_network: RainbowDQNNetwork) -> None:
    keep_prefixes = self._keys_to_copy()
    random_state = random_network.state_dict()
    current_state = network.state_dict()
    new_state = {}
    for name, value in current_state.items():
      if not value.dtype.is_floating_point:
        new_state[name] = value
      elif self._matches_prefix(name, self.config.shrink_perturb_keys):
        interpolated = value * self.config.shrink_factor + random_state[name] * self.config.perturb_factor
        new_state[name] = interpolated if self._matches_prefix(name, keep_prefixes) else random_state[name]
      elif self._matches_prefix(name, keep_prefixes):
        new_state[name] = value
      else:
        new_state[name] = random_state[name]
    network.load_state_dict(new_state)

  def _keys_to_copy(self) -> tuple[str, ...]:
    keys = []
    if not self.config.reset_projection:
      keys.extend(["projection", "predictor"])
    if not self.config.reset_encoder:
      keys.extend(["encoder", "transition_model"])
    if not self.config.reset_noise:
      keys.extend(["kernell", "biass", "weight_sigma", "bias_sigma"])
    if not self.config.reset_head:
      keys.append("head")
    return tuple(keys)

  def _matches_prefix(self, state_name: str, prefixes: tuple[str, ...]) -> bool:
    for prefix in prefixes:
      if state_name.startswith(prefix) or f".{prefix}" in state_name:
        return True
    return False

  def _build_specific_network(self, network: RainbowDQNNetwork) -> None:
    height, width = self.config.observation_shape
    dummy = torch.zeros(
        (1, self.config.stack_size, height, width),
        dtype=torch.uint8,
        device=self.device,
    )
    with torch.no_grad():
      network(dummy, self.support, eval_mode=True)

  def select_action(self, state: np.ndarray | torch.Tensor, *, eval_mode: bool = False) -> torch.Tensor:
    network = self.target_network if self.config.target_action_selection else self.online_network
    noisy_eval_mode = eval_mode and not self.config.eval_noise
    network.eval() if eval_mode else network.train()
    state_tensor = self._state_to_tensor(state)
    with torch.inference_mode():
      q_values = network(
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

  def _preprocess_state_batch(self, states: torch.Tensor) -> torch.Tensor:
    return self.online_network.preprocess(
        states,
        data_augmentation=self.config.data_augmentation,
    )

  def _preprocess_state_sequence(self, states: torch.Tensor) -> torch.Tensor:
    batch_size, seq_len = states.shape[:2]
    flat_states = states.reshape(batch_size * seq_len, *states.shape[2:])
    flat_processed = self._preprocess_state_batch(flat_states)
    return flat_processed.reshape(batch_size, seq_len, *flat_processed.shape[1:])

  def _spr_targets(
      self,
      future_states: torch.Tensor,
      *,
      future_processed: torch.Tensor | None = None,
  ) -> torch.Tensor:
    batch_size, jumps = future_states.shape[:2]
    if future_processed is None:
      flat_states = future_states.reshape(batch_size * jumps, *future_states.shape[2:])
      projected = self.target_network.encode_project(
          flat_states,
          eval_mode=True,
          data_augmentation=self.config.data_augmentation,
      )
    else:
      flat_processed = future_processed.reshape(batch_size * jumps, *future_processed.shape[2:])
      flat_latents = self.target_network.encode_processed(flat_processed)
      projected = self.target_network.encode_project_from_latent(
          flat_latents,
          eval_mode=True,
      )
    return projected.reshape(batch_size, jumps, -1)

  def _spr_loss(
      self,
      predictions: torch.Tensor,
      targets: torch.Tensor,
      same_trajectory: torch.Tensor,
  ) -> torch.Tensor:
    if self.config.spr_weight <= 0 or predictions.ndim != 3:
      return torch.zeros(targets.shape[0], device=self.device)
    horizon = min(predictions.shape[1], targets.shape[1], same_trajectory.shape[1])
    predictions = predictions[:, :horizon]
    targets = targets[:, :horizon]
    mask = same_trajectory[:, :horizon].float()
    predictions = F.normalize(predictions, p=2, dim=-1)
    targets = F.normalize(targets, p=2, dim=-1)
    loss = (predictions - targets).pow(2).sum(dim=-1)
    return (loss * mask).mean(dim=-1)

  def _same_trajectory(self, batch: dict[str, Any], states: torch.Tensor) -> torch.Tensor:
    if "same_trajectory" in batch:
      return self._batch_tensor(batch["same_trajectory"]).float()[:, 1:]
    return torch.ones((states.shape[0], max(states.shape[1] - 1, 1)), device=self.device)

  def state_dict(self) -> dict[str, Any]:
    state = super().state_dict()
    state["cycle_gradient_steps"] = self.cycle_gradient_steps
    state["next_reset"] = self.next_reset
    state["cumulative_resets"] = self.cumulative_resets
    return state

  def load_state_dict(self, state: dict[str, Any]) -> None:
    super().load_state_dict(state)
    self.cycle_gradient_steps = int(state.get("cycle_gradient_steps", 0))
    self.next_reset = int(
        state.get("next_reset", self.config.reset_every + self.config.reset_offset)
    )
    self.cumulative_resets = int(state.get("cumulative_resets", 0))
