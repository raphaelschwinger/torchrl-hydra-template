"""Shared RL utilities for DER and BBF."""

from __future__ import annotations

import torch


def identity_epsilon(
    decay_period: int,
    step: int,
    warmup_steps: int,
    epsilon: float,
) -> float:
  del decay_period, step, warmup_steps
  return epsilon


def linearly_decaying_epsilon(
    decay_period: int,
    step: int,
    warmup_steps: int,
    epsilon: float,
) -> float:
  steps_left = decay_period + warmup_steps - step
  bonus = (1.0 - epsilon) * steps_left / decay_period
  bonus = min(max(bonus, 0.0), 1.0 - epsilon)
  return epsilon + bonus


def project_distribution(
    target_support: torch.Tensor,
    next_probabilities: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
  """Project a batch of categorical targets onto a fixed support."""
  v_min, v_max = support[0], support[-1]
  delta_z = support[1] - support[0]
  clipped = target_support.clamp(v_min, v_max)
  b = (clipped - v_min) / delta_z
  lower = b.floor().long().clamp(0, support.numel() - 1)
  upper = b.ceil().long().clamp(0, support.numel() - 1)
  projected = torch.zeros_like(next_probabilities)
  batch_size, num_atoms = next_probabilities.shape
  offset = torch.arange(batch_size, device=support.device, dtype=torch.long)[:, None] * num_atoms
  lower_index = (lower + offset).reshape(-1)
  upper_index = (upper + offset).reshape(-1)
  probabilities = next_probabilities.reshape(-1)
  b_flat = b.reshape(-1)
  lower_flat = lower.reshape(-1)
  upper_flat = upper.reshape(-1)
  lower_mass = probabilities * (upper_flat.to(b.dtype) - b_flat)
  upper_mass = probabilities * (b_flat - lower_flat.to(b.dtype))
  equal_atoms = lower_index == upper_index
  lower_mass = torch.where(equal_atoms, probabilities, lower_mass)
  upper_mass = torch.where(equal_atoms, torch.zeros_like(upper_mass), upper_mass)
  projected.view(-1).index_add_(0, lower_index, lower_mass)
  projected.view(-1).index_add_(0, upper_index, upper_mass)
  return projected


def categorical_target(
    rewards: torch.Tensor,
    terminals: torch.Tensor,
    support: torch.Tensor,
    cumulative_gamma: torch.Tensor,
    next_online_q: torch.Tensor,
    next_target_q: torch.Tensor,
    next_target_probabilities: torch.Tensor,
    *,
    double_dqn: bool,
) -> torch.Tensor:
  """Build the projected categorical target for distributional DER/BBF."""
  gamma_with_terminal = cumulative_gamma * (1.0 - terminals.float())
  greedy_actions = next_online_q.argmax(dim=-1) if double_dqn else next_target_q.argmax(dim=-1)
  next_probabilities = next_target_probabilities[torch.arange(next_target_probabilities.shape[0]), greedy_actions]
  target_support = rewards[:, None] + gamma_with_terminal[:, None] * support[None, :]
  return project_distribution(target_support, next_probabilities, support)


def select_actions(
    q_values: torch.Tensor,
    *,
    eval_mode: bool,
    epsilon_eval: float,
    epsilon_train: float,
    epsilon_decay_period: int,
    training_steps: int,
    min_replay_history: int,
    epsilon_fn=linearly_decaying_epsilon,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
  if eval_mode:
    epsilon = epsilon_eval
  else:
    epsilon = epsilon_fn(
        epsilon_decay_period,
        training_steps,
        min_replay_history,
        epsilon_train,
    )
  batch_size, num_actions = q_values.shape
  generator = generator or torch.Generator(device=q_values.device)
  random_actions = torch.randint(
      0,
      num_actions,
      (batch_size,),
      device=q_values.device,
      generator=generator,
  )
  greedy_actions = q_values.argmax(dim=-1)
  probs = torch.rand((batch_size,), device=q_values.device, generator=generator)
  return torch.where(probs <= epsilon, random_actions, greedy_actions)
