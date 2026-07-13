"""Subsequence replay buffers for Atari 100K style agents."""

from __future__ import annotations

import collections
import math
from collections.abc import Iterable
from typing import Any

import numpy as np
import torch

from src.replay.sum_tree import DeterministicSumTree

ReplayElement = collections.namedtuple("ReplayElement", ["name", "shape", "type"])


def modulo_range(start: int, length: int, modulo: int) -> Iterable[int]:
  for i in range(length):
    yield (start + i) % modulo


def invalid_range(
    cursor: int,
    replay_capacity: int,
    stack_size: int,
    update_horizon: int,
) -> np.ndarray:
  if cursor >= replay_capacity:
    raise ValueError(f"cursor {cursor} must be smaller than replay_capacity {replay_capacity}.")
  return np.array(
      [(cursor - update_horizon + i) % replay_capacity for i in range(stack_size + update_horizon)]
  )


class SubsequenceReplayBuffer:
  """Numpy-backed replay buffer with subsequence sampling and optional torch output."""

  def __init__(
      self,
      observation_shape: tuple[int, ...],
      stack_size: int,
      replay_capacity: int,
      batch_size: int,
      subseq_len: int,
      n_envs: int = 1,
      update_horizon: int = 1,
      gamma: float = 0.99,
      max_sample_attempts: int = 1000,
      use_next_state: bool = True,
      extra_storage_types: list[ReplayElement] | None = None,
      observation_dtype: np.dtype = np.uint8,
      terminal_dtype: np.dtype = np.uint8,
      action_shape: tuple[int, ...] = (),
      action_dtype: np.dtype = np.int32,
      reward_shape: tuple[int, ...] = (),
      reward_dtype: np.dtype = np.float32,
      seed: int | None = None,
  ) -> None:
    if replay_capacity < update_horizon + stack_size:
      raise ValueError("There is not enough capacity to cover update_horizon and stack_size.")
    self._action_shape = action_shape
    self._action_dtype = action_dtype
    self._reward_shape = reward_shape
    self._reward_dtype = reward_dtype
    self._observation_shape = observation_shape
    self._stack_size = stack_size
    self._state_shape = observation_shape + (stack_size,)
    self._batch_size = batch_size
    self._update_horizon = update_horizon
    self._gamma = gamma
    self._observation_dtype = observation_dtype
    self._terminal_dtype = terminal_dtype
    self._max_sample_attempts = max_sample_attempts
    self._subseq_len = subseq_len
    self._use_next_state = use_next_state
    self._n_envs = n_envs
    self._replay_length = int(replay_capacity // n_envs)
    self._replay_capacity = self._replay_length * self._n_envs
    self.total_steps = 0
    self._extra_storage_types = extra_storage_types or []
    self._rng = np.random.default_rng(seed)
    self._store: dict[str, np.ndarray] = {}
    self._create_storage()
    self.add_count = np.array(0)
    self.invalid_range = np.zeros((self._stack_size,), dtype=np.int64)
    self._cumulative_discount_vector = np.array(
        [math.pow(self._gamma, n) for n in range(update_horizon + 1)],
        dtype=np.float32,
    )
    self._episode_end_indices: set[tuple[int, int]] = set()

  def _create_storage(self) -> None:
    for storage_element in self.get_storage_signature():
      array_shape = [self._replay_length, self._n_envs] + list(storage_element.shape)
      self._store[storage_element.name] = np.empty(array_shape, dtype=storage_element.type)

  def get_add_args_signature(self) -> list[ReplayElement]:
    return self.get_storage_signature()

  def get_storage_signature(self) -> list[ReplayElement]:
    storage_elements = [
        ReplayElement("observation", self._observation_shape, self._observation_dtype),
        ReplayElement("action", self._action_shape, self._action_dtype),
        ReplayElement("reward", self._reward_shape, self._reward_dtype),
        ReplayElement("terminal", (), self._terminal_dtype),
    ]
    storage_elements.extend(self._extra_storage_types)
    return storage_elements

  def _check_args_length(self, *args: Any) -> None:
    expected = len(self.get_add_args_signature())
    if len(args) != expected:
      raise ValueError(f"Add expects {expected} elements, received {len(args)}")

  def _check_add_types(self, *args: Any) -> None:
    self._check_args_length(*args)
    for i, (arg_element, store_element) in enumerate(zip(args, self.get_add_args_signature())):
      if isinstance(arg_element, np.ndarray):
        arg_shape = arg_element.shape
      elif isinstance(arg_element, (tuple, list)):
        arg_shape = np.asarray(arg_element).shape
      else:
        arg_shape = tuple()
      if arg_shape[0] != self._n_envs:
        raise ValueError(f"arg {i} leading dimension {arg_shape[0]} != n_envs {self._n_envs}")
      if tuple(arg_shape[1:]) != tuple(store_element.shape):
        raise ValueError(f"arg {i} has shape {arg_shape[1:]}, expected {store_element.shape}")

  def add(
      self,
      observation: np.ndarray,
      action: np.ndarray,
      reward: np.ndarray,
      terminal: np.ndarray,
      *args: Any,
      priority: np.ndarray | None = None,
      episode_end: np.ndarray | bool = False,
  ) -> None:
    if priority is not None:
      args = args + (priority,)
    self.total_steps += self._n_envs
    self._check_add_types(observation, action, reward, terminal, *args)
    episode_end = np.asarray(episode_end, dtype=np.uint8)
    if episode_end.shape == ():
      episode_end = np.repeat(episode_end[None], self._n_envs, axis=0)
    resets = episode_end + terminal
    for i in range(resets.shape[0]):
      key = (self.cursor(), i)
      if resets[i]:
        self._episode_end_indices.add(key)
      else:
        self._episode_end_indices.discard(key)
    self._add(observation, action, reward, terminal, *args)

  def _add(self, *args: Any) -> None:
    self._check_args_length(*args)
    transition = {
        e.name: args[idx] for idx, e in enumerate(self.get_add_args_signature())
    }
    self._add_transition(transition)

  def _add_transition(self, transition: dict[str, np.ndarray]) -> None:
    cursor = self.cursor()
    for arg_name, value in transition.items():
      self._store[arg_name][cursor] = value
    self.add_count += 1
    self.invalid_range = invalid_range(
        self.cursor(), self._replay_length, self._stack_size, self._update_horizon
    )

  def is_empty(self) -> bool:
    return bool(self.add_count == 0)

  def is_full(self) -> bool:
    return bool(self.add_count >= self._replay_length)

  def cursor(self) -> int:
    return int(self.add_count % self._replay_length)

  def ravel_indices(self, indices_t: np.ndarray, indices_b: np.ndarray) -> np.ndarray:
    return np.ravel_multi_index(
        (indices_t, indices_b), (self._replay_length, self._n_envs), mode="wrap"
    )

  def unravel_indices(self, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.unravel_index(indices, (self._replay_length, self._n_envs))

  def get_from_store(
      self,
      element_name: str,
      indices_t: np.ndarray,
      indices_b: np.ndarray,
  ) -> np.ndarray:
    return self._store[element_name][indices_t, indices_b]

  def parallel_get_stack(
      self,
      element_name: str,
      indices_t: np.ndarray,
      indices_b: np.ndarray,
      first_valid: np.ndarray | int,
  ) -> np.ndarray:
    indices_t = np.arange(-self._stack_size + 1, 1)[:, None] + indices_t[None, :]
    indices_b = indices_b[None, :].repeat(self._stack_size, axis=0)
    mask = indices_t >= first_valid
    result = self.get_from_store(element_name, indices_t % self._replay_length, indices_b)
    mask = mask.reshape(*mask.shape, *([1] * (len(result.shape) - 2)))
    result = result * mask
    return np.moveaxis(result, 0, -1)

  def get_terminal_stack(self, index_t: np.ndarray, index_b: np.ndarray) -> np.ndarray:
    return self.parallel_get_stack("terminal", index_t, index_b, 0)

  def is_valid_transition(self, index_t: np.ndarray, index_b: np.ndarray) -> tuple[bool, int]:
    index_t_scalar = int(np.asarray(index_t).reshape(-1)[0])
    index_b_scalar = int(np.asarray(index_b).reshape(-1)[0])
    if index_t_scalar < 0 or index_t_scalar >= self._replay_length:
      return False, 0
    if not self.is_full():
      if index_t_scalar >= self.cursor() - self._update_horizon - self._subseq_len:
        return False, 0
      if index_t_scalar < self._stack_size - 1:
        return False, 0
    if index_t_scalar in set(self.invalid_range):
      return False, 0
    terminals = self.get_terminal_stack(index_t, index_b)[0, :-1]
    if terminals.any():
      ep_start = int(index_t_scalar - self._stack_size + terminals.argmax() + 2)
    else:
      ep_start = 0
    for i in modulo_range(index_t_scalar, self._update_horizon, self._replay_length):
      if (i, index_b_scalar) in self._episode_end_indices and not self._store["terminal"][i, index_b]:
        return False, 0
    return True, ep_start

  def sample_index_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if self.is_full():
      min_id = self.cursor() - self._replay_length + self._stack_size - 1
      max_id = self.cursor() - self._update_horizon - self._subseq_len
    else:
      min_id = self._stack_size - 1
      max_id = self.cursor() - self._update_horizon - self._subseq_len
      if max_id <= min_id:
        raise RuntimeError(
            "Cannot sample a batch with fewer than stack_size "
            f"({self._stack_size}) + update_horizon ({self._update_horizon}) transitions."
        )
    t_indices = self._rng.integers(min_id, max_id, size=batch_size) % self._replay_length
    b_indices = self._rng.integers(0, self._n_envs, size=batch_size)
    censor_before = np.zeros_like(t_indices)
    allowed_attempts = self._max_sample_attempts
    for i in range(batch_size):
      is_valid, ep_start = self.is_valid_transition(t_indices[i:i + 1], b_indices[i:i + 1])
      censor_before[i] = ep_start
      if not is_valid:
        if allowed_attempts == 0:
          raise RuntimeError(
              f"Max sample attempts: Tried {self._max_sample_attempts} times but only sampled "
              f"{i} valid indices. Batch size is {batch_size}"
          )
        while not is_valid and allowed_attempts > 0:
          t_indices[i] = self._rng.integers(min_id, max_id) % self._replay_length
          b_indices[i] = self._rng.integers(0, self._n_envs)
          allowed_attempts -= 1
          is_valid, ep_start = self.is_valid_transition(t_indices[i:i + 1], b_indices[i:i + 1])
          censor_before[i] = ep_start
    return t_indices, b_indices, censor_before

  def restore_leading_dims(
      self, batch_size: int, subseq_len: int, tensor: np.ndarray
  ) -> np.ndarray:
    return tensor.reshape(batch_size, subseq_len, *tensor.shape[1:])

  def get_transition_elements(
      self,
      batch_size: int | None = None,
      subseq_len: int | None = None,
  ) -> list[ReplayElement]:
    subseq_len = self._subseq_len if subseq_len is None else subseq_len
    batch_size = self._batch_size if batch_size is None else batch_size
    transition_elements = [
        ReplayElement("state", (batch_size, subseq_len) + self._state_shape, self._observation_dtype),
        ReplayElement("action", (batch_size, subseq_len) + self._action_shape, self._action_dtype),
        ReplayElement("reward", (batch_size, subseq_len) + self._reward_shape, self._reward_dtype),
        ReplayElement("return", (batch_size, subseq_len) + self._reward_shape, self._reward_dtype),
        ReplayElement("discount", (), self._reward_dtype),
    ]
    if self._use_next_state:
      transition_elements.extend([
          ReplayElement("next_state", (batch_size, subseq_len) + self._state_shape, self._observation_dtype),
          ReplayElement("next_action", (batch_size, subseq_len) + self._action_shape, self._action_dtype),
          ReplayElement("next_reward", (batch_size, subseq_len) + self._reward_shape, self._reward_dtype),
      ])
    transition_elements.extend([
        ReplayElement("terminal", (batch_size, subseq_len), self._terminal_dtype),
        ReplayElement("same_trajectory", (batch_size, subseq_len), self._terminal_dtype),
        ReplayElement("indices", (batch_size,), np.int32),
    ])
    for element in self._extra_storage_types:
      transition_elements.append(
          ReplayElement(element.name, (batch_size, subseq_len) + tuple(element.shape), element.type)
      )
    return transition_elements

  def sample_transition_batch(
      self,
      batch_size: int | None = None,
      indices: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
      subseq_len: int | None = None,
      update_horizon: int | None = None,
      gamma: float | None = None,
      *,
      as_torch: bool = False,
      device: torch.device | str | None = None,
  ) -> dict[str, np.ndarray | torch.Tensor]:
    batch_size = self._batch_size if batch_size is None else batch_size
    subseq_len = self._subseq_len if subseq_len is None else subseq_len
    update_horizon = self._update_horizon if update_horizon is None else update_horizon
    if indices is None:
      t_indices, b_indices, censor_before = self.sample_index_batch(batch_size)
    else:
      t_indices, b_indices, censor_before = indices
    cumulative_discount_vector = (
        self._cumulative_discount_vector
        if gamma is None
        else np.array([math.pow(gamma, n) for n in range(update_horizon + 1)], dtype=np.float32)
    )
    state_indices = t_indices[:, None] + np.arange(subseq_len)[None, :]
    state_indices = state_indices.reshape(batch_size * subseq_len) % self._replay_length
    b_indices = b_indices[:, None].repeat(subseq_len, axis=1).reshape(batch_size * subseq_len)
    censor_before = censor_before[:, None].repeat(subseq_len, axis=1).reshape(batch_size * subseq_len)
    trajectory_indices = (np.arange(-1, update_horizon - 1)[:, None] + state_indices[None, :]) % self._replay_length
    trajectory_b_indices = b_indices[None, :].repeat(update_horizon, axis=0)
    trajectory_terminals = self._store["terminal"][trajectory_indices, trajectory_b_indices]
    trajectory_terminals[0, :] = 0
    is_terminal_transition = trajectory_terminals.any(0)
    valid_mask = (1 - trajectory_terminals).cumprod(0)
    trajectory_discount_vector = valid_mask * cumulative_discount_vector[:update_horizon, None]
    trajectory_rewards = self._store["reward"][(trajectory_indices + 1) % self._replay_length, trajectory_b_indices]
    returns = np.cumsum(trajectory_discount_vector * trajectory_rewards, axis=0)
    update_horizons = np.ones(batch_size * subseq_len, dtype=np.int32) * (update_horizon - 1)
    returns = returns[update_horizons, np.arange(batch_size * subseq_len)]
    next_indices = (state_indices + update_horizons) % self._replay_length
    outputs: dict[str, np.ndarray | torch.Tensor] = {}
    for element in self.get_transition_elements(batch_size, subseq_len):
      name = element.name
      if name == "state":
        output = self.parallel_get_stack("observation", state_indices, b_indices, censor_before)
        output = self.restore_leading_dims(batch_size, subseq_len, output)
      elif name == "return":
        output = self.restore_leading_dims(batch_size, subseq_len, returns)
      elif name == "discount":
        output = self.restore_leading_dims(
            batch_size, subseq_len, cumulative_discount_vector[update_horizons + 1]
        )
      elif name == "next_state":
        output = self.parallel_get_stack("observation", next_indices, b_indices, censor_before)
        output = self.restore_leading_dims(batch_size, subseq_len, output)
      elif name == "same_trajectory":
        output = self._store["terminal"][state_indices, b_indices]
        output = self.restore_leading_dims(batch_size, subseq_len, output)
        output[:, 0] = 0
        output = (1 - output).cumprod(1)
      elif name in ("next_action", "next_reward"):
        output = self._store[name.lstrip("next_")][next_indices, b_indices]
        output = self.restore_leading_dims(batch_size, subseq_len, output)
      elif name == "terminal":
        output = self.restore_leading_dims(batch_size, subseq_len, is_terminal_transition)
      elif name == "indices":
        output = self.ravel_indices(state_indices, b_indices).astype(np.int32)
        output = self.restore_leading_dims(batch_size, subseq_len, output)[:, 0]
      elif name in self._store:
        output = self._store[name][state_indices, b_indices]
        output = self.restore_leading_dims(batch_size, subseq_len, output)
      else:
        continue
      outputs[name] = self._to_torch(output, device) if as_torch else output
    return outputs

  def sample(self, *args: Any, **kwargs: Any) -> dict[str, np.ndarray | torch.Tensor]:
    return self.sample_transition_batch(*args, **kwargs)

  def _to_torch(
      self,
      array: np.ndarray,
      device: torch.device | str | None,
  ) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(array))
    return tensor if device is None else tensor.to(device=device)

  def reset_priorities(self) -> None:
    return None


class PrioritizedSubsequenceReplayBuffer(SubsequenceReplayBuffer):
  """Prioritized variant with deterministic stratified sampling."""

  def __init__(self, *args: Any, **kwargs: Any) -> None:
    super().__init__(*args, **kwargs)
    self.sum_tree = DeterministicSumTree(int(self._replay_capacity))

  def get_add_args_signature(self) -> list[ReplayElement]:
    return super().get_add_args_signature() + [ReplayElement("priority", (), np.float32)]

  def _add(self, *args: Any) -> None:
    self._check_args_length(*args)
    transition = {}
    priority = None
    for i, element in enumerate(self.get_add_args_signature()):
      if element.name == "priority":
        priority = args[i]
      else:
        transition[element.name] = args[i]
    if priority is None:
      raise ValueError("priority must be provided for prioritized replay.")
    indices = np.ravel_multi_index(
        (np.ones((1,), dtype=np.int32) * self.cursor(), np.arange(self._n_envs)),
        (self._replay_length, self._n_envs),
    )
    priority = np.asarray(priority, dtype=np.float32)
    for i, index in enumerate(indices):
      self.sum_tree.set(int(index), float(priority[i]))
    super()._add_transition(transition)

  def sample_index_batch(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.asarray(self.sum_tree.stratified_sample(batch_size, rng=self._rng))
    t_indices, b_indices = self.unravel_indices(indices)
    censor_before = np.zeros_like(t_indices)
    allowed_attempts = self._max_sample_attempts
    for i in range(len(indices)):
      is_valid, ep_start = self.is_valid_transition(t_indices[i:i + 1], b_indices[i:i + 1])
      censor_before[i] = ep_start
      if not is_valid:
        if allowed_attempts == 0:
          raise RuntimeError(
              f"Max sample attempts: Tried {self._max_sample_attempts} times but only sampled "
              f"{i} valid indices. Batch size is {batch_size}"
          )
        while not is_valid and allowed_attempts > 0:
          index = int(self.sum_tree.stratified_sample(1, rng=self._rng)[0])
          t_index, b_index = self.unravel_indices(np.array(index))
          allowed_attempts -= 1
          t_indices[i] = t_index
          b_indices[i] = b_index
          is_valid, ep_start = self.is_valid_transition(t_indices[i:i + 1], b_indices[i:i + 1])
          censor_before[i] = ep_start
    return t_indices, b_indices, censor_before

  def sample_transition_batch(self, *args: Any, **kwargs: Any) -> dict[str, np.ndarray | torch.Tensor]:
    transition = super().sample_transition_batch(*args, **kwargs)
    indices = transition["indices"]
    if isinstance(indices, torch.Tensor):
      priorities = self.get_priority(indices.cpu().numpy().astype(np.int32))
      transition["sampling_probabilities"] = torch.from_numpy(priorities).to(indices.device)
    else:
      transition["sampling_probabilities"] = self.get_priority(indices.astype(np.int32))
    return transition

  def set_priority(self, indices: np.ndarray, priorities: np.ndarray) -> None:
    indices = np.asarray(indices, dtype=np.int32)
    priorities = np.asarray(priorities, dtype=np.float32)
    for index, priority in zip(indices, priorities):
      self.sum_tree.set(int(index), float(priority))

  def get_priority(self, indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int32)
    return self.sum_tree.get(indices).astype(np.float32)

  def reset_priorities(self) -> None:
    self.sum_tree.reset_priorities()
