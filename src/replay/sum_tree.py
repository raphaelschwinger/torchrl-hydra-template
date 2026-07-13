"""Deterministic sum tree utilities for prioritized replay."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


class DeterministicSumTree:
  """Binary sum tree with deterministic stratified sampling.

  This mirrors the behavior of the JAX-based replay sum tree but uses NumPy
  random generators so it can serve as the foundation for the PyTorch port.
  """

  def __init__(self, capacity: int) -> None:
    if not isinstance(capacity, int) or capacity <= 0:
      raise ValueError(f"Sum tree capacity should be a positive int, got {capacity!r}.")
    self.capacity = capacity
    self.depth = int(math.ceil(math.log2(capacity)))
    self.low_idx = (2**self.depth) - 1
    self.high_idx = capacity + self.low_idx
    self.nodes = np.zeros(2 ** (self.depth + 1) - 1, dtype=np.float64)
    self.highest_set = 0
    self.max_recorded_priority = 1.0

  @property
  def total_priority(self) -> float:
    return float(self.nodes[0])

  def get(self, node_indices: np.ndarray | int) -> np.ndarray:
    indices = np.asarray(node_indices, dtype=np.int64)
    return self.nodes[indices + self.low_idx]

  def _sample_from_query(self, query_value: float) -> int:
    if self.total_priority <= 0.0:
      raise RuntimeError("Cannot sample from an empty sum tree.")
    scaled_query = query_value * self.total_priority
    index = 0
    for _ in range(self.depth):
      left_child = index * 2 + 1
      left_sum = self.nodes[left_child]
      if scaled_query < left_sum:
        index = left_child
      else:
        scaled_query -= left_sum
        index = left_child + 1
    return min(index - self.low_idx, self.highest_set)

  def sample(
      self,
      rng: np.random.Generator | None = None,
      query_value: float | None = None,
  ) -> int:
    if query_value is None:
      rng = rng or np.random.default_rng()
      query_value = float(rng.random())
    return self._sample_from_query(float(query_value))

  def stratified_sample(
      self,
      batch_size: int,
      rng: np.random.Generator | None = None,
  ) -> np.ndarray:
    if batch_size <= 0:
      raise ValueError(f"batch_size must be positive, got {batch_size}.")
    if self.total_priority <= 0.0:
      raise RuntimeError("Cannot sample from an empty sum tree.")
    rng = rng or np.random.default_rng()
    query_values = (np.arange(batch_size, dtype=np.float64) + rng.random(batch_size)) / batch_size
    scaled_queries = query_values * self.total_priority
    indices = np.zeros(batch_size, dtype=np.int64)
    for _ in range(self.depth):
      left_children = indices * 2 + 1
      left_sums = self.nodes[left_children]
      go_right = scaled_queries >= left_sums
      scaled_queries = np.where(go_right, scaled_queries - left_sums, scaled_queries)
      indices = np.where(go_right, left_children + 1, left_children)
    return np.minimum(indices - self.low_idx, self.highest_set).astype(np.int64)

  def set(self, node_index: int, value: float) -> None:
    if value < 0.0:
      raise ValueError(f"Sum tree values should be nonnegative, got {value}.")
    self.highest_set = max(node_index, self.highest_set)
    tree_index = node_index + self.low_idx
    self.max_recorded_priority = max(float(value), self.max_recorded_priority)
    delta = float(value) - self.nodes[tree_index]
    for _ in reversed(range(self.depth)):
      self.nodes[tree_index] += delta
      tree_index = (tree_index - 1) // 2
    self.nodes[tree_index] += delta
    if tree_index != 0:
      raise RuntimeError("Sum tree traversal failed to reach the root.")

  def reset_priorities(self) -> None:
    for i in range(self.highest_set + 1):
      self.set(i, self.max_recorded_priority)
