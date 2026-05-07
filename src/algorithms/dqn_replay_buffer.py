"""Replay buffer builders for DQN (Hydra `instantiate`-friendly)."""

from __future__ import annotations

import torch

from torchrl.data import LazyMemmapStorage, LazyTensorStorage, TensorDictReplayBuffer


def build_dqn_replay_buffer(
    *,
    trainer_device: torch.device | str,
    capacity: int,
    batch_size: int,
    storage: str = "tensor",
    storage_device: str | torch.device | None = None,
    scratch_dir: str | None = None,
    prefetch: int | None = None,
    pin_memory: bool = False,
) -> TensorDictReplayBuffer:
    """Construct a :class:`~torchrl.data.TensorDictReplayBuffer` from shorthand config.

    ``storage_device`` follows algorithm YAML: ``null`` means use ``trainer_device``.

    Override in Hydra by replacing ``replay_buffer._target_`` with any callable
    with a compatible signature (or a TorchRL class plus nested ``storage`` /
    ``sampler`` config without this shorthand).
    """
    resolved = storage_device if storage_device is not None else trainer_device
    dev: str | torch.device = resolved

    kind = storage.lower().strip()
    if kind == "tensor":
        st = LazyTensorStorage(max_size=capacity, device=dev)
    elif kind == "mmap":
        st = LazyMemmapStorage(
            max_size=capacity,
            scratch_dir=scratch_dir,
            device=dev,
        )
    else:
        raise ValueError(
            f"replay_buffer.storage must be 'tensor' or 'mmap', got {storage!r}"
        )

    prefetch_kw = prefetch if prefetch else None

    return TensorDictReplayBuffer(
        storage=st,
        batch_size=batch_size,
        pin_memory=pin_memory,
        prefetch=prefetch_kw,
    )
