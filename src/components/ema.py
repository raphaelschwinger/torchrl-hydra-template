"""Polyak / exponential-moving-average parameter updates."""
from __future__ import annotations

from typing import Iterable

import torch


@torch.no_grad()
def polyak_update(
    source_params: Iterable[torch.nn.Parameter],
    target_params: Iterable[torch.nn.Parameter],
    mix: float,
) -> None:
    """In-place EMA update: ``target = mix * source + (1 - mix) * target``."""
    for s, t in zip(source_params, target_params):
        t.data.copy_(mix * s.data + (1 - mix) * t.data)
