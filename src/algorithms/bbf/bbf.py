"""Bigger Better Faster (BBF) agent preset in PyTorch."""

from __future__ import annotations

import dataclasses

from src.algorithms.common.spr_bbf import SPRBBFAgent
from src.algorithms.common.spr_bbf import SPRBBFConfig


@dataclasses.dataclass
class BBFConfig(SPRBBFConfig):
  """Reference-style BBF configuration."""


class BBFAgent(SPRBBFAgent):
  """BBF uses the shared SPR/BBF core with periodic reset defaults enabled."""

  config: BBFConfig
