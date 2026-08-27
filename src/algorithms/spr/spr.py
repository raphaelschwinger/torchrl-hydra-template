"""Self-Predictive Representations (SPR) agent preset in PyTorch."""

from __future__ import annotations

import dataclasses

from src.algorithms.common.spr_bbf import SPRBBFAgent
from src.algorithms.common.spr_bbf import SPRBBFConfig


@dataclasses.dataclass
class SPRConfig(SPRBBFConfig):
  """Reference-style SPR configuration.

  The implementation reuses the shared SPR/BBF's SPR training path, but the preset disables
  BBF's periodic reset machinery and uses the DQN-scale encoder/head from SPR.
  """


class SPRAgent(SPRBBFAgent):
  """SPR is BBF's auxiliary prediction path without shared reset resets."""

  config: SPRConfig


@dataclasses.dataclass
class SRSPRConfig(SPRBBFConfig):
  """Shrink-and-Reset SPR configuration."""


class SRSPRAgent(SPRBBFAgent):
  """SR-SPR adds BBF-style shrink-and-reset to the SPR backbone."""

  config: SRSPRConfig
