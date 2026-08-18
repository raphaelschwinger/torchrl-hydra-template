# Adapted from https://github.com/nicklashansen/tdmpc2 (tdmpc2/tdmpc2.py: `act`),
# MIT license. Changes: wrapped as a TensorDictModule so torchrl's Collector and
# rollout utilities can drive it; episode starts are detected via `is_init`
# (written by torchrl's InitTracker transform) instead of an explicit `t0` flag.
"""TensorDict-compatible policy wrappers around the TD-MPC2 planner / prior."""
from __future__ import annotations

import torch
import torch.nn as nn
from tensordict.nn import TensorDictModule

from .planner import MPPIPlanner
from .world_model import WorldModel


class TDMPC2PolicyModule(nn.Module):
    """Selects actions by MPPI planning (or the policy prior if ``mpc=False``).

    Batch-1 only: TD-MPC2 plans for a single environment, like upstream.
    """

    def __init__(
        self,
        model: WorldModel,
        planner: MPPIPlanner,
        *,
        mpc: bool = True,
        eval_mode: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.planner = planner
        self.mpc = mpc
        self.eval_mode = eval_mode

    @torch.no_grad()
    def forward(self, obs: torch.Tensor, is_init: torch.Tensor) -> torch.Tensor:
        if obs.shape[:-1].numel() > 1:
            raise ValueError("TD-MPC2 plans for a single environment (batch size 1).")
        obs = obs.reshape(1, -1)
        t0 = bool(is_init.any())
        if self.mpc:
            action = self.planner.plan(obs, t0=t0, eval_mode=self.eval_mode)
        else:
            z = self.model.encode(obs)
            action, info = self.model.pi(z)
            if self.eval_mode:
                action = info["mean"]
            action = action[0]
        return action.reshape(*is_init.shape[:-1], -1) if is_init.ndim > 1 else action.flatten()


def make_policy(
    model: WorldModel,
    planner: MPPIPlanner,
    obs_key: str,
    *,
    mpc: bool = True,
    eval_mode: bool = False,
) -> TensorDictModule:
    """Wrap the planner as a TensorDictModule reading ``obs_key`` and ``is_init``."""
    return TensorDictModule(
        TDMPC2PolicyModule(model, planner, mpc=mpc, eval_mode=eval_mode),
        in_keys=[obs_key, "is_init"],
        out_keys=["action"],
    )
