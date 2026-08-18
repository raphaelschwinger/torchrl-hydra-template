# Adapted from https://github.com/nicklashansen/tdmpc2 (tdmpc2/tdmpc2.py:
# `_plan` and `_estimate_value`), MIT license. Changes: single-task only,
# device-agnostic, `cfg` replaced by explicit arguments, and the planning loop
# split into helper methods.
"""MPPI planner over the TD-MPC2 latent world model."""
from __future__ import annotations

import torch
import torch.nn as nn

from src.components import math

from .world_model import WorldModel


class MPPIPlanner(nn.Module):
    """Model Predictive Path Integral planning in latent space.

    Holds the MPPI warm-start state (``_prev_mean``); shared between the
    exploration and evaluation policies. Plans for a single environment
    (batch size 1), matching the upstream implementation.
    """

    def __init__(
        self,
        model: WorldModel,
        *,
        action_dim: int,
        discount: float,
        horizon: int = 3,
        iterations: int = 6,
        num_samples: int = 512,
        num_elites: int = 64,
        num_pi_trajs: int = 24,
        min_std: float = 0.05,
        max_std: float = 2.0,
        temperature: float = 0.5,
        device: torch.device | str = "cpu",
    ) -> None:
        super().__init__()
        self.model = model
        self.action_dim = action_dim
        self.horizon = horizon
        # Heuristic for large action spaces (upstream tdmpc2.py:35).
        self.iterations = iterations + 2 * int(action_dim >= 20)
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_pi_trajs = num_pi_trajs
        self.min_std, self.max_std = min_std, max_std
        self.temperature = temperature
        self.register_buffer("discount", torch.as_tensor(discount, device=device))
        self._prev_mean = nn.Buffer(torch.zeros(horizon, action_dim, device=device))

    @torch.no_grad()
    def plan(self, obs: torch.Tensor, t0: bool = False, eval_mode: bool = False) -> torch.Tensor:
        """Plan an action sequence with MPPI and return the first action."""
        z = self.model.encode(obs)
        pi_actions = self._pi_rollout(z) if self.num_pi_trajs > 0 else None

        z = z.repeat(self.num_samples, 1)
        mean, std = self._init_mean_std(t0)
        actions = torch.empty(
            self.horizon, self.num_samples, self.action_dim, device=mean.device
        )
        if pi_actions is not None:
            actions[:, : self.num_pi_trajs] = pi_actions

        for _ in range(self.iterations):
            mean, std, score, elite_actions = self._mppi_iteration(z, actions, mean, std)

        return self._select_action(score, elite_actions, mean, std, eval_mode)

    def _pi_rollout(self, z: torch.Tensor) -> torch.Tensor:
        """Roll out the policy prior to seed candidate action sequences."""
        pi_actions = torch.empty(
            self.horizon, self.num_pi_trajs, self.action_dim, device=z.device
        )
        _z = z.repeat(self.num_pi_trajs, 1)
        for t in range(self.horizon - 1):
            pi_actions[t], _ = self.model.pi(_z)
            _z = self.model.next(_z, pi_actions[t])
        pi_actions[-1], _ = self.model.pi(_z)
        return pi_actions

    def _init_mean_std(self, t0: bool) -> tuple[torch.Tensor, torch.Tensor]:
        """Initialize sampling distribution; warm-start from the previous plan."""
        device = self._prev_mean.device
        mean = torch.zeros(self.horizon, self.action_dim, device=device)
        std = torch.full(
            (self.horizon, self.action_dim), self.max_std, dtype=torch.float, device=device
        )
        if not t0:
            mean[:-1] = self._prev_mean[1:]
        return mean, std

    def _mppi_iteration(self, z, actions, mean, std):
        """One MPPI iteration: sample, evaluate, refit mean/std to elites."""
        r = torch.randn(
            self.horizon, self.num_samples - self.num_pi_trajs, self.action_dim,
            device=std.device,
        )
        actions[:, self.num_pi_trajs:] = (mean.unsqueeze(1) + std.unsqueeze(1) * r).clamp(-1, 1)

        value = self._estimate_value(z, actions).nan_to_num(0)
        elite_idxs = torch.topk(value.squeeze(1), self.num_elites, dim=0).indices
        elite_value, elite_actions = value[elite_idxs], actions[:, elite_idxs]

        max_value = elite_value.max(0).values
        score = torch.exp(self.temperature * (elite_value - max_value))
        score = score / score.sum(0)
        mean = (score.unsqueeze(0) * elite_actions).sum(dim=1) / (score.sum(0) + 1e-9)
        std = (
            (score.unsqueeze(0) * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1)
            / (score.sum(0) + 1e-9)
        ).sqrt().clamp(self.min_std, self.max_std)
        return mean, std, score, elite_actions

    def _select_action(self, score, elite_actions, mean, std, eval_mode: bool) -> torch.Tensor:
        """Sample one elite trajectory and return its first action (+ noise if exploring)."""
        rand_idx = math.gumbel_softmax_sample(score.squeeze(1))
        actions = torch.index_select(elite_actions, 1, rand_idx).squeeze(1)
        a, std = actions[0], std[0]
        if not eval_mode:
            a = a + std * torch.randn(self.action_dim, device=std.device)
        self._prev_mean.copy_(mean)
        return a.clamp(-1, 1)

    @torch.no_grad()
    def _estimate_value(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Estimate the return of executing ``actions`` from latent state ``z``."""
        G, discount = 0, 1
        for t in range(self.horizon):
            reward = math.two_hot_inv(
                self.model.reward(z, actions[t]),
                self.model.num_bins, self.model.vmin, self.model.vmax,
            )
            z = self.model.next(z, actions[t])
            G = G + discount * reward
            discount = discount * self.discount
        action, _ = self.model.pi(z)
        return G + discount * self.model.Q(z, action, return_type="avg")
