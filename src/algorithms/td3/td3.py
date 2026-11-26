# Adapted from https://github.com/sfujim/TD3 (TD3.py, main.py), MIT license.
# Changes: written as a subclass of this template's SAC (replay buffer, double
# critic, optimiser plumbing and runtime knobs are SAC's), PyTorch 2 instead of
# the original's PyTorch 1, and DMC's [-1, 1] action bounds (max_action = 1).
"""TD3: Twin Delayed Deep Deterministic policy gradient.

Fujimoto et al. (2018), https://arxiv.org/abs/1802.09477

Pseudocode (per agent step, after ``init_random_frames`` random steps):
    Act a = clip(pi(s) + N(0, sigma_explore), -1, 1);  store (s, a, r, s')
    Repeat ``num_updates`` times on a fresh uniform minibatch:
        Target smoothing: a' = clip(pi'(s') + clip(N(0, sigma_target), -c, c), -1, 1)
        Critic: sum_i (Q_i(s, a) - y)^2,  y = r + gamma min(Q1'(s', a'), Q2'(s', a'))
        Every ``policy_delay`` critic updates:
            Actor: minimise -Q1(s, pi(s))
            Polyak-update the target actor and the target critics

What TD3 changes in SAC: a deterministic tanh actor with a target copy
instead of the tanh-Gaussian actor, Gaussian exploration noise, target policy
smoothing in place of the entropy term, no temperature, the actor maximises
Q1 only, and the actor and target updates are delayed.
"""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.algorithms.sac.networks import DoubleCritic
from src.algorithms.sac.sac import SACAlgorithm

from .networks import DeterministicActor, TorchMLP


class TD3Networks(nn.Module):
    """Actor, target actor, double critic and target critic, in one module."""

    def __init__(self, obs_dim: int, action_dim: int, algo: "TD3Algorithm") -> None:
        super().__init__()
        self.actor = DeterministicActor(
            obs_dim, action_dim, trunk=algo.actor_trunk, hidden_dim=algo.actor_hidden_dim, depth=algo.actor_depth,
        )
        self.critic = DoubleCritic(
            obs_dim, action_dim, trunk=algo.critic_trunk, hidden_dim=algo.critic_hidden_dim,
            depth=algo.critic_depth,
        )
        self.target_actor = copy.deepcopy(self.actor).requires_grad_(False)
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)


class TD3PolicyModule(nn.Module):
    """``pi(s)`` for evaluation, ``clip(pi(s) + N(0, noise), -1, 1)`` for exploration."""

    def __init__(self, nets: TD3Networks, noise: float) -> None:
        super().__init__()
        self.nets = nets
        self.noise = noise

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        action = self.nets.actor(obs)
        if self.noise > 0:
            action = action + self.noise * torch.randn_like(action)
        return action.clamp(-1.0, 1.0)


class TD3Algorithm(SACAlgorithm):
    """TD3 as a SAC subclass. Defaults are Fujimoto's reference ``TD3.py`` /
    ``main.py``: MLP 256x2 (default PyTorch init), Adam 3e-4, batch 256,
    gamma 0.99, tau 0.005, exploration noise 0.1, target noise 0.2 clipped
    at 0.5, policy delay 2 and 25k random warm-up steps.
    """

    actor_trunk = staticmethod(TorchMLP)
    critic_trunk = staticmethod(TorchMLP)

    def __init__(
        self,
        device: torch.device | None = None,
        *,
        # --- SAC defaults that TD3 changes ------------------------------------
        init_random_frames: int = 25_000,  # main.py start_timesteps
        # --- TD3 only ---------------------------------------------------------
        lr: float = 3e-4,  # actor and critic
        exploration_noise: float = 0.1,
        target_noise: float = 0.2,  # policy_noise
        target_noise_clip: float = 0.5,  # noise_clip
        policy_delay: int = 2,  # policy_freq
        **sac_kwargs,
    ) -> None:
        for unused in ("actor_lr", "critic_lr", "temp_lr", "init_temperature", "target_entropy",
                       "log_std_min", "log_std_max", "pessimism"):
            if unused in sac_kwargs:
                raise TypeError(f"TD3 has no {unused!r} (it is a SAC-only keyword)")
        super().__init__(device, init_random_frames=init_random_frames, actor_lr=lr, critic_lr=lr, **sac_kwargs)
        if self.compile and self.compile_scope == "step":
            raise ValueError("TD3 supports compile_scope='update' only (the policy delay branches per update)")
        self.lr = lr
        self.exploration_noise = exploration_noise
        self.target_noise, self.target_noise_clip = target_noise, target_noise_clip
        self.policy_delay = policy_delay
        self._critic_updates = 0
        self._actor_info: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # What TD3 swaps in SAC's setup
    # ------------------------------------------------------------------

    def _build_networks(self, obs_dim: int, action_dim: int) -> nn.Module:
        return TD3Networks(obs_dim, action_dim, self)

    def _make_explore_policy(self) -> nn.Module:
        return TD3PolicyModule(self.nets, self.exploration_noise)

    def _make_eval_policy(self) -> nn.Module:
        return TD3PolicyModule(self.nets, 0.0)

    def _optimizer_specs(self) -> dict[str, tuple[list[nn.Parameter], dict]]:
        n = self.nets
        return {
            "actor": (list(n.actor.parameters()), {"lr": self.lr, "wd": self.weight_decay}),
            "critic": (list(n.critic.parameters()), {"lr": self.lr, "wd": self.weight_decay}),
        }

    def _compile_update(self, mode: str | None) -> None:
        self._update_critic = torch.compile(self._update_critic, mode=mode)
        self._update_actor_and_targets = torch.compile(self._update_actor_and_targets, mode=mode)

    # ------------------------------------------------------------------
    # The TD3 update
    # ------------------------------------------------------------------

    def _update(self, mb) -> dict[str, torch.Tensor]:
        """Critic every update; actor and targets every ``policy_delay`` updates."""
        obs, action = mb[self.obs_key], mb["action"]
        reward = mb["next", "reward"].reshape(-1)
        mask = 1.0 - mb["next", "terminated"].float().reshape(-1)
        info = self._update_critic(obs, action, reward, mask, mb["next", self.obs_key])
        self._critic_updates += 1
        if self._critic_updates % self.policy_delay == 0:
            self._actor_info = self._update_actor_and_targets(obs)
        return {**info, **self._actor_info}

    def _update_critic(self, obs, action, reward, mask, next_obs) -> dict[str, torch.Tensor]:
        n = self.nets
        with torch.no_grad(), self._autocast():
            next_action = n.target_actor(next_obs).float()
            noise = (self.target_noise * torch.randn_like(next_action)).clamp(
                -self.target_noise_clip, self.target_noise_clip
            )
            next_action = (next_action + noise).clamp(-1.0, 1.0)
            nq1, nq2 = n.target_critic(next_obs, next_action)
            target = reward + self.discount * mask * torch.min(nq1.float(), nq2.float())
        with self._autocast():
            q1, q2 = n.critic(obs, action)
        loss = self._critic_loss(q1.float(), q2.float(), target)
        self._step_optim("critic", loss)
        return {"critic_loss": loss.detach(), "q1": q1.detach().float().mean(), "q2": q2.detach().float().mean()}

    def _update_actor_and_targets(self, obs) -> dict[str, torch.Tensor]:
        with self._autocast():
            action = self.nets.actor(obs)
        q1 = self._q1_value(obs, action.float())
        loss = -q1.mean()
        self._step_optim("actor", loss)
        self._soft_update_target()
        return {"actor_loss": loss.detach()}

    def _q1_value(self, obs, action) -> torch.Tensor:
        """Q1 of ``action`` with the critic weights detached (gradient to the action only)."""
        q1_net = self.nets.critic.q1
        params = {k: v.detach() for k, v in q1_net.named_parameters()}
        with self._autocast():
            q1 = torch.func.functional_call(q1_net, params, (torch.cat([obs, action], dim=-1),))
        return q1.float().squeeze(-1)

    @torch.no_grad()
    def _soft_update_target(self) -> None:
        n = self.nets
        torch._foreach_lerp_(
            [*n.target_critic.parameters(), *n.target_actor.parameters()],
            [*n.critic.parameters(), *n.actor.parameters()],
            self.tau,
        )

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _extra_state(self) -> dict:
        return {**super()._extra_state(), "critic_updates": self._critic_updates}

    def _load_extra_state(self, extra: dict) -> None:
        super()._load_extra_state(extra)
        self._critic_updates = int(extra.get("critic_updates", self._critic_updates))
