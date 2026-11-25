# Adapted from https://github.com/naumix/BiggerRegularizedOptimistic
# (jaxrl/bro/bro_learner.py, jaxrl/bro/actor.py, jaxrl/bro/critic.py,
# jaxrl/bro/temperature.py, train_parallel.py), MIT license.
# Changes: PyTorch instead of JAX, one seed per process instead of vmapped
# seeds, and written as a subclass of this template's SAC: only what BRO adds
# to SAC is here.
"""BRO: Bigger, Regularized, Optimistic.

Nauman et al. (2024), https://arxiv.org/abs/2405.16158

BRO is SAC (``src.algorithms.sac``) with:
    Bigger      BroNet trunks (residual, LayerNorm), critic width 512 x 2 blocks
    Regularized AdamW (weight decay 1e-4) and full resets at fixed agent steps
    Optimistic  an exploration actor pi_o that maximises Q_ub = mean(Q) + beta_ub |Q1 - Q2| / 2,
                KL-tied to the SAC actor; beta_ub and the KL weight are dual variables
    plus a quantile critic (100 quantiles), no clipped double-Q (pessimism 0)
    and replay ratio 10.

Per gradient step (after SAC's critic, target, actor and temperature updates):
    pi_o:     minimise -mean(Q_ub(s, a_o)) + lambda KL(pi_o || pi_c),  a_o ~ pi_o(s)
    duals:    beta_ub, lambda driven to a per-dimension KL target of 0.05
"""
from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from src.algorithms.sac.networks import tanh_normal_sample
from src.algorithms.sac.sac import SACAlgorithm, SACNetworks

from .networks import BoundedLogParam, BroNet, DualTanhActor

#: Official reset schedule (agent steps); BRO (Fast) at replay ratio 2 keeps the first.
RESET_STEPS = (15_001, 50_001, 250_001, 500_001, 750_001, 1_000_001, 1_500_001, 2_000_001)


class BRONetworks(SACNetworks):
    """SAC's networks plus the optimistic actor and the two dual variables."""

    def __init__(self, obs_dim: int, action_dim: int, algo: "BROAlgorithm") -> None:
        super().__init__(obs_dim, action_dim, algo)
        self.actor_o = DualTanhActor(obs_dim, action_dim, hidden_dim=algo.actor_hidden_dim, depth=algo.actor_depth)
        self.optimism = BoundedLogParam(algo.init_optimism)
        self.regularizer = BoundedLogParam(algo.init_regularizer)


class BROPolicyModule(nn.Module):
    """Exploration with the optimistic actor (evaluation stays SAC's ``tanh(mean)``)."""

    def __init__(self, nets: BRONetworks, std_multiplier: float) -> None:
        super().__init__()
        self.nets = nets
        self.std_multiplier = std_multiplier

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        mean_c, std_c = self.nets.actor(obs)
        mean_o, std_o = self.nets.actor_o(obs, mean_c, std_c, self.std_multiplier)
        action, _ = tanh_normal_sample(mean_o, std_o)
        return action.clamp(-1.0, 1.0)


class BROAlgorithm(SACAlgorithm):
    """BRO as a SAC subclass. Defaults reproduce the official ``train_parallel.py``
    + ``configs/bro_default.py``; every other SAC keyword is accepted unchanged.
    """

    actor_trunk = staticmethod(BroNet)
    critic_trunk = staticmethod(BroNet)

    def __init__(
        self,
        device: torch.device | None = None,
        *,
        # --- SAC defaults that BRO changes ------------------------------------
        critic_hidden_dim: int = 512,
        critic_depth: int = 2,  # residual blocks
        actor_hidden_dim: int = 256,
        actor_depth: int = 1,  # residual blocks
        weight_decay: float = 1e-4,  # AdamW (optax.adamw default)
        batch_size: int = 128,
        pessimism: float = 0.0,  # mean(Q1, Q2), not min
        init_random_frames: int = 2_500,
        num_updates: int = 10,  # replay ratio
        # --- BRO only ---------------------------------------------------------
        distributional: bool = True,
        n_quantiles: int = 100,
        adj_lr: float = 3e-5,  # optimism / KL-weight learning rate
        init_optimism: float = 1.0,
        init_regularizer: float = 0.25,
        kl_target: float = 0.05,
        std_multiplier: float = 0.75,
        reset_steps: tuple[int, ...] | None = None,  # None -> RESET_STEPS (first only at num_updates=2)
        **sac_kwargs,
    ) -> None:
        super().__init__(
            device,
            critic_hidden_dim=critic_hidden_dim, critic_depth=critic_depth,
            actor_hidden_dim=actor_hidden_dim, actor_depth=actor_depth,
            weight_decay=weight_decay, batch_size=batch_size, pessimism=pessimism,
            init_random_frames=init_random_frames, num_updates=num_updates,
            **sac_kwargs,
        )
        self.distributional, self.n_quantiles = distributional, n_quantiles
        self.adj_lr, self.init_optimism, self.init_regularizer = adj_lr, init_optimism, init_regularizer
        self.kl_target, self.std_multiplier = kl_target, std_multiplier
        if reset_steps is None:
            reset_steps = RESET_STEPS[:1] if num_updates == 2 else RESET_STEPS
        self.reset_steps = tuple(int(s) for s in reset_steps)
        self._num_resets = 0

    @property
    def critic_outputs(self) -> int:
        return self.n_quantiles if self.distributional else 1

    # ------------------------------------------------------------------
    # What BRO adds to SAC's setup
    # ------------------------------------------------------------------

    def setup(self, make_env) -> None:
        super().setup(make_env)
        taus = torch.arange(self.n_quantiles + 1, dtype=torch.float32) / self.n_quantiles
        self.quantile_taus = ((taus[1:] + taus[:-1]) / 2).to(self.device)

    def _build_networks(self, obs_dim: int, action_dim: int) -> nn.Module:
        return BRONetworks(obs_dim, action_dim, self)

    def _make_explore_policy(self) -> nn.Module:
        return BROPolicyModule(self.nets, self.std_multiplier)

    def _optimizer_specs(self):
        n = self.nets
        specs = super()._optimizer_specs()
        specs["actor_o"] = (list(n.actor_o.parameters()), {"lr": self.actor_lr, "wd": self.weight_decay})
        specs["duals"] = (
            [*n.optimism.parameters(), *n.regularizer.parameters()],
            {"lr": self.adj_lr, "betas": (0.5, 0.999)},
        )
        return specs

    # ------------------------------------------------------------------
    # Resets
    # ------------------------------------------------------------------

    def step(self, batch: TensorDict) -> dict[str, float]:
        super().step(batch)
        return {"train/num_resets": float(self._num_resets)}

    def _before_updates(self, prev_frames: int, frames: int) -> None:
        if any(prev_frames < s <= frames for s in self.reset_steps):
            self.reset()

    def reset(self) -> None:
        """Full reset to the initial weights with fresh optimisers (official ``reset``)."""
        self.nets.load_state_dict(self._init_state)
        self._setup_optimizers()
        self._num_resets += 1

    # ------------------------------------------------------------------
    # Quantile critic
    # ------------------------------------------------------------------

    def _td_target(self, next_q, reward, mask, alpha, next_log_prob) -> torch.Tensor:
        if not self.distributional:
            return super()._td_target(next_q, reward, mask, alpha, next_log_prob)
        # [B, 1, N_target]: each target quantile against every predicted one.
        r, m, lp = reward[:, None, None], mask[:, None, None], next_log_prob[:, None, None]
        return r + self.discount * m * (next_q[:, None, :] - alpha * lp)

    def _critic_loss(self, q1, q2, target) -> torch.Tensor:
        if not self.distributional:
            return super()._critic_loss(q1, q2, target)
        return self._quantile_huber(target - q1[..., None]) + self._quantile_huber(target - q2[..., None])

    def _quantile_huber(self, td: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
        """Quantile-Huber loss; ``td`` is [B, N_pred, N_target], summed over predictions."""
        abs_td = td.abs()
        huber = torch.where(abs_td <= kappa, 0.5 * td**2, kappa * (abs_td - 0.5 * kappa))
        weight = (self.quantile_taus[None, :, None] - (td.detach() < 0).float()).abs()
        return (weight * huber / kappa).sum(dim=1).mean()

    def _reduce_q(self, q: torch.Tensor) -> torch.Tensor:
        return q.mean(-1) if self.distributional else q

    # ------------------------------------------------------------------
    # Optimism: what BRO adds to SAC's update
    # ------------------------------------------------------------------

    def _update(self, mb: TensorDict) -> dict[str, torch.Tensor]:
        info = super()._update(mb)  # critic, target, actor, temperature
        info.update(self._update_actor_optimistic(mb[self.obs_key]))
        info.update(self._update_duals(info["kl"]))
        return info

    def _update_actor_optimistic(self, obs) -> dict[str, torch.Tensor]:
        n = self.nets
        with torch.no_grad(), self._autocast():  # the freshly updated SAC actor
            mean_c, std_c = n.actor(obs)
        mean_c, std_c = mean_c.float(), std_c.float()
        with self._autocast():
            mean_o, std_o = n.actor_o(obs, mean_c, std_c, self.std_multiplier)
        mean_o, std_o = mean_o.float(), std_o.float()
        action, _ = tanh_normal_sample(mean_o, std_o)
        q_ub = self._q_value(obs, action, n.optimism().detach())
        s = std_o / self.std_multiplier
        kl = (torch.log(std_c / s) + (s**2 + (mean_o - mean_c) ** 2) / (2 * std_c**2) - 0.5).sum(-1)
        loss = -q_ub.mean() + n.regularizer().detach() * kl.mean()
        self._step_optim("actor_o", loss)
        return {"actor_o_loss": loss.detach(), "kl": kl.detach().mean()}

    def _update_duals(self, kl) -> dict[str, torch.Tensor]:
        """Optimism and KL weight towards the per-dimension KL target."""
        n = self.nets
        gap = kl / self.action_dim - self.kl_target
        optimism, kl_weight = n.optimism(), n.regularizer()
        self._step_optim("duals", (optimism - self.pessimism) * gap - kl_weight * gap)
        return {"optimism": optimism.detach(), "kl_weight": kl_weight.detach()}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _extra_state(self) -> dict:
        return {**super()._extra_state(), "num_resets": self._num_resets}

    def _load_extra_state(self, extra: dict) -> None:
        super()._load_extra_state(extra)
        self._num_resets = int(extra.get("num_resets", self._num_resets))
