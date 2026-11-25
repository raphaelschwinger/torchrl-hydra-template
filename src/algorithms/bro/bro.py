# Adapted from https://github.com/naumix/BiggerRegularizedOptimistic
# (jaxrl/bro/bro_learner.py, jaxrl/bro/actor.py, jaxrl/bro/critic.py,
# jaxrl/bro/temperature.py, train_parallel.py), MIT license.
# Changes: restructured as a BaseAlgorithm for this template (kwarg-only HPs,
# no `cfg`), PyTorch instead of JAX, one seed per process instead of vmapped
# seeds, transitions stored in a torchrl replay buffer, resets keyed on agent
# steps collected by the template's collector.
"""BRO: Bigger, Regularized, Optimistic.

Nauman et al. (2024), https://arxiv.org/abs/2405.16158

Pseudocode (per agent step, after ``init_random_frames`` random steps):
    Act with the optimistic actor  a ~ tanh N(mu_c + dmu_o, 0.75 * sigma_c)
    Store (s, a, r, s') in the replay buffer
    If the step is in ``reset_steps``: re-initialise every network and optimiser
    Repeat ``num_updates`` (= replay ratio) times on fresh uniform minibatches:
        Critic: quantile-Huber regression of both BroNet critics onto
            r + gamma * (mean_k Z_target_k(s', a') - alpha log pi(a'|s')),  a' ~ pi_c(s')
        Polyak-update the target critics
        Conservative actor pi_c: minimise alpha log pi_c - mean(Q_lb)
        Optimistic actor pi_o: minimise -mean(Q_ub) + lambda * KL(pi_o || pi_c)
            Q_ub = mean(Q1, Q2) + beta_ub * |Q1 - Q2| / 2
        Dual updates: alpha -> target entropy, beta_ub and lambda -> target KL
"""
from __future__ import annotations

import copy
from typing import Callable

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers import RandomSampler
from torchrl.envs import EnvBase

from src.algorithms.base import BaseAlgorithm, CollectorConfig, TrainingState

from .networks import (
    BoundedLogParam,
    DoubleCritic,
    DualTanhActor,
    LogParam,
    NormalTanhActor,
    tanh_normal_sample,
)


class _BROModules(nn.Module):
    """Every learnable piece of BRO, so a reset is one ``load_state_dict``."""

    def __init__(self, obs_dim: int, action_dim: int, algo: "BROAlgorithm") -> None:
        super().__init__()
        n_out = algo.n_quantiles if algo.distributional else 1
        self.actor = NormalTanhActor(
            obs_dim, action_dim, hidden_dim=algo.actor_hidden_dim, depth=algo.actor_depth,
            log_std_min=algo.log_std_min, log_std_max=algo.log_std_max,
        )
        self.actor_o = DualTanhActor(
            obs_dim, action_dim, hidden_dim=algo.actor_hidden_dim, depth=algo.actor_depth,
        )
        self.critic = DoubleCritic(
            obs_dim, action_dim, hidden_dim=algo.critic_hidden_dim, depth=algo.critic_depth, n_outputs=n_out,
        )
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_alpha = LogParam(algo.init_temperature)
        self.optimism = BoundedLogParam(algo.init_optimism)
        self.regularizer = BoundedLogParam(algo.init_regularizer)


class BROPolicyModule(nn.Module):
    """Optimistic sampling for exploration, ``tanh(mean)`` of pi_c for evaluation."""

    def __init__(self, nets: _BROModules, std_multiplier: float, *, eval_mode: bool) -> None:
        super().__init__()
        self.nets = nets
        self.std_multiplier = std_multiplier
        self.eval_mode = eval_mode

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        mean_c, std_c = self.nets.actor(obs)
        if self.eval_mode:
            return torch.tanh(mean_c)
        mean_o, std_o = self.nets.actor_o(obs, mean_c, std_c, self.std_multiplier)
        action, _ = tanh_normal_sample(mean_o, std_o)
        return action.clamp(-1.0, 1.0)


class BROAlgorithm(BaseAlgorithm):
    """BRO with a quantile BroNet critic ensemble, optimistic exploration and resets.

    Defaults reproduce the official ``train_parallel.py`` + ``bro_default.py``
    configuration (replay ratio 10, batch 128, 100 quantiles, resets).

    Like TD-MPC2, the networks are built in ``setup()`` from scalar
    hyperparameters: the optimistic actor consumes the conservative actor's
    mean, and a reset must restore all of them (and their optimisers) at once.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        *,
        obs_key: str = "observation",
        # --- Architecture ----------------------------------------------------
        critic_hidden_dim: int = 512,
        critic_depth: int = 2,
        actor_hidden_dim: int = 256,
        actor_depth: int = 1,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        distributional: bool = True,
        n_quantiles: int = 100,
        # --- Optimisation ----------------------------------------------------
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        adj_lr: float = 3e-5,
        weight_decay: float = 1e-4,  # optax.adamw default
        discount: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 128,
        # --- Entropy / optimism ----------------------------------------------
        init_temperature: float = 1.0,
        target_entropy: float | None = None,  # None -> -action_dim / 2
        init_optimism: float = 1.0,
        init_regularizer: float = 0.25,
        pessimism: float = 0.0,
        kl_target: float = 0.05,
        std_multiplier: float = 0.75,
        # --- Resets (agent steps; the official list, truncated to 1 at RR 2) --
        reset_steps: tuple[int, ...] | None = None,
        # --- Replay ----------------------------------------------------------
        buffer_size: int = 1_000_000,
        buffer_device: str = "cpu",
        # --- Data collection / learning schedule ----------------------------
        frames_per_batch: int = 1,
        init_random_frames: int = 2_500,
        num_updates: int = 10,  # replay ratio: gradient steps per agent step
    ) -> None:
        super().__init__(device)
        self.obs_key = obs_key
        self.critic_hidden_dim, self.critic_depth = critic_hidden_dim, critic_depth
        self.actor_hidden_dim, self.actor_depth = actor_hidden_dim, actor_depth
        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        self.distributional, self.n_quantiles = distributional, n_quantiles
        self.actor_lr, self.critic_lr, self.temp_lr, self.adj_lr = actor_lr, critic_lr, temp_lr, adj_lr
        self.weight_decay = weight_decay
        self.discount, self.tau, self.batch_size = discount, tau, batch_size
        self.init_temperature, self.target_entropy = init_temperature, target_entropy
        self.init_optimism, self.init_regularizer = init_optimism, init_regularizer
        self.pessimism, self.kl_target, self.std_multiplier = pessimism, kl_target, std_multiplier
        if reset_steps is None:
            reset_steps = (15_001, 50_001, 250_001, 500_001, 750_001, 1_000_001, 1_500_001, 2_000_001)
            if num_updates == 2:
                reset_steps = reset_steps[:1]
        self.reset_steps = tuple(int(s) for s in reset_steps)
        self.buffer_size, self.buffer_device = buffer_size, buffer_device
        self.frames_per_batch = frames_per_batch
        self.init_random_frames = init_random_frames
        self.num_updates = num_updates
        self._collected_frames = 0
        self._num_resets = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, make_env: Callable[[], EnvBase]) -> None:
        proof_env = make_env()
        obs_dim = int(proof_env.observation_spec[self.obs_key].shape[-1])
        self.action_dim = int(proof_env.action_spec.shape[-1])
        del proof_env
        if self.target_entropy is None:
            self.target_entropy = -self.action_dim / 2

        self.nets = _BROModules(obs_dim, self.action_dim, self).to(self.device)
        # A reset restores exactly the initial weights, like the official
        # `init_models(self.seeds)` with the same PRNG keys.
        self._init_state = copy.deepcopy(self.nets.state_dict())
        self._setup_optimizers()

        taus = torch.arange(self.n_quantiles + 1, dtype=torch.float32) / self.n_quantiles
        self.quantile_taus = ((taus[1:] + taus[:-1]) / 2).to(self.device)

        self._explore_policy = TensorDictModule(
            BROPolicyModule(self.nets, self.std_multiplier, eval_mode=False),
            in_keys=[self.obs_key], out_keys=["action"],
        )
        self._eval_policy = TensorDictModule(
            BROPolicyModule(self.nets, self.std_multiplier, eval_mode=True),
            in_keys=[self.obs_key], out_keys=["action"],
        )
        self._buffer_keys = (self.obs_key, "action", ("next", self.obs_key), ("next", "reward"), ("next", "terminated"))
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(self.buffer_size, device=self.buffer_device),
            sampler=RandomSampler(),
            batch_size=self.batch_size * self.num_updates,
        )

    def _setup_optimizers(self) -> None:
        n = self.nets
        wd = self.weight_decay
        self.actor_optim = torch.optim.AdamW(n.actor.parameters(), lr=self.actor_lr, weight_decay=wd, eps=1e-8)
        self.actor_o_optim = torch.optim.AdamW(n.actor_o.parameters(), lr=self.actor_lr, weight_decay=wd, eps=1e-8)
        self.critic_optim = torch.optim.AdamW(n.critic.parameters(), lr=self.critic_lr, weight_decay=wd, eps=1e-8)
        self.temp_optim = torch.optim.Adam(n.log_alpha.parameters(), lr=self.temp_lr, betas=(0.5, 0.999))
        self.adj_optim = torch.optim.Adam(
            [*n.optimism.parameters(), *n.regularizer.parameters()], lr=self.adj_lr, betas=(0.5, 0.999)
        )
        self._optims = (self.actor_optim, self.actor_o_optim, self.critic_optim, self.temp_optim, self.adj_optim)

    def reset(self) -> None:
        """Full reset: initial weights and fresh optimiser state (official ``reset``)."""
        self.nets.load_state_dict(self._init_state)
        self._setup_optimizers()
        self._num_resets += 1

    def get_collector_config(self) -> CollectorConfig:
        return CollectorConfig(
            frames_per_batch=self.frames_per_batch,
            init_random_frames=self.init_random_frames,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def step(self, batch: TensorDict) -> dict[str, float]:
        """Store the batch; after warm-up, reset on schedule and run ``num_updates`` updates."""
        batch = batch.reshape(-1)
        self.replay_buffer.extend(batch.select(*self._buffer_keys))
        prev, self._collected_frames = self._collected_frames, self._collected_frames + batch.numel()
        if self._collected_frames < self.init_random_frames:
            return {}
        if any(prev < s <= self._collected_frames for s in self.reset_steps):
            self.reset()

        # `num_updates` gradient steps per agent step, each on its own uniform
        # minibatch (official `sample_parallel_multibatch`): one buffer draw of
        # num_updates * batch_size, split.
        for _ in range(batch.numel()):
            sample = self.replay_buffer.sample().to(self.device, non_blocking=True)
            for mb in sample.split(self.batch_size):
                self._last_info = self._update(mb)
        # Diagnostics stay on the device until a logging boundary
        # (`pop_train_metrics`), so the update loop never forces a host sync.
        return {"train/num_resets": float(self._num_resets)}

    def pop_train_metrics(self) -> dict[str, float]:
        """Last update's diagnostics (like the official logger), read at log time."""
        info = getattr(self, "_last_info", None)
        if info is None:
            return {}
        self._last_info = None
        return {f"train/{k}": float(v) for k, v in info.items()}

    def _update(self, mb: TensorDict) -> dict[str, torch.Tensor]:
        """One BRO gradient step (official ``bro_learner._update``)."""
        obs, action = mb[self.obs_key], mb["action"]
        reward = mb["next", "reward"].reshape(-1)
        mask = 1.0 - mb["next", "terminated"].float().reshape(-1)
        next_obs = mb["next", self.obs_key]

        info = self._update_critic(obs, action, reward, mask, next_obs)
        self._soft_update_target()
        info.update(self._update_actor(obs))
        info.update(self._update_actor_optimistic(obs))
        info.update(self._update_duals(info["entropy"], info["kl"]))
        return info

    def _q_mix(self, q1, q2, coef):
        """mean(Q1, Q2) + coef * |Q1 - Q2| / 2 (coef < 0 is a lower bound)."""
        return (q1 + q2) / 2 + coef * (q1 - q2).abs() / 2

    def _update_critic(self, obs, action, reward, mask, next_obs) -> dict[str, torch.Tensor]:
        n = self.nets
        with torch.no_grad():
            alpha = n.log_alpha()
            mean, std = n.actor(next_obs)
            next_action, next_log_prob = tanh_normal_sample(mean, std)
            next_q = self._q_mix(*n.target_critic(next_obs, next_action), -self.pessimism)
            if self.distributional:  # [B, 1, N_target]
                target = reward[:, None, None] + self.discount * mask[:, None, None] * next_q[:, None, :]
                target = target - self.discount * alpha * mask[:, None, None] * next_log_prob[:, None, None]
            else:
                target = reward + self.discount * mask * (next_q - alpha * next_log_prob)

        q1, q2 = n.critic(obs, action)
        if self.distributional:
            loss = self._quantile_huber(target - q1[..., None]) + self._quantile_huber(target - q2[..., None])
        else:
            loss = ((q1 - target) ** 2 + (q2 - target) ** 2).mean()
        self.critic_optim.zero_grad(set_to_none=True)
        loss.backward()
        self.critic_optim.step()
        return {"critic_loss": loss.detach(), "q1": q1.detach().mean(), "q2": q2.detach().mean()}

    def _quantile_huber(self, td: torch.Tensor, kappa: float = 1.0) -> torch.Tensor:
        """Quantile-Huber loss; ``td`` is [B, N_pred, N_target], summed over predictions."""
        abs_td = td.abs()
        huber = torch.where(abs_td <= kappa, 0.5 * td**2, kappa * (abs_td - 0.5 * kappa))
        weight = (self.quantile_taus[None, :, None] - (td.detach() < 0).float()).abs()
        return (weight * huber / kappa).sum(dim=1).mean()

    @torch.no_grad()
    def _soft_update_target(self) -> None:
        torch._foreach_lerp_(
            list(self.nets.target_critic.parameters()), list(self.nets.critic.parameters()), self.tau
        )

    def _critic_frozen(self, obs, action):
        """Critic forward whose gradient flows to the action only."""
        critic = self.nets.critic
        critic.requires_grad_(False)
        try:
            return critic(obs, action)
        finally:
            critic.requires_grad_(True)

    def _update_actor(self, obs) -> dict[str, torch.Tensor]:
        n = self.nets
        mean, std = n.actor(obs)
        action, log_prob = tanh_normal_sample(mean, std)
        q = self._q_mix(*self._critic_frozen(obs, action), -self.pessimism)
        if self.distributional:
            q = q.mean(-1)
        loss = (log_prob * n.log_alpha().detach() - q).mean()
        self.actor_optim.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_optim.step()
        return {"actor_loss": loss.detach(), "entropy": -log_prob.detach().mean(), "std": std.detach().mean()}

    def _update_actor_optimistic(self, obs) -> dict[str, torch.Tensor]:
        n = self.nets
        with torch.no_grad():  # the freshly updated conservative actor
            mean_c, std_c = n.actor(obs)
        mean_o, std_o = n.actor_o(obs, mean_c, std_c, self.std_multiplier)
        action, _ = tanh_normal_sample(mean_o, std_o)
        q_ub = self._q_mix(*self._critic_frozen(obs, action), n.optimism().detach())
        if self.distributional:
            q_ub = q_ub.mean(-1)
        s = std_o / self.std_multiplier
        kl = (torch.log(std_c / s) + (s**2 + (mean_o - mean_c) ** 2) / (2 * std_c**2) - 0.5).sum(-1)
        loss = -q_ub.mean() + n.regularizer().detach() * kl.mean()
        self.actor_o_optim.zero_grad(set_to_none=True)
        loss.backward()
        self.actor_o_optim.step()
        return {"actor_o_loss": loss.detach(), "kl": kl.detach().mean()}

    def _update_duals(self, entropy, kl) -> dict[str, torch.Tensor]:
        """Temperature to target entropy; optimism and KL weight to target KL."""
        n = self.nets
        alpha = n.log_alpha()
        temp_loss = alpha * (entropy - self.target_entropy)
        empirical_kl = kl / self.action_dim
        optimism, kl_weight = n.optimism(), n.regularizer()
        adj_loss = (optimism - self.pessimism) * (empirical_kl - self.kl_target) - kl_weight * (
            empirical_kl - self.kl_target
        )
        self.temp_optim.zero_grad(set_to_none=True)
        self.adj_optim.zero_grad(set_to_none=True)
        (temp_loss + adj_loss).backward()
        self.temp_optim.step()
        self.adj_optim.step()
        return {"temperature": alpha.detach(), "optimism": optimism.detach(), "kl_weight": kl_weight.detach()}

    # ------------------------------------------------------------------
    # Policy access
    # ------------------------------------------------------------------

    def get_policy(self) -> TensorDictModule:
        return self._eval_policy

    def get_explore_policy(self) -> TensorDictModule:
        return self._explore_policy

    # ------------------------------------------------------------------
    # Checkpointing (the replay buffer is not checkpointed)
    # ------------------------------------------------------------------

    def _get_training_state(self) -> TrainingState:
        return TrainingState(
            step=0,
            policy_state_dict=self.nets.state_dict(),
            optimizer_state_dict={f"opt{i}": o.state_dict() for i, o in enumerate(self._optims)},
            extra={"collected_frames": self._collected_frames, "num_resets": self._num_resets},
        )

    def _load_training_state(self, state: TrainingState) -> None:
        self.nets.load_state_dict(state.policy_state_dict)
        for i, o in enumerate(self._optims):
            o.load_state_dict(state.optimizer_state_dict[f"opt{i}"])
        if state.extra:
            self._collected_frames = int(state.extra["collected_frames"])
            self._num_resets = int(state.extra["num_resets"])
