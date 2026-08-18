# Adapted from https://github.com/nicklashansen/tdmpc2 (tdmpc2/tdmpc2.py,
# tdmpc2/common/buffer.py, tdmpc2/trainer/online_trainer.py), MIT license.
# Changes: restructured as a BaseAlgorithm for this template (kwarg-only HPs,
# no `cfg`), single-task/state-obs only, device-agnostic, replay buffer stores
# standard `next`-keyed transitions, update loop split into <=50-LOC helpers.
"""TD-MPC2: Scalable, Robust World Models for Continuous Control.

Hansen et al. (2024), https://arxiv.org/abs/2310.16828

Pseudocode (online, single-task):
    Initialise world model {encoder h, dynamics d, reward R, policy prior p, Q-ensemble}
    For each step:
        Act by MPPI planning over latent rollouts of the model (warm-started)
        Store transition in slice replay buffer
        Sample B trajectory slices of length H
        Latent rollout: z_{t+1} = d(z_t, a_t); consistency loss ||z_{t+1} - h(s_{t+1})||
        Reward loss: soft-CE(R(z_t, a_t), r_t)          (discrete regression)
        Value loss:  soft-CE(Q(z_t, a_t), r_t + gamma * Q_target(z_{t+1}, p(z_{t+1})))
        Policy prior loss: -E[Q(z, p(z)) + alpha * entropy]  (on detached latents)
        Polyak-update target Q-ensemble
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers import SliceSampler
from torchrl.envs import EnvBase

from src.algorithms.base import BaseAlgorithm, CollectorConfig, TrainingState
from src.components import math
from src.components.scale import RunningScale

from .planner import MPPIPlanner
from .policy import make_policy
from .world_model import WorldModel, api_model_conversion


class TDMPC2Algorithm(BaseAlgorithm):
    """TD-MPC2 with a latent world model, MPPI planning and a Q-ensemble.

    Defaults reproduce the upstream single-task configuration at
    ``model_size=5`` (5M parameters), the setting used for all single-task
    benchmark results in the paper.

    The world model and replay buffer are built internally from scalar
    hyperparameters (no ``_partial_`` factories): the five subnetworks share
    ``latent_dim``/``simnorm_dim``/``num_bins`` coupling and compatibility with
    official upstream checkpoints pins the architecture and parameter names.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        *,
        obs_key: str = "observation",
        # --- Architecture (model_size=5 preset) ----------------------------
        enc_dim: int = 256,
        mlp_dim: int = 512,
        latent_dim: int = 512,
        num_enc_layers: int = 2,
        num_q: int = 5,
        dropout: float = 0.01,
        simnorm_dim: int = 8,
        # --- Discrete regression (two-hot) ---------------------------------
        num_bins: int = 101,
        vmin: float = -10.0,
        vmax: float = 10.0,
        # --- Policy prior ---------------------------------------------------
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        entropy_coef: float = 1e-4,
        # --- Planning (MPPI) -------------------------------------------------
        mpc: bool = True,
        horizon: int = 3,
        iterations: int = 6,
        num_samples: int = 512,
        num_elites: int = 64,
        num_pi_trajs: int = 24,
        min_std: float = 0.05,
        max_std: float = 2.0,
        temperature: float = 0.5,
        # --- Optimisation ----------------------------------------------------
        lr: float = 3e-4,
        enc_lr_scale: float = 0.3,
        grad_clip_norm: float = 20.0,
        tau: float = 0.01,
        rho: float = 0.5,
        batch_size: int = 256,
        consistency_coef: float = 20.0,
        reward_coef: float = 0.1,
        value_coef: float = 0.1,
        # --- Discount heuristic (episode length in decision steps) ----------
        episode_length: int = 500,
        discount_denom: float = 5,
        discount_min: float = 0.95,
        discount_max: float = 0.995,
        # --- Replay ----------------------------------------------------------
        buffer_size: int = 1_000_000,
        buffer_device: str = "cpu",  # "cuda" recovers upstream speed if it fits
        # --- Data collection / learning schedule ----------------------------
        frames_per_batch: int = 50,
        init_random_frames: int = 2_500,  # upstream seed_steps = max(1000, 5*ep_len)
        num_updates: int = 50,  # gradient updates per collector batch (utd=1.0)
        pretrain_updates: int | None = None,  # None -> init_random_frames (upstream)
        # --- Misc -------------------------------------------------------------
        compile: bool = False,  # torch.compile(mode="reduce-overhead"); CUDA only
    ) -> None:
        super().__init__(device)
        self.obs_key = obs_key
        self.enc_dim, self.mlp_dim, self.latent_dim = enc_dim, mlp_dim, latent_dim
        self.num_enc_layers, self.num_q = num_enc_layers, num_q
        self.dropout, self.simnorm_dim = dropout, simnorm_dim
        self.num_bins, self.vmin, self.vmax = num_bins, vmin, vmax
        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        self.entropy_coef = entropy_coef
        self.mpc, self.horizon, self.iterations = mpc, horizon, iterations
        self.num_samples, self.num_elites, self.num_pi_trajs = (
            num_samples, num_elites, num_pi_trajs,
        )
        self.min_std, self.max_std, self.temperature = min_std, max_std, temperature
        self.lr, self.enc_lr_scale = lr, enc_lr_scale
        self.grad_clip_norm, self.tau, self.rho = grad_clip_norm, tau, rho
        self.batch_size = batch_size
        self.consistency_coef, self.reward_coef, self.value_coef = (
            consistency_coef, reward_coef, value_coef,
        )
        self.episode_length, self.discount_denom = episode_length, discount_denom
        self.discount_min, self.discount_max = discount_min, discount_max
        self.buffer_size, self.buffer_device = buffer_size, buffer_device
        self.frames_per_batch = frames_per_batch
        self.init_random_frames = init_random_frames
        self.num_updates = num_updates
        self.pretrain_updates = pretrain_updates
        self.compile = compile
        self._collected_frames = 0
        self._pretrained = False

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, make_env: Callable[[], EnvBase]) -> None:
        proof_env = make_env()
        obs_dim = int(proof_env.observation_spec[self.obs_key].shape[-1])
        action_dim = int(proof_env.action_spec.shape[-1])
        del proof_env

        self.model = WorldModel(
            obs_dim, action_dim,
            latent_dim=self.latent_dim, mlp_dim=self.mlp_dim, enc_dim=self.enc_dim,
            num_enc_layers=self.num_enc_layers, num_q=self.num_q,
            dropout=self.dropout, simnorm_dim=self.simnorm_dim,
            num_bins=self.num_bins, vmin=self.vmin, vmax=self.vmax,
            log_std_min=self.log_std_min, log_std_max=self.log_std_max, tau=self.tau,
        ).to(self.device)
        self.model.eval()

        self._setup_optimizers()
        self.scale = RunningScale(self.tau, device=self.device)
        self.discount = self._get_discount(self.episode_length)
        self.planner = MPPIPlanner(
            self.model, action_dim=action_dim, discount=self.discount,
            horizon=self.horizon, iterations=self.iterations,
            num_samples=self.num_samples, num_elites=self.num_elites,
            num_pi_trajs=self.num_pi_trajs, min_std=self.min_std,
            max_std=self.max_std, temperature=self.temperature, device=self.device,
        ).to(self.device)
        self._explore_policy = make_policy(
            self.model, self.planner, self.obs_key, mpc=self.mpc, eval_mode=False
        )
        self._eval_policy = make_policy(
            self.model, self.planner, self.obs_key, mpc=self.mpc, eval_mode=True
        )
        self._setup_buffer()
        if self.compile:
            self._update = torch.compile(self._update, mode="reduce-overhead")
            self.planner.plan = torch.compile(self.planner.plan, mode="reduce-overhead")

    def _setup_optimizers(self) -> None:
        """Grouped Adam over the model (scaled encoder lr) + separate policy Adam."""
        capturable = self.device is not None and torch.device(self.device).type == "cuda"
        self.optim = torch.optim.Adam(
            [
                {"params": self.model._encoder.parameters(), "lr": self.lr * self.enc_lr_scale},
                {"params": self.model._dynamics.parameters()},
                {"params": self.model._reward.parameters()},
                {"params": self.model._Qs.parameters()},
            ],
            lr=self.lr,
            capturable=capturable,
        )
        self.pi_optim = torch.optim.Adam(
            self.model._pi.parameters(), lr=self.lr, eps=1e-5, capturable=capturable
        )

    def _setup_buffer(self) -> None:
        """Slice replay buffer: samples ``batch_size`` sub-trajectories of ``horizon``."""
        self._buffer_keys = (
            self.obs_key,
            "action",
            "episode",
            ("next", self.obs_key),
            ("next", "reward"),
            ("next", "terminated"),
        )
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(self.buffer_size, device=self.buffer_device),
            sampler=SliceSampler(
                num_slices=self.batch_size,
                traj_key="episode",
                end_key=None,
                truncated_key=None,
                strict_length=True,
            ),
            batch_size=self.batch_size * self.horizon,
        )

    def _get_discount(self, episode_length: int) -> float:
        """Discount heuristic scaling with episode length (upstream tdmpc2.py:58)."""
        frac = episode_length / self.discount_denom
        return min(max((frac - 1) / frac, self.discount_min), self.discount_max)

    def get_collector_config(self) -> CollectorConfig:
        return CollectorConfig(
            frames_per_batch=self.frames_per_batch,
            init_random_frames=self.init_random_frames,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def step(self, batch: TensorDict) -> dict[str, float]:
        """One collector iteration: store transitions, run gradient updates."""
        batch = batch.reshape(-1)
        batch = batch.set("episode", batch["collector", "traj_ids"])
        self.replay_buffer.extend(batch.select(*self._buffer_keys))
        self._collected_frames += batch.numel()

        # Warm-up: collect random transitions before any gradient update.
        if self._collected_frames < self.init_random_frames:
            return {}

        # One-off pretraining burst on seed data (upstream online_trainer.py:114).
        if not self._pretrained:
            n = self.pretrain_updates if self.pretrain_updates is not None else self.init_random_frames
            self._pretrained = True
        else:
            n = self.num_updates

        infos = []
        for _ in range(n):
            obs, action, reward, terminated = self._sample()
            if self.compile:
                torch.compiler.cudagraph_mark_step_begin()
            infos.append(self._update(obs, action, reward, terminated).clone())
        metrics = torch.stack(infos).mean().to_dict()
        return {f"train/{k}": float(v) for k, v in metrics.items()}

    def _sample(self):
        """Sample [horizon+1, B] obs / [horizon, B] action, reward, terminated."""
        td = self.replay_buffer.sample()
        td = td.view(self.batch_size, self.horizon).permute(1, 0).to(self.device)
        obs = torch.cat([td[self.obs_key], td["next", self.obs_key][-1:]], dim=0)
        action = td["action"]
        reward = td["next", "reward"]
        terminated = td["next", "terminated"].float()
        return obs.contiguous(), action.contiguous(), reward.contiguous(), terminated.contiguous()

    def _update(self, obs, action, reward, terminated) -> TensorDict:
        """One TD-MPC2 model + policy update (upstream tdmpc2.py:260)."""
        with torch.no_grad():
            next_z = self.model.encode(obs[1:])
            td_targets = self._td_target(next_z, reward, terminated)

        self.model.train()
        zs, consistency_loss = self._latent_rollout(obs[0], action, next_z)
        reward_loss, value_loss = self._reward_value_loss(zs[:-1], action, reward, td_targets)

        total_loss = (
            self.consistency_coef * consistency_loss
            + self.reward_coef * reward_loss
            + self.value_coef * value_loss
        )
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
        self.optim.step()
        self.optim.zero_grad(set_to_none=True)

        pi_info = self.update_pi(zs.detach())
        self.model.soft_update_target_Q()

        self.model.eval()
        info = TensorDict(
            {
                "consistency_loss": consistency_loss,
                "reward_loss": reward_loss,
                "value_loss": value_loss,
                "total_loss": total_loss,
                "grad_norm": grad_norm,
            }
        )
        info.update(pi_info)
        return info.detach().mean()

    @torch.no_grad()
    def _td_target(self, next_z, reward, terminated) -> torch.Tensor:
        """TD-target r + gamma * (1 - terminated) * min Q_target(z', pi(z'))."""
        action, _ = self.model.pi(next_z)
        return reward + self.discount * (1 - terminated) * self.model.Q(
            next_z, action, return_type="min", target=True
        )

    def _latent_rollout(self, obs0, action, next_z):
        """Roll the dynamics through the slice; rho-weighted consistency loss."""
        zs = torch.empty(
            self.horizon + 1, self.batch_size, self.latent_dim, device=self.device
        )
        z = self.model.encode(obs0)
        zs[0] = z
        consistency_loss = 0
        for t, (_action, _next_z) in enumerate(zip(action.unbind(0), next_z.unbind(0))):
            z = self.model.next(z, _action)
            consistency_loss = consistency_loss + F.mse_loss(z, _next_z) * self.rho**t
            zs[t + 1] = z
        return zs, consistency_loss / self.horizon

    def _reward_value_loss(self, _zs, action, reward, td_targets):
        """Rho-weighted soft cross-entropy losses for reward and value heads."""
        qs = self.model.Q(_zs, action, return_type="all")
        reward_preds = self.model.reward(_zs, action)

        reward_loss, value_loss = 0, 0
        for t, (rew_pred_t, rew_t, td_target_t, qs_t) in enumerate(
            zip(reward_preds.unbind(0), reward.unbind(0), td_targets.unbind(0), qs.unbind(1))
        ):
            reward_loss = reward_loss + (
                math.soft_ce(rew_pred_t, rew_t, self.num_bins, self.vmin, self.vmax).mean()
                * self.rho**t
            )
            for q_t in qs_t.unbind(0):
                value_loss = value_loss + (
                    math.soft_ce(q_t, td_target_t, self.num_bins, self.vmin, self.vmax).mean()
                    * self.rho**t
                )
        return reward_loss / self.horizon, value_loss / (self.horizon * self.num_q)

    def update_pi(self, zs: torch.Tensor) -> TensorDict:
        """Update the policy prior to maximise scaled Q-values + entropy."""
        action, info = self.model.pi(zs)
        qs = self.model.Q(zs, action, return_type="avg", detach=True)
        self.scale.update(qs[0])
        qs = self.scale(qs)

        # Loss is a rho-weighted sum of Q-values over the horizon.
        rho = torch.pow(self.rho, torch.arange(len(qs), device=self.device))
        pi_loss = (
            -(self.entropy_coef * info["scaled_entropy"] + qs).mean(dim=(1, 2)) * rho
        ).mean()
        pi_loss.backward()
        pi_grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model._pi.parameters(), self.grad_clip_norm
        )
        self.pi_optim.step()
        self.pi_optim.zero_grad(set_to_none=True)

        return TensorDict(
            {
                "pi_loss": pi_loss,
                "pi_grad_norm": pi_grad_norm,
                "pi_entropy": info["entropy"],
                "pi_scale": self.scale.value,
            }
        )

    # ------------------------------------------------------------------
    # Policy access
    # ------------------------------------------------------------------

    def get_policy(self) -> TensorDictModule:
        return self._eval_policy

    def get_explore_policy(self) -> TensorDictModule:
        return self._explore_policy

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def load_checkpoint(self, path: Path) -> int:
        """Load a template ``TrainingState`` or an official TD-MPC2 checkpoint.

        Official checkpoints (https://www.tdmpc2.com/models) are dicts of the
        form ``{"model": state_dict}`` holding world-model weights only.
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        if isinstance(state, TrainingState):
            self._load_training_state(state)
            return state.step
        state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
        state_dict = api_model_conversion(self.model.state_dict(), state_dict)
        self.model.load_state_dict(state_dict)
        return 0

    def _get_training_state(self) -> TrainingState:
        return TrainingState(
            step=0,
            policy_state_dict=self.model.state_dict(),
            optimizer_state_dict={
                "model": self.optim.state_dict(),
                "pi": self.pi_optim.state_dict(),
            },
            extra={
                "scale": self.scale.state_dict(),
                "prev_mean": self.planner._prev_mean.detach().cpu(),
                "collected_frames": self._collected_frames,
                "pretrained": self._pretrained,
            },
        )

    def _load_training_state(self, state: TrainingState) -> None:
        self.model.load_state_dict(state.policy_state_dict)
        self.optim.load_state_dict(state.optimizer_state_dict["model"])
        self.pi_optim.load_state_dict(state.optimizer_state_dict["pi"])
        if state.extra:
            self.scale.load_state_dict(state.extra["scale"])
            self.planner._prev_mean.copy_(state.extra["prev_mean"].to(self.planner._prev_mean.device))
            self._collected_frames = int(state.extra["collected_frames"])
            self._pretrained = bool(state.extra["pretrained"])
