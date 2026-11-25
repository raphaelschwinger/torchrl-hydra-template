# Adapted from https://github.com/ikostrikov/jaxrl (jaxrl/agents/sac/: learner,
# actor, critic, temperature), MIT license, as BRO's official code keeps it
# (github.com/naumix/BiggerRegularizedOptimistic, jaxrl/bro/ with BRO's extras
# switched off). Changes: restructured as a BaseAlgorithm for this template
# (kwarg-only HPs, no `cfg`), PyTorch instead of JAX, transitions in a torchrl
# replay buffer, and the update split into hooks that BRO overrides.
"""Soft Actor-Critic.

Haarnoja et al. (2018), https://arxiv.org/abs/1812.05905

Pseudocode (per agent step, after ``init_random_frames`` random steps):
    Act a ~ pi(s) = tanh N(mu(s), sigma(s));  store (s, a, r, s')
    Repeat ``num_updates`` times on a fresh uniform minibatch:
        Critic: (Q_i(s, a) - y)^2,  y = r + gamma (Q_lb'(s', a') - alpha log pi(a'|s')),  a' ~ pi(s')
        Polyak-update the target critics
        Actor:  minimise alpha log pi(a|s) - Q_lb(s, a),  a ~ pi(s)
        Temperature: alpha * (H(pi) - H_target)

``Q_lb = mean(Q1, Q2) - pessimism * |Q1 - Q2| / 2``: ``pessimism=1`` is the
usual clipped double-Q ``min(Q1, Q2)``; BRO lowers it to 0.

Extension points for subclasses (BRO): ``_build_networks``,
``_optimizer_specs``, ``_td_target`` / ``_critic_loss`` / ``_reduce_q``,
``_update``, ``_before_updates`` and ``_make_explore_policy``.
"""
from __future__ import annotations

import contextlib
import copy
import functools
from typing import Callable

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch.func import functional_call
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers import RandomSampler
from torchrl.envs import EnvBase

from src.algorithms.base import BaseAlgorithm, CollectorConfig, TrainingState

from .networks import MLP, DoubleCritic, LogParam, NormalTanhActor, tanh_normal_sample


class SACNetworks(nn.Module):
    """Actor, double critic, target critic and temperature, in one module.

    One module so that a full reset (BRO) is one ``load_state_dict`` and the
    checkpoint is one state dict.
    """

    def __init__(self, obs_dim: int, action_dim: int, algo: "SACAlgorithm") -> None:
        super().__init__()
        self.actor = NormalTanhActor(
            obs_dim, action_dim, trunk=algo.actor_trunk, hidden_dim=algo.actor_hidden_dim,
            depth=algo.actor_depth, log_std_min=algo.log_std_min, log_std_max=algo.log_std_max,
        )
        self.critic = DoubleCritic(
            obs_dim, action_dim, trunk=algo.critic_trunk, hidden_dim=algo.critic_hidden_dim,
            depth=algo.critic_depth, n_outputs=algo.critic_outputs,
        )
        self.target_critic = copy.deepcopy(self.critic).requires_grad_(False)
        self.log_alpha = LogParam(algo.init_temperature)


class SACPolicyModule(nn.Module):
    """``tanh(mean)`` for evaluation, a sample from the actor for exploration."""

    def __init__(self, nets: SACNetworks, *, eval_mode: bool) -> None:
        super().__init__()
        self.nets = nets
        self.eval_mode = eval_mode

    @torch.no_grad()
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        mean, std = self.nets.actor(obs)
        if self.eval_mode:
            return torch.tanh(mean)
        action, _ = tanh_normal_sample(mean, std)
        return action.clamp(-1.0, 1.0)


class SACAlgorithm(BaseAlgorithm):
    """SAC with a double critic, tanh-Gaussian actor and learned temperature.

    Defaults are jaxrl's SAC (the code BRO builds on): MLP 256x2 actor and
    critics with orthogonal init, Adam 3e-4, batch 256, tau 0.005, initial
    temperature 1, target entropy ``-|A|/2``, one update per agent step.
    """

    #: Trunk factory ``trunk(in_dim, hidden_dim, depth, out_dim=None)``; BRO uses BroNet.
    actor_trunk = staticmethod(MLP)
    critic_trunk = staticmethod(MLP)

    def __init__(
        self,
        device: torch.device | None = None,
        *,
        obs_key: str = "observation",
        # --- Architecture ----------------------------------------------------
        critic_hidden_dim: int = 256,
        critic_depth: int = 2,
        actor_hidden_dim: int = 256,
        actor_depth: int = 2,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        # --- Optimisation ----------------------------------------------------
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        weight_decay: float = 0.0,  # > 0 switches the networks to AdamW
        discount: float = 0.99,
        tau: float = 0.005,
        batch_size: int = 256,
        pessimism: float = 1.0,  # 1 = clipped double-Q min(Q1, Q2)
        # --- Entropy ---------------------------------------------------------
        init_temperature: float = 1.0,
        target_entropy: float | None = None,  # None -> -action_dim / 2
        # --- Replay ----------------------------------------------------------
        buffer_size: int = 1_000_000,
        buffer_device: str = "cpu",
        # --- Data collection / learning schedule ----------------------------
        frames_per_batch: int = 1,
        init_random_frames: int = 5_000,
        num_updates: int = 1,  # replay ratio: gradient steps per agent step
        # --- Runtime knobs (none changes the update equations; amp changes numerics)
        compile: bool | str = False,  # torch.compile: True = default mode, or a mode string
        compile_scope: str = "update",  # "update" = one gradient step, "step" = all num_updates of an agent step
        amp: bool = False,  # autocast(bfloat16) around forward passes
        tf32: bool = False,  # TF32 for float32 matmuls
        fused_optim: bool = False,  # fused optimiser kernels
    ) -> None:
        super().__init__(device)
        if compile_scope not in ("update", "step"):
            raise ValueError(f"compile_scope must be 'update' or 'step', got {compile_scope!r}")
        self.obs_key = obs_key
        self.critic_hidden_dim, self.critic_depth = critic_hidden_dim, critic_depth
        self.actor_hidden_dim, self.actor_depth = actor_hidden_dim, actor_depth
        self.log_std_min, self.log_std_max = log_std_min, log_std_max
        self.actor_lr, self.critic_lr, self.temp_lr = actor_lr, critic_lr, temp_lr
        self.weight_decay = weight_decay
        self.discount, self.tau, self.batch_size = discount, tau, batch_size
        self.pessimism = pessimism
        self.init_temperature, self.target_entropy = init_temperature, target_entropy
        self.buffer_size, self.buffer_device = buffer_size, buffer_device
        self.frames_per_batch = frames_per_batch
        self.init_random_frames = init_random_frames
        self.num_updates = num_updates
        self.compile, self.compile_scope = compile, compile_scope
        self.amp, self.tf32, self.fused_optim = amp, tf32, fused_optim
        self._collected_frames = 0
        self._last_info: dict[str, torch.Tensor] | None = None

    #: Critic output width: 1 for SAC, the number of quantiles for BRO.
    @property
    def critic_outputs(self) -> int:
        return 1

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @property
    def _on_cuda(self) -> bool:
        return self.device is not None and torch.device(self.device).type == "cuda"

    def setup(self, make_env: Callable[[], EnvBase]) -> None:
        proof_env = make_env()
        obs_dim = int(proof_env.observation_spec[self.obs_key].shape[-1])
        self.action_dim = int(proof_env.action_spec.shape[-1])
        del proof_env
        if self.target_entropy is None:
            self.target_entropy = -self.action_dim / 2
        if self.tf32 and self._on_cuda:
            torch.set_float32_matmul_precision("high")

        self.nets = self._build_networks(obs_dim, self.action_dim).to(self.device)
        self._init_state = copy.deepcopy(self.nets.state_dict())  # for resets
        self._setup_optimizers()

        self._explore_policy = TensorDictModule(
            self._make_explore_policy(), in_keys=[self.obs_key], out_keys=["action"]
        )
        self._eval_policy = TensorDictModule(
            SACPolicyModule(self.nets, eval_mode=True), in_keys=[self.obs_key], out_keys=["action"]
        )
        self._buffer_keys = (self.obs_key, "action", ("next", self.obs_key), ("next", "reward"), ("next", "terminated"))
        self.replay_buffer = ReplayBuffer(
            storage=LazyTensorStorage(self.buffer_size, device=self.buffer_device),
            sampler=RandomSampler(),
            batch_size=self.batch_size * self.num_updates,
        )
        self._updates_fn = self._updates
        if self.compile:
            mode = None if self.compile is True else str(self.compile)
            if self.compile_scope == "step":
                self._updates_fn = torch.compile(self._updates, mode=mode)
            else:
                self._update = torch.compile(self._update, mode=mode)

    def _build_networks(self, obs_dim: int, action_dim: int) -> nn.Module:
        return SACNetworks(obs_dim, action_dim, self)

    def _make_explore_policy(self) -> nn.Module:
        return SACPolicyModule(self.nets, eval_mode=False)

    def _optimizer_specs(self) -> dict[str, tuple[list[nn.Parameter], dict]]:
        """name -> (params, optimiser kwargs). ``wd`` > 0 selects AdamW."""
        n = self.nets
        return {
            "actor": (list(n.actor.parameters()), {"lr": self.actor_lr, "wd": self.weight_decay}),
            "critic": (list(n.critic.parameters()), {"lr": self.critic_lr, "wd": self.weight_decay}),
            "temp": (list(n.log_alpha.parameters()), {"lr": self.temp_lr, "betas": (0.5, 0.999)}),
        }

    def _setup_optimizers(self) -> None:
        # CUDA-graph capture needs device-side step counters (`capturable`).
        extra = {"capturable": bool(self.compile) and self._on_cuda}
        if self.fused_optim and self._on_cuda:
            extra["fused"] = True
        self.optims: dict[str, torch.optim.Optimizer] = {}
        for name, (params, kw) in self._optimizer_specs().items():
            kw = dict(kw)
            wd = kw.pop("wd", 0.0)
            opt_cls = functools.partial(torch.optim.AdamW, weight_decay=wd) if wd > 0 else torch.optim.Adam
            self.optims[name] = opt_cls(params, eps=1e-8, **kw, **extra)

    def get_collector_config(self) -> CollectorConfig:
        return CollectorConfig(
            frames_per_batch=self.frames_per_batch,
            init_random_frames=self.init_random_frames,
        )

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def step(self, batch: TensorDict) -> dict[str, float]:
        """Store the batch; after warm-up, ``num_updates`` updates per agent step."""
        batch = batch.reshape(-1)
        self.replay_buffer.extend(batch.select(*self._buffer_keys))
        prev, self._collected_frames = self._collected_frames, self._collected_frames + batch.numel()
        if self._collected_frames < self.init_random_frames:
            return {}
        self._before_updates(prev, self._collected_frames)
        for _ in range(batch.numel()):
            sample = self._sample()
            if self.compile and self.compile_scope == "step":
                torch.compiler.cudagraph_mark_step_begin()
            info = self._updates_fn(sample)
            self._last_info = {k: v.clone() for k, v in info.items()}
        # Diagnostics stay on the device until a logging boundary.
        return {}

    def _before_updates(self, prev_frames: int, frames: int) -> None:
        """Hook between storing a batch and the updates (BRO resets here)."""

    def _sample(self) -> TensorDict:
        """One uniform draw of ``num_updates * batch_size`` transitions, on the device."""
        return self.replay_buffer.sample().to(self.device, non_blocking=True)

    def _updates(self, sample: TensorDict) -> dict[str, torch.Tensor]:
        """``num_updates`` gradient steps, one per ``batch_size`` slice of ``sample``."""
        per_update = self.compile and self.compile_scope == "update"
        info: dict[str, torch.Tensor] = {}
        for mb in sample.split(self.batch_size):
            if per_update:
                torch.compiler.cudagraph_mark_step_begin()
            info = self._update(mb)
            if per_update:
                info = {k: v.clone() for k, v in info.items()}
        return info

    def pop_train_metrics(self) -> dict[str, float]:
        """The last update's diagnostics, read only at logging boundaries (one host sync)."""
        info, self._last_info = self._last_info, None
        return {} if info is None else {f"train/{k}": float(v) for k, v in info.items()}

    # ------------------------------------------------------------------
    # The SAC update
    # ------------------------------------------------------------------

    def _update(self, mb: TensorDict) -> dict[str, torch.Tensor]:
        """Critic, target critic, actor, temperature."""
        obs, action = mb[self.obs_key], mb["action"]
        reward = mb["next", "reward"].reshape(-1)
        mask = 1.0 - mb["next", "terminated"].float().reshape(-1)
        info = self._update_critic(obs, action, reward, mask, mb["next", self.obs_key])
        self._soft_update_target()
        info.update(self._update_actor(obs))
        info.update(self._update_temperature(info["entropy"]))
        return info

    def _update_critic(self, obs, action, reward, mask, next_obs) -> dict[str, torch.Tensor]:
        n = self.nets
        with torch.no_grad(), self._autocast():
            mean, std = n.actor(next_obs)
            next_action, next_log_prob = tanh_normal_sample(mean.float(), std.float())
            nq1, nq2 = n.target_critic(next_obs, next_action)
            next_q = self._q_mix(nq1.float(), nq2.float(), -self.pessimism)
            target = self._td_target(next_q, reward, mask, n.log_alpha(), next_log_prob)
        with self._autocast():
            q1, q2 = n.critic(obs, action)
        loss = self._critic_loss(q1.float(), q2.float(), target)
        self._step_optim("critic", loss)
        return {"critic_loss": loss.detach(), "q1": q1.detach().float().mean(), "q2": q2.detach().float().mean()}

    def _td_target(self, next_q, reward, mask, alpha, next_log_prob) -> torch.Tensor:
        """y = r + gamma * mask * (Q'(s', a') - alpha log pi(a'|s'))."""
        return reward + self.discount * mask * (next_q - alpha * next_log_prob)

    def _critic_loss(self, q1, q2, target) -> torch.Tensor:
        return ((q1 - target) ** 2 + (q2 - target) ** 2).mean()

    def _update_actor(self, obs) -> dict[str, torch.Tensor]:
        n = self.nets
        with self._autocast():
            mean, std = n.actor(obs)
        action, log_prob = tanh_normal_sample(mean.float(), std.float())
        q = self._q_value(obs, action, -self.pessimism)
        loss = (log_prob * n.log_alpha().detach() - q).mean()
        self._step_optim("actor", loss)
        return {"actor_loss": loss.detach(), "entropy": -log_prob.detach().mean(), "std": std.detach().float().mean()}

    def _update_temperature(self, entropy) -> dict[str, torch.Tensor]:
        alpha = self.nets.log_alpha()
        self._step_optim("temp", alpha * (entropy - self.target_entropy))
        return {"temperature": alpha.detach()}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _q_mix(self, q1, q2, coef):
        """mean(Q1, Q2) + coef * |Q1 - Q2| / 2; ``coef = -1`` is ``min(Q1, Q2)``."""
        return (q1 + q2) / 2 + coef * (q1 - q2).abs() / 2

    def _reduce_q(self, q: torch.Tensor) -> torch.Tensor:
        """Per-sample scalar from the critic output (identity; BRO averages quantiles)."""
        return q

    def _q_value(self, obs, action, coef) -> torch.Tensor:
        """Mixed Q of ``action`` with the critic weights detached (gradient to the action only)."""
        critic = self.nets.critic
        params = {k: v.detach() for k, v in critic.named_parameters()}
        with self._autocast():
            q1, q2 = functional_call(critic, params, (obs, action))
        return self._reduce_q(self._q_mix(q1.float(), q2.float(), coef))

    def _step_optim(self, name: str, loss: torch.Tensor) -> None:
        opt = self.optims[name]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    @torch.no_grad()
    def _soft_update_target(self) -> None:
        torch._foreach_lerp_(
            list(self.nets.target_critic.parameters()), list(self.nets.critic.parameters()), self.tau
        )

    def _autocast(self):
        """bf16 autocast for forward passes, or a no-op."""
        if self.amp and self._on_cuda:
            return torch.autocast("cuda", dtype=torch.bfloat16)
        return contextlib.nullcontext()

    # ------------------------------------------------------------------
    # Policy access and checkpointing (the replay buffer is not checkpointed)
    # ------------------------------------------------------------------

    def get_policy(self) -> TensorDictModule:
        return self._eval_policy

    def get_explore_policy(self) -> TensorDictModule:
        return self._explore_policy

    def _get_training_state(self) -> TrainingState:
        return TrainingState(
            step=0,
            policy_state_dict=self.nets.state_dict(),
            optimizer_state_dict={k: o.state_dict() for k, o in self.optims.items()},
            extra=self._extra_state(),
        )

    def _load_training_state(self, state: TrainingState) -> None:
        self.nets.load_state_dict(state.policy_state_dict)
        for k, o in self.optims.items():
            o.load_state_dict(state.optimizer_state_dict[k])
        self._load_extra_state(state.extra or {})

    def _extra_state(self) -> dict:
        return {"collected_frames": self._collected_frames}

    def _load_extra_state(self, extra: dict) -> None:
        self._collected_frames = int(extra.get("collected_frames", self._collected_frames))
