"""BBF — Bigger, Better, Faster (Schwarzer et al., ICML 2023).

"Bigger, Better, Faster: Human-level Atari with human-level efficiency"
https://arxiv.org/abs/2305.19452

Model-free state of the art on Atari 100k. A Rainbow-style distributional
Q-learner pushed to a high replay ratio, made stable by six interacting design
choices:

    1. Bigger network        Impala-CNN ResNet encoder at 4x width.
    2. Self-prediction (SPR) auxiliary loss: predict your own future latents
                             k = 1..5 steps ahead through a learned
                             transition model (Schwarzer et al. 2021).
    3. Periodic resets       every ``reset_interval`` gradient steps the heads
                             are re-initialised and encoder + transition model
                             are interpolated 50% towards random weights
                             (shrink-and-perturb; applied to the online and
                             EMA target networks alike), fighting overfitting
                             and plasticity loss at high replay ratios.
    4. Annealed update       after every reset, the n-step horizon decays
                             exponentially 10 -> 3 and the discount rises
                             0.97 -> 0.997 over ``cycle_steps`` gradient steps.
    5. Regularisation        weight decay (AdamW), DrQ-style augmentation
                             (random shift + intensity) on every replayed
                             frame, EMA target network (tau = 0.005).
    6. Rainbow backbone      C51 distributional RL, Double DQN, dueling,
                             prioritized replay. NoisyNets are removed.

Design decisions specific to this template (verified against the official JAX
release, ``configs/BBF.gin``):

  - **Hand-written C51.** The categorical projection (``_project_distribution``)
    and n-step target are computed directly so a single EMA target network
    feeds both the value loss and SPR (rather than ``DistributionalDQNLoss``,
    which manages its own delayed target internally).
  - **torchrl-native buffer.** The prioritized *subsequence* buffer is a plain
    ``TensorDictReplayBuffer`` with a ``PrioritizedSliceSampler``: it samples
    contiguous windows ``t .. t+window`` so the annealed n-step (which must not
    bake ``n`` into storage) works, while reusing torchrl's storage + sum-tree.
    n-step returns and episode-boundary masking are computed at *sample* time.
    Deviation from Dopamine: torchrl reduces a slice's sampling priority over
    the window (``reduction='max'``); we key priorities on the window's start
    transition (priority = C51 loss, sampled ``prop.`` to ``loss**alpha``).
  - **Life loss.** This template's ``EpisodicLifeEnv`` flags life loss as
    ``terminated`` at the gym level, so ``cut = terminated | done`` marks every
    bootstrap boundary; there is no separate ``end-of-life`` key.

Hyperparameter defaults follow ``configs/BBF.gin`` at ``replay_ratio=2``; the
paper's flagship setting is ``replay_ratio=8``. Note the gin's
``reset_every = 20_000`` counts *environment* steps (checked once per
``training_steps`` in the official ``_train_step``), i.e. resets happen every
40k gradient steps at any replay ratio — ``reset_interval`` below is in
gradient steps, so it stays 40_000 for both RR2 and RR8.
"""
from __future__ import annotations

import math
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers import PrioritizedSliceSampler, SliceSampler
from torchrl.envs import EnvBase
from torchrl.modules import EGreedyModule, QValueActor

from src.algorithms.base import BaseAlgorithm, CollectorConfig, TrainingState
from src.algorithms.bbf.networks import BBFNetwork


class BBFAlgorithm(BaseAlgorithm):
    """BBF: distributional Q-learning at replay ratio 2-8 with SPR, resets,
    annealed n-step/discount, augmentation and a scaled Impala encoder."""

    def __init__(
        self,
        device: torch.device | None = None,
        *,
        obs_key: str = "pixels",
        # --- Network (the "Bigger") ----------------------------------------
        width_scale: int = 4,
        hidden_dim: int = 2048,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        # --- Optimisation ----------------------------------------------------
        lr: float = 1e-4,
        weight_decay: float = 0.1,
        adam_eps: float = 1.5e-4,
        batch_size: int = 32,
        max_grad_norm: float | None = None,   # official BBF does not clip
        # --- Update rule -----------------------------------------------------
        replay_ratio: float = 2,              # gradient steps per env step (paper: 8)
        max_update_horizon: int = 10,
        min_update_horizon: int = 3,
        min_gamma: float = 0.97,
        max_gamma: float = 0.997,
        cycle_steps: int = 10_000,            # anneal window after each reset (grad steps)
        double_dqn: bool = True,
        dueling: bool = True,
        # --- SPR auxiliary loss ----------------------------------------------
        spr_weight: float = 5.0,              # 0 disables SPR
        spr_depth: int = 5,                   # prediction horizon k
        # --- Resets (shrink-and-perturb) --------------------------------------
        reset_interval: int = 40_000,         # grad steps (paper: 40k at any RR); 0 disables
        shrink_factor: float = 0.5,           # keep 50% of encoder/transition weights
        perturb_factor: float = 0.5,
        no_resets_after: int = 0,             # grad steps; 0 = never stop resetting
        # --- Target network ----------------------------------------------------
        target_tau: float = 0.005,            # EMA per gradient step
        # --- Replay -------------------------------------------------------------
        replay_capacity: int = 105_000,       # >= total env steps: ring never wraps
        prioritized: bool = True,
        prb_alpha: float = 0.5,               # sampling prop. to loss**alpha
        prb_beta: float = 0.5,                # importance-sampling exponent
        # --- Augmentation --------------------------------------------------------
        data_augmentation: bool = True,
        aug_pad: int = 4,
        intensity_scale: float = 0.05,
        # --- Exploration / collection ---------------------------------------------
        eps_start: float = 1.0,
        eps_end: float = 0.0,
        eps_annealing_frames: int = 2_001,
        eps_eval: float = 0.001,
        min_replay_history: int = 2_000,      # env steps before learning starts
        frames_per_batch: int = 1,
        max_frames_per_traj: int = -1,
        renormalize_latent: bool = True,
    ) -> None:
        super().__init__(device)
        self.obs_key = obs_key
        self.width_scale = width_scale
        self.hidden_dim = hidden_dim
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.lr = lr
        self.weight_decay = weight_decay
        self.adam_eps = adam_eps
        self.batch_size = batch_size
        self.max_grad_norm = max_grad_norm
        self.replay_ratio = replay_ratio
        self.max_update_horizon = max_update_horizon
        self.min_update_horizon = min_update_horizon
        self.min_gamma = min_gamma
        self.max_gamma = max_gamma
        self.cycle_steps = cycle_steps
        self.double_dqn = double_dqn
        self.dueling = dueling
        self.spr_weight = spr_weight
        self.spr_depth = spr_depth
        self.reset_interval = reset_interval
        self.shrink_factor = shrink_factor
        self.perturb_factor = perturb_factor
        self.no_resets_after = no_resets_after
        self.target_tau = target_tau
        self.replay_capacity = replay_capacity
        self.prioritized = prioritized
        self.prb_alpha = prb_alpha
        self.prb_beta = prb_beta
        self.data_augmentation = data_augmentation
        self.aug_pad = aug_pad
        self.intensity_scale = intensity_scale
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_annealing_frames = eps_annealing_frames
        self.eps_eval = eps_eval
        self.min_replay_history = min_replay_history
        self.frames_per_batch = frames_per_batch
        self.max_frames_per_traj = max_frames_per_traj
        self.renormalize_latent = renormalize_latent

        # window sampled from the buffer: enough to cover the largest n-step
        # horizon *and* the SPR rollout. slice_len = window + 1 frames.
        self.window = max(self.max_update_horizon, self.spr_depth)

        self._collected_frames = 0
        self._grad_steps = 0
        self._steps_since_reset = 0
        self._num_resets = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, make_env: Callable[[], EnvBase]) -> None:
        proof_env = make_env()
        obs_shape = tuple(proof_env.observation_spec[self.obs_key].shape)
        action_spec = proof_env.action_spec
        num_actions = int(action_spec.space.n)
        proof_env.close()
        self._action_spec = action_spec
        self._obs_shape = obs_shape

        def make_network() -> BBFNetwork:
            return BBFNetwork(
                obs_shape,
                num_actions,
                width_scale=self.width_scale,
                hidden_dim=self.hidden_dim,
                num_atoms=self.num_atoms,
                v_min=self.v_min,
                v_max=self.v_max,
                dueling=self.dueling,
                renorm=self.renormalize_latent,
            )

        self._make_network = make_network
        self.network = make_network().to(self.device)
        self.target_network = make_network().to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_network.requires_grad_(False)
        self.support = self.network.support.to(self.device)

        # Policies: greedy actor for eval, eps-greedy for collection. Both act
        # with the EMA *target* network, as in the official release
        # (``BBF.gin: target_action_selection = True``, applied in train and
        # eval mode alike): the target changes smoothly under policy churn and
        # recovers via EMA after each shrink-and-perturb reset, whereas the
        # online net's heads are freshly random. The EMA update mutates the
        # target's parameters in place, so this actor always sees them current.
        self.q_actor = QValueActor(
            module=self.target_network, spec=action_spec, in_keys=[self.obs_key]
        ).to(self.device)
        self.greedy_module = EGreedyModule(
            spec=action_spec,
            eps_init=self.eps_start,
            eps_end=self.eps_end,
            annealing_num_steps=self.eps_annealing_frames,
            device=self.device,
        )
        self._explore_policy = TensorDictSequential(self.q_actor, self.greedy_module)
        self._eval_policy = TensorDictSequential(
            self.q_actor, FixedEpsilonGreedy(action_spec, self.eps_eval)
        )

        self.replay_buffer = self._make_replay_buffer(self.window)
        self.optimizer = self._make_optimizer()

    def _make_replay_buffer(self, window: int) -> TensorDictReplayBuffer:
        """Prioritized (or uniform) contiguous-window buffer.

        A constant ``traj`` key makes the whole buffer a single trajectory so
        windows may span episode/life boundaries (masked later via ``cut``);
        ``replay_capacity >= total env steps`` guarantees the ring never wraps,
        so a window never straddles the write head.
        """
        slice_len = window + 1
        storage = LazyTensorStorage(self.replay_capacity, device="cpu")
        if self.prioritized:
            sampler = PrioritizedSliceSampler(
                max_capacity=self.replay_capacity,
                alpha=self.prb_alpha,
                beta=self.prb_beta,
                slice_len=slice_len,
                traj_key="traj",
                strict_length=True,
            )
        else:
            sampler = SliceSampler(slice_len=slice_len, traj_key="traj", strict_length=True)
        return TensorDictReplayBuffer(storage=storage, sampler=sampler, batch_size=None)

    def _make_optimizer(self) -> torch.optim.AdamW:
        decay, no_decay = [], []
        for p in self.network.parameters():
            (decay if p.ndim > 1 else no_decay).append(p)
        return torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": self.weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ],
            lr=self.lr,
            eps=self.adam_eps,
        )

    def get_collector_config(self) -> CollectorConfig:
        return CollectorConfig(
            frames_per_batch=self.frames_per_batch,
            init_random_frames=self.min_replay_history,
            max_frames_per_traj=self.max_frames_per_traj,
        )

    # ------------------------------------------------------------------
    # Schedules (exponential, restart after each reset) — paper section 4
    # ------------------------------------------------------------------

    @staticmethod
    def _exp_interp(start: float, end: float, t: int, period: int) -> float:
        frac = min(max(t / max(period, 1), 0.0), 1.0)
        return math.exp(math.log(start) + frac * (math.log(end) - math.log(start)))

    def _current_horizon(self) -> int:
        return int(
            round(
                self._exp_interp(
                    self.max_update_horizon,
                    self.min_update_horizon,
                    self._steps_since_reset,
                    self.cycle_steps,
                )
            )
        )

    def _current_gamma(self) -> float:
        # interpolate exponentially in (1 - gamma) space: 0.03 -> 0.003
        return 1.0 - self._exp_interp(
            1.0 - self.min_gamma,
            1.0 - self.max_gamma,
            self._steps_since_reset,
            self.cycle_steps,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def step(self, batch: TensorDict) -> dict[str, float]:
        """One collector iteration: store transitions, then ``replay_ratio``
        gradient updates per collected env step."""
        if batch.batch_dims != 1:
            raise ValueError(
                "BBF requires a temporally contiguous stream: set trainer.num_envs=1"
            )
        num_frames = batch.numel()
        self._store(batch)
        self._collected_frames += num_frames
        # Official ``linearly_decaying_epsilon`` holds eps = 1 through the
        # random warm-up and only then decays it over ``eps_annealing_frames``
        # (2001) env steps. Annealing during the warm-up (whose actions the
        # collector randomises anyway) would make the policy greedy the moment
        # learning starts, skipping ~2k frames of exploration.
        if self._collected_frames > self.min_replay_history:
            self.greedy_module.step(num_frames)

        metrics = {
            "train/epsilon": float(self.greedy_module.eps),
            "train/n_step": float(self._current_horizon()),
            "train/gamma": self._current_gamma(),
            "train/num_resets": float(self._num_resets),
        }
        if self._collected_frames < self.min_replay_history:
            return metrics

        num_updates = max(1, round(self.replay_ratio * num_frames))
        rl_losses = torch.zeros(num_updates)
        spr_losses = torch.zeros(num_updates)
        for u in range(num_updates):
            # Strict ``>`` mirrors the official ``training_steps > next_reset``
            # (with ``reset_offset=1``): the k-th reset fires at grad step
            # k*(reset_interval+1), so the reset scheduled exactly at the last
            # gradient step of a run never fires. With ``>=`` a 100k-frame RR2
            # run would shrink-and-perturb on its final update and hand a
            # freshly reset policy to the last checkpoint / final eval.
            if (
                self.reset_interval > 0
                and self._steps_since_reset > self.reset_interval
                and (self.no_resets_after == 0 or self._grad_steps < self.no_resets_after)
            ):
                self._shrink_and_perturb()
            rl_loss, spr_loss = self._update()
            rl_losses[u] = rl_loss
            spr_losses[u] = spr_loss
            self._grad_steps += 1
            self._steps_since_reset += 1

        metrics.update(
            {
                "train/q_loss": rl_losses.mean().item(),
                "train/spr_loss": spr_losses.mean().item(),
                "train/grad_steps": float(self._grad_steps),
            }
        )
        return metrics

    def _store(self, batch: TensorDict) -> None:
        """Append a collected batch to the replay buffer as flat per-step
        transitions (uint8 frame stacks + integer action + reward + cut)."""
        flat = batch.reshape(-1)
        n = flat.numel()
        pixels = flat[self.obs_key]
        obs_u8 = (pixels * 255.0).round_().clamp_(0, 255).to(torch.uint8).cpu()
        action = flat["action"]
        if action.dim() > 1:  # one-hot action encoding -> integer index
            action = action.argmax(-1)
        action = action.reshape(n).long().cpu()
        reward = flat[("next", "reward")].reshape(n).float().cpu()
        done = flat[("next", "done")].reshape(n).bool()
        terminated = flat.get(("next", "terminated"), default=done).reshape(n).bool()
        cut = (terminated | done).cpu()
        transitions = TensorDict(
            {
                "pixels": obs_u8,
                "action": action,
                "reward": reward,
                "cut": cut,
                "traj": torch.zeros(n, dtype=torch.long),
            },
            batch_size=[n],
        )
        self.replay_buffer.extend(transitions)

    def _sample(self) -> dict[str, torch.Tensor]:
        """Sample ``batch_size`` contiguous windows and unpack to (B, ...)."""
        sl = self.window + 1
        sample = self.replay_buffer.sample(self.batch_size * sl).reshape(self.batch_size, sl)
        sample = sample.to(self.device)
        if self.prioritized:
            w = sample.get("priority_weight")[:, 0].float()
            weights = w / w.max().clamp_min(1e-8)
            # Storage indices live on CPU; keep them there for update_priority.
            start_idx = sample.get("index")[:, 0].reshape(-1).cpu()
        else:
            weights = torch.ones(self.batch_size, device=self.device)
            start_idx = None
        return {
            "obs": sample.get("pixels").float() / 255.0,   # (B, window+1, C, H, W)
            "action": sample.get("action"),                # (B, window+1)
            "reward": sample.get("reward"),                # (B, window+1)
            "cut": sample.get("cut"),                       # (B, window+1)
            "weights": weights,                             # (B,)
            "start_idx": start_idx,                         # (B,) storage indices or None
        }

    def _update(self) -> tuple[float, float]:
        """One gradient step; returns (C51 loss, SPR loss) for logging."""
        n = self._current_horizon()
        gamma = self._current_gamma()
        sample = self._sample()
        b = sample["action"].shape[0]
        arange = torch.arange(b, device=self.device)

        obs = sample["obs"]                                # (B, window+1, C, H, W)
        obs_t = self._augment(obs[:, 0])
        returns, bootstrap, alive = _masked_nstep_return(
            sample["reward"], sample["cut"], gamma, n
        )

        # --- C51 target: project n-step Bellman backup onto the support ----
        with torch.no_grad():
            obs_tn = self._augment(obs[:, n])
            target_probs = F.softmax(
                self.target_network.q_logits(self.target_network.encode(obs_tn)), -1
            )                                                          # (B, A, atoms)
            if self.double_dqn:
                next_q = self.network(obs_tn)                          # online net
            else:
                next_q = (target_probs * self.support).sum(-1)
            a_star = next_q.argmax(-1)
            next_dist = target_probs[arange, a_star]                   # (B, atoms)
            target_dist = _project_distribution(
                next_dist, returns, bootstrap, self.support, gamma**n
            )

        # --- Online distribution at (s_t, a_t) ------------------------------
        latent_t = self.network.encode(obs_t)
        logits_t = self.network.q_logits(latent_t)                     # (B, A, atoms)
        log_p = F.log_softmax(logits_t[arange, sample["action"][:, 0]], -1)
        rl_loss_elem = -(target_dist * log_p).sum(-1)                  # (B,)

        # --- SPR: predict own future latents through the transition model ---
        if self.spr_weight > 0:
            k = self.spr_depth
            with torch.no_grad():
                future = obs[:, 1 : k + 1].reshape(-1, *obs.shape[2:])
                future = self._augment(future)
                spr_targets = self.target_network.project(
                    self.target_network.encode(future)
                ).view(b, k, -1)
                spr_targets = F.normalize(spr_targets, dim=-1)
            z_hat = latent_t
            predictions = []
            for j in range(k):
                z_hat = self.network.transition_model(z_hat, sample["action"][:, j])
                predictions.append(self.network.predict(self.network.project(z_hat)))
            spr_pred = F.normalize(torch.stack(predictions, 1), dim=-1)
            per_jump = ((spr_pred - spr_targets) ** 2).sum(-1)         # 2 - 2 cos
            spr_loss_elem = (per_jump * alive[:, :k]).mean(dim=1)      # (B,)
        else:
            spr_loss_elem = torch.zeros_like(rl_loss_elem)

        loss = (
            sample["weights"] * (rl_loss_elem + self.spr_weight * spr_loss_elem)
        ).mean()

        self.optimizer.zero_grad()
        loss.backward()
        if self.max_grad_norm is not None:
            nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
        self.optimizer.step()

        if self.prioritized:
            # Key the window's priority on its start transition (C51 loss);
            # sampling is prop. to loss**alpha via the sum-tree.
            self.replay_buffer.update_priority(
                sample["start_idx"], rl_loss_elem.detach().cpu() + 1e-10
            )
        self._ema_update_target()
        return (
            float(rl_loss_elem.detach().mean()),
            float(spr_loss_elem.detach().mean()),
        )

    # ------------------------------------------------------------------
    # BBF machinery: augmentation, EMA target, shrink-and-perturb
    # ------------------------------------------------------------------

    def _augment(self, x: torch.Tensor) -> torch.Tensor:
        """DrQ-style random shift (pad+crop) plus intensity jitter."""
        if not self.data_augmentation:
            return x
        x = _random_shift(x, self.aug_pad)
        noise = 1.0 + self.intensity_scale * torch.randn(
            x.shape[0], 1, 1, 1, device=x.device
        ).clamp_(-2.0, 2.0)
        return x * noise

    @torch.no_grad()
    def _ema_update_target(self) -> None:
        for tp, p in zip(self.target_network.parameters(), self.network.parameters()):
            tp.lerp_(p, self.target_tau)

    @torch.no_grad()
    def _shrink_and_perturb(self) -> None:
        """Reset: heads fully re-initialised; encoder + transition model are
        interpolated ``shrink_factor`` towards their old weights and
        ``perturb_factor`` towards a fresh random init. The EMA target network
        gets the same treatment against its own independent fresh init
        (official ``jit_reset`` with ``reset_target=True``, the BBF default),
        and the optimiser state is re-created — except the Adam moments of the
        shrink-and-perturbed submodules, which the official reset copies over
        (``keys_to_copy = ("encoder", "transition_model")``). Knowledge is
        carried across the reset by the replay buffer, the interpolated
        encoder weights and their optimiser moments."""
        for net in (self.network, self.target_network):
            fresh = self._make_network().to(self.device)
            for (name, p), (_, q) in zip(
                net.named_parameters(), fresh.named_parameters()
            ):
                if name.startswith(("encoder.", "transition_model.")):
                    p.mul_(self.shrink_factor).add_(q, alpha=self.perturb_factor)
                else:
                    p.copy_(q)
        old_state = self.optimizer.state
        self.optimizer = self._make_optimizer()
        for name, p in self.network.named_parameters():
            if name.startswith(("encoder.", "transition_model.")) and p in old_state:
                state = old_state[p]
                # optax's shared step count restarts at 0 in the official
                # reset, so Adam's bias correction re-warms up here too.
                state["step"] = torch.zeros_like(state["step"])
                self.optimizer.state[p] = state
        self._steps_since_reset = 0
        self._num_resets += 1

    # ------------------------------------------------------------------
    # Policy access
    # ------------------------------------------------------------------

    def get_policy(self) -> TensorDictModule:
        return self._eval_policy

    def get_explore_policy(self) -> TensorDictModule:
        return self._explore_policy

    # ------------------------------------------------------------------
    # Checkpointing (note: the replay buffer is not checkpointed)
    # ------------------------------------------------------------------

    def _get_training_state(self) -> TrainingState:
        return TrainingState(
            step=0,
            policy_state_dict=self.network.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            extra={
                "target_state_dict": self.target_network.state_dict(),
                "collected_frames": self._collected_frames,
                "grad_steps": self._grad_steps,
                "steps_since_reset": self._steps_since_reset,
                "num_resets": self._num_resets,
            },
        )

    def _load_training_state(self, state: TrainingState) -> None:
        self.network.load_state_dict(state.policy_state_dict)
        self.optimizer.load_state_dict(state.optimizer_state_dict)
        if state.extra:
            self.target_network.load_state_dict(state.extra["target_state_dict"])
            self._collected_frames = int(state.extra.get("collected_frames", 0))
            self._grad_steps = int(state.extra.get("grad_steps", 0))
            self._steps_since_reset = int(state.extra.get("steps_since_reset", 0))
            self._num_resets = int(state.extra.get("num_resets", 0))


class FixedEpsilonGreedy(TensorDictModuleBase):
    """Evaluation ε-greedy (ε = 0.001 in BBF).

    Unlike ``EGreedyModule`` it also acts under ``ExplorationType.MODE`` (used
    by ``BaseTrainer.evaluate``). The tiny ε matters on Atari: a fully
    deterministic policy can freeze (e.g. never pressing FIRE to launch the
    Breakout ball)."""

    def __init__(self, action_spec, eps: float) -> None:
        self.in_keys = ["action"]
        self.out_keys = ["action"]
        super().__init__()
        self.action_spec = action_spec
        self.eps = eps

    def forward(self, tensordict: TensorDict) -> TensorDict:
        if self.eps > 0 and float(torch.rand(())) < self.eps:
            random_action = self.action_spec.rand().to(tensordict["action"].device)
            tensordict.set("action", random_action)
        return tensordict


def _random_shift(x: torch.Tensor, pad: int) -> torch.Tensor:
    """Per-sample integer random shift of up to ``pad`` pixels (DrQ aug):
    replicate-pad then crop at a random offset, via nearest grid_sample."""
    b, _, h, w = x.shape
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    offsets = torch.randint(0, 2 * pad + 1, (b, 2), device=x.device).float()
    ys = torch.arange(h, device=x.device).float().view(1, h, 1) + offsets[:, 0].view(b, 1, 1)
    xs = torch.arange(w, device=x.device).float().view(1, 1, w) + offsets[:, 1].view(b, 1, 1)
    gy = (2.0 * ys + 1.0) / (h + 2 * pad) - 1.0
    gx = (2.0 * xs + 1.0) / (w + 2 * pad) - 1.0
    grid = torch.stack([gx.expand(b, h, w), gy.expand(b, h, w)], dim=-1)
    return F.grid_sample(x_pad, grid, mode="nearest", align_corners=False)


def _masked_nstep_return(
    reward: torch.Tensor,
    cut: torch.Tensor,
    gamma: float,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """n-step return over a sampled window, masked at episode cuts.

    ``cut[:, i]`` marks transition ``t+i`` as a bootstrap boundary (life loss,
    terminal, or env reset after it). The boundary transition's own reward
    still counts; everything after it is masked out.

    Returns:
        returns:   (B,)  sum_{i<n} gamma^i r_{t+i}, masked
        bootstrap: (B,)  1 if no cut within the horizon (else no bootstrapping)
        alive:     (B, W) alive[:, i] = 1 while no cut happened in steps <= i
    """
    alive = torch.cumprod(1.0 - cut.float(), dim=1)
    reward_mask = torch.cat([torch.ones_like(alive[:, :1]), alive[:, :-1]], dim=1)
    discounts = gamma ** torch.arange(n, device=reward.device, dtype=torch.float32)
    returns = (reward[:, :n] * reward_mask[:, :n] * discounts).sum(1)
    bootstrap = alive[:, n - 1]
    return returns, bootstrap, alive


def _project_distribution(
    next_dist: torch.Tensor,
    returns: torch.Tensor,
    bootstrap: torch.Tensor,
    support: torch.Tensor,
    gamma_n: float,
) -> torch.Tensor:
    """Categorical projection of ``returns + gamma_n * bootstrap * support``
    onto the fixed support (Bellemare et al. 2017, eq. 7)."""
    num_atoms = support.shape[0]
    v_min, v_max = float(support[0]), float(support[-1])
    delta = (v_max - v_min) / (num_atoms - 1)

    tz = returns.unsqueeze(1) + gamma_n * bootstrap.unsqueeze(1) * support.unsqueeze(0)
    tz = tz.clamp(v_min, v_max)
    pos = (tz - v_min) / delta                       # fractional atom index
    lower = pos.floor().long().clamp(0, num_atoms - 1)
    upper = pos.ceil().long().clamp(0, num_atoms - 1)
    lower_weight = (upper.float() - pos).where(lower != upper, torch.ones_like(pos))
    upper_weight = pos - lower.float()

    projected = torch.zeros_like(next_dist)
    projected.scatter_add_(1, lower, next_dist * lower_weight)
    projected.scatter_add_(1, upper, next_dist * upper_weight)
    return projected
