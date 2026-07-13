"""TorchRL adapter for pixel-based DER, SPR, and BBF agents.

The learning core is intentionally ported close to ``BBF-pytorch``.  This
adapter only translates between TorchRL TensorDict collection and the
NumPy-backed replay/agent API used by those agents.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Type

import numpy as np
import torch
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.envs import EnvBase

from src.replay.prioritized import PrioritizedSubsequenceReplayBuffer
from src.algorithms.base import BaseAlgorithm, CollectorConfig, TrainingState


ConfigT = Any
AgentT = Any


class PixelControlPolicy(nn.Module):
    """Stateful TensorDict policy that builds BBF-style frame stacks."""

    def __init__(
        self,
        agent: AgentT,
        *,
        obs_key: str,
        eval_mode: bool,
        one_hot_actions: bool,
    ) -> None:
        super().__init__()
        self.agent = agent
        self.obs_key = obs_key
        self.eval_mode = eval_mode
        self.one_hot_actions = one_hot_actions
        self._frames: list[deque[np.ndarray]] = []

    def reset_stack(self) -> None:
        self._frames = []

    def forward(self, td: TensorDict) -> TensorDict:
        pixels = td.get(self.obs_key)
        frames = _pixels_to_numpy_frames(pixels)
        batch_size = frames.shape[0]
        is_init = _is_init(td, batch_size)
        if len(self._frames) != batch_size:
            self._frames = [
                deque(maxlen=self.agent.config.stack_size) for _ in range(batch_size)
            ]
            is_init = np.ones(batch_size, dtype=bool)

        states = []
        for i, frame in enumerate(frames):
            if is_init[i] or len(self._frames[i]) == 0:
                self._frames[i].clear()
                for _ in range(self.agent.config.stack_size):
                    self._frames[i].append(frame)
            else:
                self._frames[i].append(frame)
            states.append(np.stack(list(self._frames[i]), axis=-1))

        state = np.stack(states, axis=0)
        if (
            not self.eval_mode
            and self.agent.training_steps < int(self.agent.config.min_replay_history)
        ):
            actions = torch.randint(
                self.agent.config.num_actions,
                (batch_size,),
                generator=self.agent.generator,
                device=self.agent.device,
            )
        else:
            actions = self.agent.select_action(state, eval_mode=self.eval_mode).reshape(-1)
        if self.one_hot_actions:
            actions = F.one_hot(
                actions.long(),
                num_classes=self.agent.config.num_actions,
            ).to(torch.int64)
        if len(td.batch_size) == 0 and actions.shape[0] == 1:
            actions = actions[0]
        td.set("action", actions.to(td.device))
        return td


class PixelControlAlgorithm(BaseAlgorithm):
    """Shared framework implementation for DER, SPR, and BBF pixel-control agents."""

    agent_cls: Type[AgentT]
    config_cls: Type[ConfigT]
    prefetch_fixed_replay = False

    def __init__(
        self,
        device: torch.device | None = None,
        *,
        obs_key: str = "pixels",
        replay_capacity: int = 200_000,
        seed: int = 0,
        frames_per_batch: int = 1,
        max_frames_per_traj: int = -1,
        num_actions: int | None = None,
        observation_shape: tuple[int, int] = (84, 84),
        stack_size: int = 4,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        gamma: float = 0.99,
        update_horizon: int = 10,
        max_update_horizon: int | None = None,
        min_gamma: float | None = None,
        cycle_steps: int | None = None,
        learning_rate: float = 1e-4,
        adam_eps: float = 1.5e-4,
        weight_decay: float = 0.0,
        noisy: bool = True,
        dueling: bool = True,
        double_dqn: bool = True,
        distributional: bool = True,
        replay_ratio: int = 32,
        batch_size: int = 32,
        target_update_period: int = 8000,
        target_update_tau: float = 1.0,
        max_target_update_tau: float | None = None,
        epsilon_train: float = 0.01,
        epsilon_eval: float = 0.001,
        epsilon_decay_period: int = 2000,
        min_replay_history: int = 1600,
        encoder_type: str = "dqn",
        hidden_dim: int = 512,
        width_scale: int = 1,
        renormalize_output: bool = False,
        data_augmentation: bool = False,
        batches_to_group: int = 1,
        eval_noise: bool = True,
        target_eval_mode: bool = False,
        spr_weight: float | None = None,
        jumps: int | None = None,
        reset_every: int | None = None,
        reset_offset: int | None = None,
        no_resets_after: int | None = None,
        reset_priorities: bool | None = None,
        reset_interval_scaling: float | str | None = None,
        offline_update_frac: float | None = None,
        shrink_perturb_keys: tuple[str, ...] | None = None,
        shrink_factor: float | None = None,
        perturb_factor: float | None = None,
        reset_projection: bool | None = None,
        reset_encoder: bool | None = None,
        reset_noise: bool | None = None,
        reset_head: bool | None = None,
        reset_target: bool | None = None,
        target_action_selection: bool | None = None,
        match_online_target_rngs: bool | None = None,
        entropy_decay_period: int | None = None,
        entropy_initial_coef: float | None = None,
        entropy_final_coef: float | None = None,
        policy_learning_rate: float | None = None,
        alpha_learning_rate: float | None = None,
    ) -> None:
        super().__init__(device)
        self.obs_key = obs_key
        self.replay_capacity = replay_capacity
        self.seed = seed
        self.frames_per_batch = frames_per_batch
        self.max_frames_per_traj = max_frames_per_traj
        self._config_kwargs = {
            "num_actions": num_actions,
            "observation_shape": observation_shape,
            "stack_size": stack_size,
            "num_atoms": num_atoms,
            "v_min": v_min,
            "v_max": v_max,
            "gamma": gamma,
            "update_horizon": update_horizon,
            "learning_rate": learning_rate,
            "adam_eps": adam_eps,
            "weight_decay": weight_decay,
            "noisy": noisy,
            "dueling": dueling,
            "double_dqn": double_dqn,
            "distributional": distributional,
            "replay_ratio": replay_ratio,
            "batch_size": batch_size,
            "target_update_period": target_update_period,
            "target_update_tau": target_update_tau,
            "epsilon_train": epsilon_train,
            "epsilon_eval": epsilon_eval,
            "epsilon_decay_period": epsilon_decay_period,
            "min_replay_history": min_replay_history,
            "encoder_type": encoder_type,
            "hidden_dim": hidden_dim,
            "width_scale": width_scale,
            "renormalize_output": renormalize_output,
            "data_augmentation": data_augmentation,
            "batches_to_group": batches_to_group,
            "eval_noise": eval_noise,
            "target_eval_mode": target_eval_mode,
        }
        optional = {
            "max_update_horizon": max_update_horizon,
            "min_gamma": min_gamma,
            "cycle_steps": cycle_steps,
            "max_target_update_tau": max_target_update_tau,
            "spr_weight": spr_weight,
            "jumps": jumps,
            "reset_every": reset_every,
            "reset_offset": reset_offset,
            "no_resets_after": no_resets_after,
            "reset_priorities": reset_priorities,
            "reset_interval_scaling": reset_interval_scaling,
            "offline_update_frac": offline_update_frac,
            "shrink_perturb_keys": shrink_perturb_keys,
            "shrink_factor": shrink_factor,
            "perturb_factor": perturb_factor,
            "reset_projection": reset_projection,
            "reset_encoder": reset_encoder,
            "reset_noise": reset_noise,
            "reset_head": reset_head,
            "reset_target": reset_target,
            "target_action_selection": target_action_selection,
            "match_online_target_rngs": match_online_target_rngs,
            "entropy_decay_period": entropy_decay_period,
            "entropy_initial_coef": entropy_initial_coef,
            "entropy_final_coef": entropy_final_coef,
            "policy_learning_rate": policy_learning_rate,
            "alpha_learning_rate": alpha_learning_rate,
        }
        self._config_kwargs.update(
            {key: value for key, value in optional.items() if value is not None}
        )
        self._collected_frames = 0
        self._sample_executor: ThreadPoolExecutor | None = None
        self._prefetched_sample: Future[dict[str, Any]] | None = None

    def setup(self, make_env: Callable[[], EnvBase]) -> None:
        proof_env = make_env()
        try:
            action_spec = proof_env.action_spec
            num_actions = self._config_kwargs["num_actions"]
            if num_actions is None:
                num_actions = int(action_spec.space.n)
            self._config_kwargs["num_actions"] = num_actions
            self._one_hot_actions = bool(
                len(action_spec.shape) > 0 and action_spec.shape[-1] == num_actions
            )
        finally:
            proof_env.close()

        config_kwargs = dict(self._config_kwargs)
        config_kwargs["device"] = str(self.device)
        config = self.config_cls(**config_kwargs)
        self.agent = self.agent_cls(config, seed=self.seed)
        self.replay = PrioritizedSubsequenceReplayBuffer(
            observation_shape=config.observation_shape,
            stack_size=config.stack_size,
            replay_capacity=self.replay_capacity,
            batch_size=config.batch_size,
            subseq_len=self._subseq_len(config),
            n_envs=1,
            update_horizon=self._replay_update_horizon(config),
            gamma=config.gamma,
            seed=self.seed,
        )
        self._policy = PixelControlPolicy(
            self.agent,
            obs_key=self.obs_key,
            eval_mode=True,
            one_hot_actions=self._one_hot_actions,
        )
        self._explore_policy = PixelControlPolicy(
            self.agent,
            obs_key=self.obs_key,
            eval_mode=False,
            one_hot_actions=self._one_hot_actions,
        )

    def get_collector_config(self) -> CollectorConfig:
        return CollectorConfig(
            frames_per_batch=self.frames_per_batch,
            init_random_frames=0,
            max_frames_per_traj=self.max_frames_per_traj,
        )

    def step(self, batch: TensorDict) -> dict[str, float]:
        flat = batch.reshape(-1).cpu()
        all_metrics: list[dict[str, Any]] = []
        for transition in flat:
            if self._collected_frames > int(self.agent.config.min_replay_history):
                self.agent.training_steps = self._collected_frames
                all_metrics.append(self._train_step_updates())
                self._finish_replay_prefetch_before_add()
            self._add_transition(transition)
            self._collected_frames += 1
            self.agent.training_steps = self._collected_frames
        metrics = _merge_metrics(all_metrics, prefix="train/")
        metrics["train/epsilon"] = self._current_epsilon()
        return metrics

    def get_policy(self) -> TensorDictModule:
        self._policy.reset_stack()
        return self._policy  # type: ignore[return-value]

    def get_explore_policy(self) -> TensorDictModule:
        return self._explore_policy  # type: ignore[return-value]

    def _add_transition(self, td: TensorDict) -> None:
        obs = _pixels_to_numpy_frames(td.get(self.obs_key))[0]
        action = _action_to_int(td.get("action"))
        reward = float(td.get(("next", "reward")).reshape(-1)[0].item())
        done = _terminal_from_transition(td)
        self.replay.add(
            obs[None],
            np.array([action], dtype=np.int32),
            np.array([reward], dtype=np.float32),
            np.array([done], dtype=np.uint8),
            priority=np.array(
                [self.replay.sum_tree.max_recorded_priority],
                dtype=np.float32,
            ),
            episode_end=np.array([done], dtype=np.uint8),
        )

    def _train_step_updates(self) -> dict[str, Any]:
        all_metrics: list[dict[str, Any]] = []
        prefetch_enabled = self._can_prefetch_replay()
        for _ in range(self._update_groups_per_train_step()):
            batch = (
                self._take_replay_sample()
                if prefetch_enabled
                else self._sample_replay_batch()
            )
            if prefetch_enabled:
                self._start_replay_prefetch()
            metrics = self.agent.train_step(batch)
            if getattr(self.agent, "reset_priorities_requested", False):
                self.replay.reset_priorities()
                self.agent.reset_priorities_requested = False
            self.replay.set_priority(batch["indices"], metrics["priorities"])
            all_metrics.append(metrics)
        return _merge_metrics(all_metrics, prefix="")

    def _can_prefetch_replay(self) -> bool:
        """Overlap fixed-schedule SPR replay sampling with GPU training."""
        return (
            self.device.type == "cuda"
            and self.prefetch_fixed_replay
        )

    def _replay_sample_kwargs(self) -> dict[str, Any]:
        return {
            "batch_size": self._current_sample_batch_size(),
            "update_horizon": self._current_sample_update_horizon(),
            "gamma": self._current_sample_gamma(),
            "as_torch": False,
        }

    def _sample_replay_batch(self) -> dict[str, Any]:
        return self.replay.sample_transition_batch(**self._replay_sample_kwargs())

    def _ensure_sample_executor(self) -> ThreadPoolExecutor:
        if self._sample_executor is None:
            self._sample_executor = ThreadPoolExecutor(max_workers=1)
        return self._sample_executor

    def _start_replay_prefetch(self) -> None:
        if self._prefetched_sample is not None:
            return
        self._prefetched_sample = self._ensure_sample_executor().submit(
            self._sample_replay_batch
        )

    def _take_replay_sample(self) -> dict[str, Any]:
        if self._prefetched_sample is None:
            return self._sample_replay_batch()
        future = self._prefetched_sample
        self._prefetched_sample = None
        return future.result()

    def _finish_replay_prefetch_before_add(self) -> None:
        if self._prefetched_sample is not None:
            self._prefetched_sample.result()

    def _updates_per_train_step(self) -> int:
        replay_ratio = int(self.agent.config.replay_ratio)
        batch_size = int(self.agent.config.batch_size)
        return max(1, replay_ratio // batch_size)

    def _current_batches_to_group(self) -> int:
        updates = self._updates_per_train_step()
        configured = int(self.agent.config.batches_to_group)
        grouped = min(configured, updates)
        if updates % grouped != 0:
            raise ValueError(
                f"updates_per_train_step={updates} must be divisible by "
                f"batches_to_group={grouped}."
            )
        return grouped

    def _update_groups_per_train_step(self) -> int:
        return max(1, self._updates_per_train_step() // self._current_batches_to_group())

    def _current_sample_batch_size(self) -> int:
        return int(self.agent.config.batch_size * self._current_batches_to_group())

    def _current_sample_update_horizon(self) -> int | None:
        if hasattr(self.agent, "current_update_horizon"):
            return int(self.agent.current_update_horizon())
        return None

    def _current_sample_gamma(self) -> float | None:
        if hasattr(self.agent, "current_gamma"):
            return float(self.agent.current_gamma())
        return None

    def _current_epsilon(self) -> float:
        if self._collected_frames <= int(self.agent.config.min_replay_history):
            return 1.0
        if self.agent.config.epsilon_decay_period <= 0:
            return float(self.agent.config.epsilon_train)
        steps_left = (
            self.agent.config.epsilon_decay_period
            + self.agent.config.min_replay_history
            - self.agent.training_steps
        )
        bonus = (
            (1.0 - self.agent.config.epsilon_train)
            * steps_left
            / self.agent.config.epsilon_decay_period
        )
        bonus = min(max(bonus, 0.0), 1.0 - self.agent.config.epsilon_train)
        return float(self.agent.config.epsilon_train + bonus)

    def _get_training_state(self) -> TrainingState:
        return TrainingState(
            step=self._collected_frames,
            policy_state_dict=self.agent.state_dict(),
            optimizer_state_dict={},
            extra={"replay": self.replay},
        )

    def _load_training_state(self, state: TrainingState) -> None:
        self.agent.load_state_dict(state.policy_state_dict)
        self._collected_frames = int(state.step)
        if state.extra and "replay" in state.extra:
            self.replay = state.extra["replay"]

    def _subseq_len(self, config: ConfigT) -> int:
        return int(getattr(config, "jumps", 0)) + 1

    def _replay_update_horizon(self, config: ConfigT) -> int:
        return int(getattr(config, "max_update_horizon", config.update_horizon))


def _pixels_to_numpy_frames(pixels: torch.Tensor) -> np.ndarray:
    tensor = pixels.detach().cpu()
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim == 3:
        if tensor.shape[0] == 1:
            tensor = tensor.squeeze(0).unsqueeze(0)
        else:
            tensor = tensor.unsqueeze(0)
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.ndim == 4 and tensor.shape[-1] == 1:
        tensor = tensor[..., 0]
    if tensor.dtype.is_floating_point:
        max_value = float(tensor.max().item()) if tensor.numel() else 0.0
        if max_value <= 1.0:
            tensor = tensor * 255.0
        tensor = tensor.clamp(0, 255).to(torch.uint8)
    return tensor.numpy().astype(np.uint8)


def _is_init(td: TensorDict, batch_size: int) -> np.ndarray:
    value = td.get("is_init", default=None)
    end_of_life = td.get("end-of-life", default=None)
    if value is None:
        out = np.zeros(batch_size, dtype=bool)
    else:
        out = value.detach().cpu().reshape(-1).numpy().astype(bool)
    if end_of_life is not None:
        out = out | end_of_life.detach().cpu().reshape(-1).numpy().astype(bool)
    return out


def _terminal_from_transition(td: TensorDict) -> bool:
    end_of_life = td.get(("next", "end-of-life"), default=None)
    if end_of_life is not None:
        return bool(end_of_life.reshape(-1)[0].item())

    for key in ("done", "terminated", "truncated"):
        value = td.get(("next", key), default=None)
        if value is not None and bool(value.reshape(-1)[0].item()):
            return True
    return False


def _action_to_int(action: torch.Tensor) -> int:
    action = action.detach().cpu()
    if action.numel() > 1:
        return int(action.reshape(-1).argmax().item())
    return int(action.reshape(-1)[0].item())


def _merge_metrics(metrics: list[dict[str, Any]], *, prefix: str) -> dict[str, float]:
    if not metrics:
        return {}
    merged: dict[str, float] = {}
    keys = set().union(*(metric.keys() for metric in metrics))
    for key in keys:
        if key == "priorities":
            continue
        values = [metric[key] for metric in metrics if key in metric]
        first = values[0]
        if isinstance(first, np.ndarray):
            merged[f"{prefix}{key}"] = float(
                np.mean(np.stack([np.asarray(value) for value in values], axis=0))
            )
        else:
            merged[f"{prefix}{key}"] = float(np.mean([float(value) for value in values]))
    return merged
