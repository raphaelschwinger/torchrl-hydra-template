"""Base trainer and callback infrastructure."""
from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import torch
from omegaconf import DictConfig
from torchrl.envs.utils import ExplorationType, set_exploration_type

from src.algorithms.base import BaseAlgorithm
from src.environments.environment import Environment
from src.utils.device import resolve_device


class TrainerEvent(Enum):
    ON_TRAIN_START = auto()
    ON_METRICS = auto()
    ON_STEP_END = auto()
    ON_TRAIN_END = auto()


@runtime_checkable
class Callback(Protocol):
    """Callbacks implement whichever hooks they care about; the rest are skipped.

    ``on_metrics`` and ``on_step_end`` are deliberately distinct. ``on_metrics``
    means "here is one row of metrics at this x position" and fires once per
    completed episode as well as on log boundaries — loggers want it.
    ``on_step_end`` means "the training loop advanced past a boundary" and fires
    only on log boundaries — the progress bar and checkpointer want that, and
    would misbehave if driven per episode.
    """

    def on_train_start(self, state: dict[str, Any]) -> None: ...
    def on_metrics(self, metrics: dict[str, float], step: int) -> None: ...
    def on_step_end(self, metrics: dict[str, float], step: int) -> None: ...
    def on_train_end(self, state: dict[str, Any]) -> None: ...


def fire_callbacks(
    event: TrainerEvent,
    callbacks: list,
    *args: Any,
    **kwargs: Any,
) -> None:
    """Dispatch a training event to all callbacks that implement the matching method."""
    method_name = event.name.lower()
    for cb in callbacks:
        method = getattr(cb, method_name, None)
        if callable(method):
            method(*args, **kwargs)


class BaseTrainer(ABC):
    """Base class for all trainers.

    Owns: device resolution, environment creation, the evaluation protocol,
    metric emission, callbacks, and checkpoint orchestration.

    All metrics are logged against ``global_step``, which counts **agent steps**
    (post frame-skip / action-repeat) uniformly across algorithms. ``frames``
    (``global_step * action_repeat``) is logged alongside it. No algorithm may
    define its own x-axis.

    Args:
        cfg: full Hydra config
        algorithm: algorithm instance (already ``__init__``'d, not yet set up)
        environment: environment config wrapper used for training
        eval_environment: optional separate environment used by ``evaluate()``;
            falls back to ``environment`` when ``None``
        callbacks: list of callback objects
    """

    def __init__(
        self,
        cfg: DictConfig,
        algorithm: BaseAlgorithm,
        environment: Environment,
        eval_environment: Environment | None = None,
        callbacks: list | None = None,
    ) -> None:
        self.cfg = cfg
        self.trainer_cfg = cfg.trainer
        self.eval_cfg = cfg.get("evaluation") or {}
        self.algorithm = algorithm
        self.environment = environment
        self.eval_environment = eval_environment or environment
        self.callbacks = callbacks or []

        self.device = resolve_device(
            self.trainer_cfg.accelerator,
            list(self.trainer_cfg.devices),
        )
        self.algorithm.device = self.device

        self._step: int = 0
        self._eval_env = None
        self._eval_calls: int = 0
        # (global_step, return) for every episode from the canonical source,
        # used for the end-of-run summary.
        self._canonical_episodes: list[tuple[int, float]] = []

    @property
    def action_repeat(self) -> int:
        """Env frames per agent step, for deriving ``frames`` from ``global_step``."""
        return int(getattr(self.environment, "action_repeat", 1) or 1)

    @property
    def canonical_source(self) -> str:
        """Which stream feeds ``charts/episodic_return``: ``"train"`` or ``"eval"``."""
        return str(self.eval_cfg.get("canonical_source", "train"))

    def setup(self) -> None:
        """Create environment and set up the algorithm."""
        num_envs = int(self.trainer_cfg.get("num_envs", 1))

        def make_env():
            return self.environment.make_env(
                num_envs=num_envs,
                device=str(self.device),
            )

        self.train_env = make_env()
        self.train_env.set_seed(int(self.trainer_cfg.seed))
        self.algorithm.setup(make_env)

    def start(self) -> None:
        """Fire ``ON_TRAIN_START`` (logger init, checkpoint dir creation)."""
        fire_callbacks(
            TrainerEvent.ON_TRAIN_START,
            self.callbacks,
            state={"cfg": self.cfg},
        )

    def finish(self) -> None:
        """Fire ``ON_TRAIN_END`` and release the evaluation env."""
        try:
            fire_callbacks(
                TrainerEvent.ON_TRAIN_END,
                self.callbacks,
                state={"cfg": self.cfg, "summary": self.summary()},
            )
        finally:
            self.close_eval_env()

    def run(self, train: bool = True, evaluate: bool = True) -> dict[str, float]:
        """Full lifecycle: start -> optional training -> optional final eval -> finish.

        ``train`` and ``evaluate`` come from the top-level ``train:`` / ``eval:``
        flags. ``ON_TRAIN_END`` fires in a ``finally`` block so checkpoint
        saving and logger teardown still happen if evaluation raises.
        """
        metrics: dict[str, float] = {}
        self.start()
        try:
            if train:
                metrics = self._training_loop()
            if evaluate:
                metrics = {**metrics, **self.run_final_evaluation()}
        finally:
            self.finish()
        return metrics

    def fit(self) -> dict[str, float]:
        """Train and run the post-training evaluation stage."""
        return self.run()

    @abstractmethod
    def _training_loop(self) -> dict[str, float]:
        """Subclass-specific training loop."""

    # ------------------------------------------------------------------ metrics

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Emit one metric row at ``step`` (agent steps).

        ``global_step`` and ``frames`` are injected into every row. This is not
        cosmetic: openrlbenchmark joins on them with
        ``run.history(keys=[xaxis, "_runtime", metric]).dropna()``, so a metric
        logged without ``global_step`` *in the same row* silently contributes
        nothing to a comparison.
        """
        step = self._step if step is None else int(step)
        row = dict(metrics)
        row["global_step"] = int(step)
        row["frames"] = int(step) * self.action_repeat
        fire_callbacks(
            TrainerEvent.ON_METRICS, self.callbacks, metrics=row, step=int(step)
        )

    def log_episodes(
        self,
        returns: list[float],
        lengths: list[float] | None,
        step: int,
        source: str,
    ) -> None:
        """Emit **one row per episode** from ``source`` (``"train"`` / ``"eval"``).

        Episodes are logged individually rather than as a windowed mean because
        openrlbenchmark reports the mean of the *last 100 logged points* of the
        metric; a pre-aggregated point per log boundary would make that window
        mean something different for every run.

        Episodes from the configured ``canonical_source`` are mirrored into
        ``charts/episodic_return`` / ``charts/episodic_length``, which is what
        openrlbenchmark reads by default.
        """
        if not returns:
            return
        canonical = source == self.canonical_source
        for i, ret in enumerate(returns):
            row: dict[str, float] = {f"charts/{source}_episodic_return": float(ret)}
            length = lengths[i] if lengths is not None and i < len(lengths) else None
            if length is not None:
                row[f"charts/{source}_episodic_length"] = float(length)
            if canonical:
                row["charts/episodic_return"] = float(ret)
                if length is not None:
                    row["charts/episodic_length"] = float(length)
                self._canonical_episodes.append((int(step), float(ret)))
            self.log_metrics(row, step)

    def summary(self) -> dict[str, float]:
        """End-of-run headline numbers, using openrlbenchmark's own rule.

        Mean/std of the canonical episodic return over the last
        ``evaluation.summary_window`` episodes. ``evaluation.summary_max_step``
        drops episodes completed past a benchmark budget the run deliberately
        trains beyond (DreamerV3 trains 10% past Atari-100k but reports at 100k).
        """
        episodes = self._canonical_episodes
        max_step = self.eval_cfg.get("summary_max_step", None)
        if max_step is not None:
            episodes = [(s, r) for s, r in episodes if s <= int(max_step)]
        if not episodes:
            return {}
        window = int(self.eval_cfg.get("summary_window", 100) or 100)
        recent = [r for _, r in episodes[-window:]]
        t = torch.tensor(recent, dtype=torch.float32)
        return {
            "eval/final_return_mean": t.mean().item(),
            "eval/final_return_std": t.std().item() if len(recent) > 1 else 0.0,
            "eval/final_return_episodes": float(len(recent)),
        }

    # --------------------------------------------------------------- evaluation

    def close_eval_env(self) -> None:
        if self._eval_env is not None:
            self._eval_env.close()
            self._eval_env = None

    def _get_eval_env(self):
        """Build the evaluation env once and reuse it across evaluation points.

        Rebuilding per call would pay env construction on every periodic
        evaluation, which is significant for ALE.
        """
        if self._eval_env is None:
            self._eval_env = self.eval_environment.make_env(
                num_envs=1,
                device=str(self.device),
            )
        return self._eval_env

    def _algorithm_modules(self) -> list[torch.nn.Module]:
        return [m for m in vars(self.algorithm).values() if isinstance(m, torch.nn.Module)]

    def evaluate(self, num_episodes: int) -> tuple[list[float], list[float]]:
        """Roll out ``num_episodes`` complete episodes with the evaluation policy.

        Returns:
            ``(returns, lengths)`` — one entry per episode.
        """
        env = self._get_eval_env()
        seed = self.eval_cfg.get("seed", None)
        if seed is not None:
            # Reproducible without making every evaluation point identical.
            env.set_seed(int(seed) + self._eval_calls)
        self._eval_calls += 1

        # Snapshot *before* get_policy(): some algorithms mutate shared module
        # state there (Rainbow toggles noisy layers via .train(eval_noise)).
        # That was safe only while evaluation could not overlap training;
        # periodic evaluation breaks that assumption.
        modes = [(m, m.training) for m in self._algorithm_modules()]

        use_explore = str(self.eval_cfg.get("policy", "eval")) == "explore"
        policy = (
            self.algorithm.get_explore_policy()
            if use_explore
            else self.algorithm.get_policy()
        )
        exploration = ExplorationType.RANDOM if use_explore else ExplorationType.MODE

        returns: list[float] = []
        lengths: list[float] = []
        try:
            with torch.no_grad(), set_exploration_type(exploration):
                for _ in range(num_episodes):
                    td = env.reset()
                    episode_return = 0.0
                    episode_length = 0
                    done = False
                    while not done:
                        td = policy(td)
                        td = env.step(td)
                        episode_return += td["next", "reward"].sum().item()
                        episode_length += 1
                        done = (
                            td["next", "done"].any().item()
                            or td["next", "terminated"].any().item()
                        )
                        # `td["next"]` would keep only env-written keys and drop
                        # everything the policy left at the root — which for a
                        # recurrent policy is its whole state (DreamerPolicy's
                        # `stoch` / `deter` / `prev_action`), silently resetting
                        # the RSSM on every step. `env.step_mdp` is what the
                        # collector uses (`_StepMDP(keep_other=True)`), so
                        # evaluation now advances exactly like collection.
                        td = env.step_mdp(td)
                    returns.append(episode_return)
                    lengths.append(float(episode_length))
        finally:
            for module, was_training in modes:
                module.train(was_training)
        return returns, lengths

    def run_evaluation(
        self, num_episodes: int, step: int | None = None
    ) -> dict[str, float]:
        """Evaluate and log: one row per episode plus the ``eval/*`` aggregate."""
        if num_episodes <= 0:
            return {}
        step = self._step if step is None else int(step)
        returns, lengths = self.evaluate(num_episodes)

        self.log_episodes(returns, lengths, step, source="eval")

        t = torch.tensor(returns, dtype=torch.float32)
        aggregate = {
            "eval/return_mean": t.mean().item(),
            # std of a single episode is NaN, which poisons downstream averaging.
            "eval/return_std": t.std().item() if len(returns) > 1 else 0.0,
            "eval/return_min": t.min().item(),
            "eval/return_max": t.max().item(),
            "eval/episodes": float(len(returns)),
        }
        self.log_metrics(aggregate, step)
        return aggregate

    def run_final_evaluation(self) -> dict[str, float]:
        """Post-training evaluation stage (top-level ``eval:`` flag)."""
        num_episodes = int(self.eval_cfg.get("final_num_episodes", 0) or 0)
        if num_episodes <= 0:
            return {}
        metrics = self.run_evaluation(num_episodes, step=self._step)
        print(
            f"Final evaluation ({num_episodes} episodes): "
            + ", ".join(f"{k}={v:.2f}" for k, v in metrics.items())
        )
        return metrics

    def save_checkpoint(self, path: str | Path) -> None:
        """Save algorithm state + trainer step."""
        self.algorithm.save_checkpoint(path, step=self._step)

    def load_checkpoint(self, path: str | Path) -> None:
        """Restore algorithm state + trainer step."""
        self._step = self.algorithm.load_checkpoint(path)

    def _should_log(self, log_every: int, batch_frames: int) -> bool:
        """Check if we crossed a ``log_every`` boundary this iteration."""
        prev = self._step - batch_frames
        return prev // log_every < self._step // log_every
