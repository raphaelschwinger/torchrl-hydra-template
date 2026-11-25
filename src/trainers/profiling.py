"""A `StepTrainer` that records where a training run's wall-clock time goes.

Selected with ``trainer._target_=src.trainers.profiling.ProfilingStepTrainer``
and ``+profiling.enabled=true``. Subclassing is the supported extension point;
the training loop is not forked.

**It deliberately does not override ``_training_loop()``.** Leaf timers on the
environment, the policy and the algorithm's own methods, plus a residual bucket,
give the same decomposition as a re-implemented loop without carrying a copy of
upstream loop code that would drift on the next template update.
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path

from src.profiling.phases import ROOT
from src.profiling.probes import probes_for, resolve
from src.profiling.timer import PhaseTimer
from src.trainers.step_trainer import StepTrainer
from src.utils.gpu import occupancy, physical_index

ENV_METHODS = ("reset", "step", "step_and_maybe_reset", "rand_step")


class ProfilingStepTrainer(StepTrainer):
    """`StepTrainer` with a hierarchical wall-clock profile attached."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        cfg = (self.cfg.get("profiling") or {}) if hasattr(self.cfg, "get") else {}
        self.profiling_enabled = bool(cfg.get("enabled", True))
        self.profiling_output = cfg.get("output", None)
        self.timer = PhaseTimer(
            sync_cuda=bool(cfg.get("sync_cuda", True)) and self.device.type == "cuda",
            device=self.device,
        )
        self.setup_seconds = 0.0
        self._probe_report: dict[str, str] = {}
        self._in_final_evaluation = False
        self._windows: list[dict] = []
        self._last_snapshot: dict = {}
        self._start_contention = self._contention_record()

    def setup(self) -> None:
        started = time.perf_counter()
        try:
            super().setup()
        finally:
            self.setup_seconds = time.perf_counter() - started

    def _create_collector(self) -> None:
        if self.profiling_enabled:
            self._instrument_env(self.train_env, "env_step")
            self._instrument_policies()
            self._instrument_algorithm()
        super()._create_collector()

    def _instrument_env(self, env, phase: str) -> None:
        for method in ENV_METHODS:
            self.timer.wrap(env, method, phase)

    def _instrument_policies(self) -> None:
        seen: set[int] = set()
        for getter in ("get_explore_policy", "get_policy"):
            factory = getattr(self.algorithm, getter, None)
            if factory is None:
                continue
            try:
                module = factory()
            except Exception as exc:
                self._probe_report[getter] = f"unavailable: {exc}"
                continue
            if module is None or id(module) in seen:
                continue
            seen.add(id(module))
            ok = self.timer.wrap(module, "forward", "rollout_inference")
            self._probe_report[getter] = "ok" if ok else "no forward attribute"

    def _instrument_algorithm(self) -> None:
        for dotted, phase in probes_for(self.algorithm):
            owner, attribute = resolve(self.algorithm, dotted)
            if owner is None:
                self._probe_report[dotted] = "missing"
                continue
            ok = self.timer.wrap(owner, attribute, phase)
            self._probe_report[dotted] = f"{phase}" if ok else "not callable"

    def _get_eval_env(self):
        existing = self._eval_env
        env = super()._get_eval_env()
        if self.profiling_enabled and existing is None:
            self._instrument_env(env, "env_step")
        return env

    def run_evaluation(self, num_episodes: int, step: int | None = None) -> dict:
        if not self.profiling_enabled or self._in_final_evaluation:
            return super().run_evaluation(num_episodes, step=step)
        with self.timer.phase("eval_periodic"):
            return super().run_evaluation(num_episodes, step=step)

    def run_final_evaluation(self) -> dict:
        if not self.profiling_enabled:
            return super().run_final_evaluation()
        self._in_final_evaluation = True
        try:
            with self.timer.phase("eval_final"):
                return super().run_final_evaluation()
        finally:
            self._in_final_evaluation = False

    def run(self, train: bool = True, evaluate: bool = True) -> dict:
        try:
            with self.timer.phase(ROOT):
                return super().run(train=train, evaluate=evaluate)
        finally:
            self._write_profile()

    def _training_loop(self) -> dict:
        if not self.profiling_enabled:
            return super()._training_loop()
        self._last_snapshot = self.timer.snapshot()
        original = self.log_metrics

        def log_metrics(metrics, step=None):
            original(metrics, step)
            self._record_window(step if step is not None else self._step)

        self.log_metrics = log_metrics  # type: ignore[method-assign]
        try:
            return super()._training_loop()
        finally:
            del self.log_metrics

    def _record_window(self, step: int) -> None:
        delta = self.timer.delta(self._last_snapshot)
        self._last_snapshot = self.timer.snapshot()
        self._windows.append(
            {
                "step": int(step),
                "phases": {k: v["self_seconds"] for k, v in delta.items()},
            }
        )

    def _write_profile(self) -> None:
        record = {
            "phases": self.timer.leaves(),
            "calls": dict(self.timer.calls),
            "totals": dict(self.timer.total_seconds),
            "total_wall_seconds": self.timer.total_seconds.get(ROOT, 0.0),
            "setup_seconds": self.setup_seconds,
            "agent_steps": int(self._step),
            "action_repeat": self.action_repeat,
            "sync_cuda": bool(self.timer.sync_cuda),
            "probes": self._probe_report,
            "windows": self._windows,
            "environment": self._environment_record(),
            "summary": self.summary(),
            "eval_episodes": [
                {"step": int(step), "return": float(value)}
                for step, value in self._canonical_episodes
            ],
        }
        destination = self.profiling_output or (
            Path(self.cfg.paths.output_dir) / "profile_phases.json"
        )
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        print(f"wrote profile to {path}", flush=True)

    def _contention_record(self) -> dict:
        try:
            loadavg_start = os.getloadavg()[0]
        except OSError:
            loadavg_start = float("nan")
        reading = occupancy(physical_index(default=None))
        return {
            "loadavg_start": loadavg_start,
            "gpu_mem_used_start_mib": reading["gpu_mem_used_mib"],
            "gpu_util_start_pct": reading["gpu_util_pct"],
        }

    def _environment_record(self) -> dict:
        import torch

        record = {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": str(self.device),
            "cpu_count": os.cpu_count() or 0,
            "gpu_index": self.device.index if self.device.type == "cuda" else -1,
            "gpu_name": "",
            "torchrl": "",
        }
        try:
            record["loadavg"] = os.getloadavg()[0]
        except OSError:
            record["loadavg"] = float("nan")
        record.update(self._start_contention)
        if self.device.type == "cuda":
            record["gpu_name"] = torch.cuda.get_device_name(self.device)
        try:
            import torchrl

            record["torchrl"] = torchrl.__version__
        except ImportError:
            pass
        return record
