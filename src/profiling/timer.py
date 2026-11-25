"""Hierarchical wall-clock phase timer.

The profiling study needs a decomposition of one training run that *sums to the
total wall-clock*, not a sampling profile. Nested regions are therefore charged
by **self time**: a phase is credited with its own elapsed time minus whatever
its children consumed, so the leaves partition the root exactly and the residual
between the root and the leaves is real unaccounted overhead rather than
double-counted work.

Nesting also does the attribution that a flat counter cannot. Evaluation rollouts
step the same environment and call the same policy module as collection does; with
a flat timer their cost would land in the training buckets and the "evaluation is
not free" claim would be unmeasurable. Because the eval region is opened *around*
them, their samples land at ``eval_final/env_step`` instead of ``env_step``, and
the figure can collapse the whole ``eval_*`` subtree into one bar.

``sync_cuda`` is what makes the numbers mean anything on a GPU. CUDA kernels are
launched asynchronously, so an unsynchronised timer measures the launch, not the
work, and charges the wait to whichever phase happens to block next. Synchronising
at every phase boundary fixes the attribution at the cost of serialising any
CPU/GPU overlap the run would otherwise get -- a distortion the study documents
and quantifies rather than hides.
"""

from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass, field

SEPARATOR = "/"


@dataclass
class _Frame:
    path: str
    started: float
    child_seconds: float = 0.0


@dataclass
class PhaseTimer:
    """Accumulates self-time and call counts per nested phase path."""

    sync_cuda: bool = False
    device: object | None = None

    self_seconds: dict[str, float] = field(default_factory=dict)
    total_seconds: dict[str, float] = field(default_factory=dict)
    calls: dict[str, int] = field(default_factory=dict)
    _stack: list[_Frame] = field(default_factory=list)
    _open: dict[str, int] = field(default_factory=dict)

    # -- measurement ------------------------------------------------------
    def _sync(self) -> None:
        if not self.sync_cuda:
            return
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize(self.device)

    @contextlib.contextmanager
    def phase(self, name: str, collapse_nested: bool = False):
        """Time ``name`` as a child of whatever phase is currently open.

        ``collapse_nested`` folds a re-entrant call into the region already
        open under the same name instead of opening a child. TorchRL's
        ``step_and_maybe_reset`` calls ``step`` internally, so without it the
        environment shows up as ``env_step/env_step`` -- correctly costed, since
        self-time never double-counts, but split across two rows for a reason
        that is an implementation detail of the caller.
        """
        if collapse_nested and self._open.get(name):
            yield
            return
        parent = self._stack[-1].path if self._stack else ""
        path = f"{parent}{SEPARATOR}{name}" if parent else name
        self._sync()
        frame = _Frame(path=path, started=time.perf_counter())
        self._stack.append(frame)
        self._open[name] = self._open.get(name, 0) + 1
        try:
            yield
        finally:
            self._open[name] -= 1
            self._sync()
            elapsed = time.perf_counter() - frame.started
            popped = self._stack.pop()
            # A phase re-entered under a different parent is a different path,
            # which is exactly what we want: `env_step` under `eval_final` is
            # not the same measurement as `env_step` under the collector.
            assert popped is frame
            own = elapsed - frame.child_seconds
            self.self_seconds[path] = self.self_seconds.get(path, 0.0) + own
            self.total_seconds[path] = self.total_seconds.get(path, 0.0) + elapsed
            self.calls[path] = self.calls.get(path, 0) + 1
            if self._stack:
                self._stack[-1].child_seconds += elapsed

    def wrap(self, obj, attribute: str, name: str, collapse_nested: bool = True) -> bool:
        """Time every call to ``obj.attribute`` as phase ``name``.

        Patches the *instance* attribute, so the class -- and any other object of
        it -- is untouched. Returns False when the attribute does not exist,
        because the probe tables describe several algorithms and a missing hook
        is a fact to record, not a crash.
        """
        original = getattr(obj, attribute, None)
        if original is None or not callable(original):
            return False

        def timed(*args, **kwargs):
            with self.phase(name, collapse_nested=collapse_nested):
                return original(*args, **kwargs)

        timed.__wrapped__ = original  # type: ignore[attr-defined]
        setattr(obj, attribute, timed)
        return True

    # -- reporting --------------------------------------------------------
    def snapshot(self) -> dict[str, dict[str, float]]:
        """Current totals, safe to call while phases are open."""
        return {
            path: {
                "self_seconds": seconds,
                "total_seconds": self.total_seconds.get(path, 0.0),
                "calls": float(self.calls.get(path, 0)),
            }
            for path, seconds in sorted(self.self_seconds.items())
        }

    def delta(self, previous: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
        """Difference against an earlier snapshot, for per-window series."""
        now = self.snapshot()
        out: dict[str, dict[str, float]] = {}
        for path, values in now.items():
            before = previous.get(path, {})
            out[path] = {k: v - before.get(k, 0.0) for k, v in values.items()}
        return out

    def leaves(self) -> dict[str, float]:
        """Self-time per path, dropping paths that are pure containers.

        A path with self-time is kept even if it also has children: a region can
        legitimately do work of its own on top of what it nests.
        """
        return dict(sorted(self.self_seconds.items()))
