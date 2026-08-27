"""A resident-set-size ceiling for a child process tree."""

from __future__ import annotations

import contextlib
import os
import signal
import threading
from pathlib import Path

GIB = 1024**3
DEFAULT_INTERVAL = 2.0


def _children(pid: int) -> list[int]:
    kids: list[int] = []
    try:
        for task in Path(f"/proc/{pid}/task").iterdir():
            try:
                kids.extend(int(p) for p in (task / "children").read_text().split())
            except (OSError, ValueError):
                continue
    except OSError:
        return []
    return kids


def _rss_bytes(pid: int) -> int:
    try:
        with open(f"/proc/{pid}/status") as handle:
            for line in handle:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        return 0
    return 0


def tree_rss_bytes(pid: int) -> int:
    total = 0
    stack = [pid]
    seen: set[int] = set()
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        total += _rss_bytes(current)
        stack.extend(_children(current))
    return total


class RssWatchdog:
    """Track a process tree's peak RSS and kill it if it crosses ``limit_bytes``."""

    def __init__(
        self,
        pid: int,
        limit_bytes: int | None,
        interval: float = DEFAULT_INTERVAL,
    ) -> None:
        self.pid = pid
        self.limit_bytes = limit_bytes
        self.interval = interval
        self.peak_bytes = 0
        self.killed = False
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @property
    def peak_gib(self) -> float:
        return self.peak_bytes / GIB

    def __enter__(self) -> RssWatchdog:
        self._thread.start()
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=self.interval * 2)

    def _run(self) -> None:
        while not self._stop.is_set():
            used = tree_rss_bytes(self.pid)
            self.peak_bytes = max(self.peak_bytes, used)
            if self.limit_bytes is not None and used > self.limit_bytes:
                self.killed = True
                self.kill()
                return
            self._stop.wait(self.interval)

    def kill(self) -> None:
        with contextlib.suppress(OSError):
            os.killpg(os.getpgid(self.pid), signal.SIGKILL)
