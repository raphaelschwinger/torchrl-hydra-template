from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class RunData:
    run_id: str
    algo_label: str
    env_name: str
    seed: int | None
    summary_score: float | None
    # (frame, score) pairs fetched only when fetch_history=True
    history: list[tuple[int, float]] = field(default_factory=list)


class DataSource(Protocol):
    def fetch_runs(
        self,
        *,
        algos: list[str] | None,
        envs: list[str] | None,
        metric: str,
        fetch_history: bool = False,
        history_metric: str = "episode/score",
    ) -> list[RunData]: ...
