from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class BenchmarkSpec:
    name: str
    games: list[str]               # ALE/DMC/etc env IDs
    budget_frames: int             # evaluation frame budget
    score_metric: str              # W&B summary key used as eval score

    # Called as prepare(raw_score, game_id) before scores enter rliable.
    # Default: identity (raw scores pass through unchanged — e.g. DMControl).
    # Atari100k sets this to human-normalise using human/random baselines.
    # DMControl could set this to lambda raw, _: raw / 1000.0 etc.
    prepare: Callable[[float, str], float] = field(
        default=lambda raw, _: raw
    )
