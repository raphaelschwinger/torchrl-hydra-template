"""Atari100k benchmark spec — 26-game suite, 400k game frame budget.

Kaiser et al. (2020) "Model-Based Reinforcement Learning for Atari".
Human/random baselines from the DreamerV3 paper appendix (Hafner et al. 2025).
"""
from .base import BenchmarkSpec

GAMES = [
    "ALE/Alien-v5",
    "ALE/Amidar-v5",
    "ALE/Assault-v5",
    "ALE/Asterix-v5",
    "ALE/BankHeist-v5",
    "ALE/BattleZone-v5",
    "ALE/Boxing-v5",
    "ALE/Breakout-v5",
    "ALE/ChopperCommand-v5",
    "ALE/CrazyClimber-v5",
    "ALE/DemonAttack-v5",
    "ALE/Freeway-v5",
    "ALE/Frostbite-v5",
    "ALE/Gopher-v5",
    "ALE/Hero-v5",
    "ALE/Jamesbond-v5",
    "ALE/Kangaroo-v5",
    "ALE/Krull-v5",
    "ALE/KungFuMaster-v5",
    "ALE/MsPacman-v5",
    "ALE/Pong-v5",
    "ALE/PrivateEye-v5",
    "ALE/Qbert-v5",
    "ALE/RoadRunner-v5",
    "ALE/Seaquest-v5",
    "ALE/UpNDown-v5",
]

_HUMAN: dict[str, float] = {
    "ALE/Alien-v5": 7128,
    "ALE/Amidar-v5": 1720,
    "ALE/Assault-v5": 742,
    "ALE/Asterix-v5": 8503,
    "ALE/BankHeist-v5": 753,
    "ALE/BattleZone-v5": 37188,
    "ALE/Boxing-v5": 12,
    "ALE/Breakout-v5": 30,
    "ALE/ChopperCommand-v5": 7388,
    "ALE/CrazyClimber-v5": 35829,
    "ALE/DemonAttack-v5": 1971,
    "ALE/Freeway-v5": 30,
    "ALE/Frostbite-v5": 4335,
    "ALE/Gopher-v5": 2412,
    "ALE/Hero-v5": 30826,
    "ALE/Jamesbond-v5": 303,
    "ALE/Kangaroo-v5": 3035,
    "ALE/Krull-v5": 2666,
    "ALE/KungFuMaster-v5": 22736,
    "ALE/MsPacman-v5": 6952,
    "ALE/Pong-v5": 15,
    "ALE/PrivateEye-v5": 69571,
    "ALE/Qbert-v5": 13455,
    "ALE/RoadRunner-v5": 7845,
    "ALE/Seaquest-v5": 42055,
    "ALE/UpNDown-v5": 11693,
}

_RANDOM: dict[str, float] = {
    "ALE/Alien-v5": 228,
    "ALE/Amidar-v5": 6,
    "ALE/Assault-v5": 222,
    "ALE/Asterix-v5": 210,
    "ALE/BankHeist-v5": 14,
    "ALE/BattleZone-v5": 2360,
    "ALE/Boxing-v5": 0,
    "ALE/Breakout-v5": 2,
    "ALE/ChopperCommand-v5": 811,
    "ALE/CrazyClimber-v5": 10780,
    "ALE/DemonAttack-v5": 152,
    "ALE/Freeway-v5": 0,
    "ALE/Frostbite-v5": 65,
    "ALE/Gopher-v5": 258,
    "ALE/Hero-v5": 1027,
    "ALE/Jamesbond-v5": 29,
    "ALE/Kangaroo-v5": 52,
    "ALE/Krull-v5": 1598,
    "ALE/KungFuMaster-v5": 258,
    "ALE/MsPacman-v5": 307,
    "ALE/Pong-v5": -21,
    "ALE/PrivateEye-v5": 25,
    "ALE/Qbert-v5": 164,
    "ALE/RoadRunner-v5": 12,
    "ALE/Seaquest-v5": 68,
    "ALE/UpNDown-v5": 533,
}


def _normalize(raw: float, game: str) -> float:
    h = _HUMAN.get(game)
    r = _RANDOM.get(game)
    if h is None or r is None or (h - r) == 0:
        return raw  # unknown game — pass through
    return (raw - r) / (h - r)


ATARI100K = BenchmarkSpec(
    name="atari100k",
    games=GAMES,
    budget_frames=400_000,
    score_metric="eval/score_mean_last10pct",
    prepare=_normalize,
)
