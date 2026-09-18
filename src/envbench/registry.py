"""Builder registry: `kind` -> callable returning a configured `Provider`.

Importing this module must stay cheap and dependency-free. Every provider's real
imports (gymnasium, envpool, dm_control, jax) live inside the builder bodies, so
that CI — which installs base dependencies only — can import the registry, run
the tests and render the figure without any provider installed.
"""

from __future__ import annotations

from collections.abc import Callable

from src.envbench.atari import (
    build_ale_native,
    build_envpool_atari,
    build_gym_vector,
    build_sb3_vector,
)
from src.envbench.base import Provider
from src.envbench.dmc import (
    build_dmc_native,
    build_envpool_dmc,
    build_gym_shimmy,
    build_mjx_playground,
)
from src.envbench.spec import CellSpec

BUILDERS: dict[str, Callable[[CellSpec], Provider]] = {
    # Atari
    "gym_vector": build_gym_vector,
    "ale_native": build_ale_native,
    "envpool_atari": build_envpool_atari,
    "sb3_vector": build_sb3_vector,
    # DMC
    "dmc_native": build_dmc_native,
    "gym_shimmy": build_gym_shimmy,
    "envpool_dmc": build_envpool_dmc,
    "mjx_playground": build_mjx_playground,
}
