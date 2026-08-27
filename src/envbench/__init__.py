"""Environment-provider throughput measurement.

Measures how fast each practically-available environment provider can be stepped
on Atari and DMC, with random actions and no policy or learner, so that the
number attributable to the *provider* is not confounded with inference or
training cost.

Two things make the numbers comparable, and both are enforced rather than
assumed: every provider is configured to the same observation contract
(`research.envbench.atari`, `research.envbench.dmc`), and every cell is timed
under one stated convention (`research.envbench.timing`).

Importing this package pulls in no environment library; see `registry`.
"""

from src.envbench.base import Provider
from src.envbench.registry import BUILDERS
from src.envbench.spec import (
    RESULT_MARKER,
    STATUSES,
    CellResult,
    CellSpec,
    ContractError,
    ProviderUnavailable,
)
from src.envbench.timing import run_cell

__all__ = [
    "BUILDERS",
    "RESULT_MARKER",
    "STATUSES",
    "CellResult",
    "CellSpec",
    "ContractError",
    "Provider",
    "ProviderUnavailable",
    "run_cell",
]
