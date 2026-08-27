"""Cell specifications, results, and the committed parquet schema.

A *cell* is one measurement: one provider, at one batch size, with one seed. The
cell is the unit of process isolation (see `research.tasks.envbench` for why), so
a spec has to survive a JSON round trip into a fresh interpreter — hence plain
dataclasses of plain types rather than anything Hydra- or provider-specific.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

# Every state a cell can end in. Only `ok` carries numbers; the rest carry a
# reason. A provider that will not install is a row with NaN and an explanation,
# never a silent absence — which provider is *practically available* is part of
# what the table reports.
STATUSES = frozenset({"ok", "unavailable", "unsupported", "failed", "timeout"})

# The marker the worker prints its one result line behind. Providers are chatty
# (ALE banners, JAX warnings, MuJoCo notices), so the driver needs an
# unambiguous handle on the single line that is data.
RESULT_MARKER = "ENVBENCH_RESULT "


@dataclass(frozen=True)
class CellSpec:
    """Everything one measurement needs, and nothing that cannot be JSON-encoded."""

    suite: str  # "atari" | "dmc"
    task: str  # "Pong" | "cheetah_run"
    provider: str  # registry key, e.g. "envpool"
    kind: str  # builder key in research.envbench.registry.BUILDERS
    num_envs: int
    seed: int
    action_repeat: int
    obs_contract: str  # "atari-std" | "dmc-state"
    parallelism: str  # "none" | "process" | "thread" | "gpu-batched"
    device: str  # "cpu" | "cuda"
    # Expected simulated seconds advanced per agent step. The DMC fairness
    # invariant: providers with different action-repeat conventions would
    # otherwise be timed on different amounts of physics. 0.0 disables the check
    # (Atari has no meaningful wall-clock-to-sim-time mapping).
    sim_seconds_per_agent_step: float = 0.0
    options: dict[str, Any] = field(default_factory=dict)

    @property
    def cell_id(self) -> str:
        return f"{self.suite}.{self.provider}.n{self.num_envs}.s{self.seed}"

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> CellSpec:
        return cls(**data)


@dataclass
class CellResult:
    """The outcome of one cell. Always produced, even when everything failed."""

    status: str
    sps: float = float("nan")  # agent steps per second, aggregated over the batch
    fps: float = float("nan")  # emulator frames per second = sps * action_repeat
    iters: int = 0
    wall_s: float = float("nan")  # length of the timed window actually achieved
    setup_s: float = float("nan")  # construction + reset + JIT, excluded from sps
    loadavg: float = float("nan")  # 1-min load average at cell start (interference audit)
    cpu_count: int = 0
    gpu_name: str = ""
    provider_version: str = ""
    obs_signature: str = ""  # e.g. "uint8(4,84,84)" — proves providers agree
    sim_seconds_per_agent_step: float = 0.0
    error: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


class ContractError(RuntimeError):
    """A provider could not be configured to the benchmark's observation contract.

    Raised rather than warned: a silently mismatched contract (EnvPool defaulting
    to no sticky actions, SB3's wrapper adding reward clipping) produces numbers
    that look fine and are not comparable, which is the one failure mode that
    would make the paper wrong rather than merely incomplete.
    """


class ProviderUnavailable(RuntimeError):
    """The provider is not installed, or its backing device is absent."""
