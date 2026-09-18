"""The timing loops and the single-cell driver.

Two loops, because CPU and accelerator providers need different accounting:

*   `time_loop` — fixed *time*, not fixed steps. Throughput here spans four
    orders of magnitude (a serial dm_control loop at N=1 against MJX at N=8192),
    so any fixed step budget is either an hour for one provider or milliseconds
    for another.

*   `time_loop_async` — for JAX. Under async dispatch the host runs arbitrarily
    far ahead of the device, so a wall-clock-bounded loop counts *dispatches*,
    not simulated steps, and then blocks for a long time at the end. Instead we
    calibrate the per-step cost with individually blocked steps, compute a fixed
    iteration count, run exactly that many unblocked, and synchronise once. That
    measures sustained pipeline throughput with a single synchronisation, which
    is what a real rollout does.
"""

from __future__ import annotations

import contextlib
import os
import time
from typing import TYPE_CHECKING

from src.envbench.spec import (
    CellResult,
    CellSpec,
    ContractError,
    ProviderUnavailable,
)

if TYPE_CHECKING:
    from src.envbench.base import Provider

# Per-env observation shape each contract requires. Checked before timing: a
# provider that silently hands back a different observation is not doing the
# same work, and the resulting number would be incomparable.
EXPECTED_OBS_SHAPE: dict[str, tuple[int, ...]] = {
    "atari-std": (4, 84, 84),
    "dmc-state": (17,),
}


def usable_cpus() -> int:
    """Cores this process may actually run on.

    `os.process_cpu_count` is 3.13+; the affinity mask is the 3.12 equivalent and
    is what matters inside a container with a restricted CPU set.
    """
    getter = getattr(os, "process_cpu_count", None)
    if getter is not None:
        return int(getter() or 0)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return int(os.cpu_count() or 0)


def time_loop(
    provider: Provider,
    *,
    warmup_seconds: float,
    measure_seconds: float,
    min_iters: int,
) -> tuple[int, float]:
    """Warm up, then step until the measurement window closes."""
    perf = time.perf_counter

    start = perf()
    warm = 0
    while perf() - start < warmup_seconds or warm < 2:
        provider.step()
        warm += 1

    start = perf()
    iters = 0
    while True:
        provider.step()
        iters += 1
        elapsed = perf() - start
        if elapsed >= measure_seconds and iters >= min_iters:
            return iters, elapsed


def time_loop_async(
    provider: Provider,
    *,
    measure_seconds: float,
    min_iters: int,
    max_iters: int,
    calibration_iters: int = 20,
) -> tuple[int, float]:
    """Calibrated fixed-iteration loop with one terminal synchronisation."""
    perf = time.perf_counter

    for _ in range(2):  # absorb tracing/compilation before calibrating
        provider.step_blocking()

    start = perf()
    for _ in range(calibration_iters):
        provider.step_blocking()
    per_step = (perf() - start) / calibration_iters

    planned = round(measure_seconds / per_step) if per_step > 0 else max_iters
    iters = int(min(max(planned, min_iters), max_iters))

    start = perf()
    for _ in range(iters):
        provider.step()
    provider.synchronize()
    return iters, perf() - start


def check_contract(provider: Provider, spec: CellSpec) -> None:
    """Fail loudly when a provider was not configured to the benchmark contract."""
    expected = EXPECTED_OBS_SHAPE.get(spec.obs_contract)
    if expected is not None and tuple(provider.obs_shape) != expected:
        raise ContractError(
            f"{spec.provider}: observation is {provider.obs_shape}, "
            f"contract {spec.obs_contract} requires {expected}"
        )

    if spec.sim_seconds_per_agent_step > 0.0:
        got = provider.sim_seconds_per_agent_step
        if abs(got - spec.sim_seconds_per_agent_step) > 1e-9:
            raise ContractError(
                f"{spec.provider}: advances {got}s of simulated time per agent step, "
                f"contract requires {spec.sim_seconds_per_agent_step}s — the providers "
                f"would be timed on different amounts of physics"
            )


def run_cell(
    spec: CellSpec,
    *,
    warmup_seconds: float,
    measure_seconds: float,
    min_iters: int,
    max_iters: int,
) -> CellResult:
    """Build one provider, verify the contract, measure it, and always return a row."""
    from src.envbench.registry import BUILDERS

    result = CellResult(status="failed")
    result.cpu_count = usable_cpus()
    with contextlib.suppress(OSError):  # getloadavg is not available everywhere
        result.loadavg = os.getloadavg()[0]

    builder = BUILDERS.get(spec.kind)
    if builder is None:
        result.status = "unsupported"
        result.error = f"no builder registered for kind={spec.kind!r}"
        return result

    provider: Provider | None = None
    try:
        start = time.perf_counter()
        provider = builder(spec)
        result.setup_s = time.perf_counter() - start

        result.provider_version = provider.versions
        result.obs_signature = provider.obs_signature
        result.sim_seconds_per_agent_step = provider.sim_seconds_per_agent_step
        result.gpu_name = _gpu_name(provider)

        check_contract(provider, spec)

        if provider.device == "cuda":
            iters, wall = time_loop_async(
                provider,
                measure_seconds=measure_seconds,
                min_iters=min_iters,
                max_iters=max_iters,
            )
        else:
            iters, wall = time_loop(
                provider,
                warmup_seconds=warmup_seconds,
                measure_seconds=measure_seconds,
                min_iters=min_iters,
            )

        result.iters = iters
        result.wall_s = wall
        result.sps = spec.num_envs * iters / wall
        result.fps = result.sps * spec.action_repeat
        result.status = "ok"
    except ProviderUnavailable as exc:
        result.status = "unavailable"
        result.error = f"{type(exc).__name__}: {exc}"
    except ContractError as exc:
        result.status = "unsupported"
        result.error = f"{type(exc).__name__}: {exc}"
    except (ImportError, ModuleNotFoundError) as exc:
        result.status = "unavailable"
        result.error = f"{type(exc).__name__}: {exc}"
    except Exception as exc:  # a cell must never abort the sweep
        result.status = "failed"
        result.error = f"{type(exc).__name__}: {exc}"
    finally:
        if provider is not None:
            # Teardown noise must never mask the measurement it follows.
            with contextlib.suppress(Exception):
                provider.close()

    result.error = result.error[:200]
    return result


def _gpu_name(provider: Provider) -> str:
    if provider.device != "cuda":
        return ""
    try:
        import jax

        return str(jax.devices()[0].device_kind)
    except Exception:
        return "unknown"
