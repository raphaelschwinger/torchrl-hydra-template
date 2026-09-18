"""GPU occupancy helpers for wall-clock measurement studies."""

from __future__ import annotations

import os
import subprocess

IDLE_MEMORY_MIB = 1024
IDLE_UTILISATION_PCT = 5


def physical_index(default: int = 0) -> int | None:
    """Physical GPU index as the driver sees it (not CUDA's remapped index)."""
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible:
        return default
    first = visible.split(",")[0].strip()
    try:
        return int(first)
    except ValueError:
        return None


def occupancy(index: int | None) -> dict[str, float]:
    """Memory in use (MiB) and utilisation (%) on one card, or NaNs."""
    blank = {"gpu_mem_used_mib": float("nan"), "gpu_util_pct": float("nan")}
    if index is None:
        return blank
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                f"--id={int(index)}",
                "--query-gpu=memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return blank
    if not out:
        return blank
    parts = [p.strip() for p in out.splitlines()[0].split(",")]
    try:
        return {"gpu_mem_used_mib": float(parts[0]), "gpu_util_pct": float(parts[1])}
    except (IndexError, ValueError):
        return blank


def is_idle(reading: dict[str, float]) -> bool:
    memory = reading.get("gpu_mem_used_mib", float("nan"))
    util = reading.get("gpu_util_pct", float("nan"))
    if memory == memory and memory > IDLE_MEMORY_MIB:
        return False
    return not (util == util and util > IDLE_UTILISATION_PCT)
