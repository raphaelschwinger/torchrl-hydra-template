"""Running one cell of a wall-clock study, under supervision."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

from src.utils.gpu import occupancy
from src.utils.paths import repo_root
from src.utils.rss import DEFAULT_INTERVAL, GIB, RssWatchdog


def tail_error(log_path: Path, window: int = 64_000) -> str:
    try:
        with log_path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - window))
            text = handle.read().decode("utf-8", "replace")
    except OSError:
        return ""
    for line in reversed(text.replace("\r", "\n").splitlines()):
        if line.strip():
            return line.strip()
    return ""


def launch_cell(
    command: list[str],
    log_path: Path,
    *,
    timeout_seconds: float,
    rss_limit_gib: float | None,
    gpu_index: int | None,
    poll_interval: float = DEFAULT_INTERVAL,
) -> dict[str, Any]:
    environment = dict(os.environ)
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PROJECT_ROOT"] = str(repo_root())

    before = occupancy(gpu_index)
    limit_bytes = int(float(rss_limit_gib) * GIB) if rss_limit_gib else None
    status, error = "ok", ""

    with log_path.open("w") as handle:
        handle.write(f"$ {' '.join(command)}\n\n")
        handle.write(f"# gpu occupancy before launch: {before}\n\n")
        handle.flush()
        process = subprocess.Popen(
            command,
            stdout=handle,
            stderr=subprocess.STDOUT,
            cwd=repo_root(),
            env=environment,
            start_new_session=True,
        )
        with RssWatchdog(process.pid, limit_bytes, interval=poll_interval) as watchdog:
            try:
                returncode = process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                watchdog.kill()
                returncode = process.wait()
                status = "timeout"
                error = f"cell exceeded {timeout_seconds:.0f}s"
        peak_gib = watchdog.peak_gib
        if watchdog.killed:
            status = "oom"
            error = (
                f"peak RSS {peak_gib:.1f} GiB exceeded the {float(rss_limit_gib):.0f} GiB budget"
            )
        elif status == "ok" and returncode != 0:
            status = "failed"
            error = tail_error(log_path) or f"exit {returncode}"

    return {
        "status": status,
        "error": error[:200],
        "peak_rss_gib": peak_gib,
        "gpu_mem_used_start_mib": before["gpu_mem_used_mib"],
        "gpu_util_start_pct": before["gpu_util_pct"],
    }
