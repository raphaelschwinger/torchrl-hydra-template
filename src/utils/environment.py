"""Machine and toolchain facts for measured studies."""

from __future__ import annotations

import os
import platform
import subprocess
from typing import Any


def usable_cpus() -> int:
    getter = getattr(os, "process_cpu_count", None)
    if getter is not None:
        return int(getter() or 0)
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return int(os.cpu_count() or 0)


def environment_record() -> dict[str, Any]:
    record: dict[str, Any] = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "usable_cpus": usable_cpus(),
    }
    try:
        cpu = subprocess.run(
            ["lscpu"], capture_output=True, text=True, check=False, timeout=30
        ).stdout
        for line in cpu.splitlines():
            if line.startswith("Model name:"):
                record["cpu_model"] = line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        ).stdout.strip()
        if gpu:
            record["gpus"] = gpu.splitlines()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        with open("/proc/meminfo") as handle:
            for line in handle:
                if line.startswith("MemTotal:"):
                    record["mem_total_gib"] = round(int(line.split()[1]) / 1024**2, 1)
                    break
    except OSError:
        pass
    return record
