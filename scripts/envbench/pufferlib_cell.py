#!/usr/bin/env python3
"""Run one PufferLib throughput cell, standalone.

PufferLib pins `numpy<2`, `gymnasium<=0.29.1` and `gym<=0.23`, against this
repository's `numpy>=2` and `gymnasium 1.3`. It therefore cannot share an
environment with the `research` package, and cannot import the normal worker.
This script imports nothing from `research`: it speaks the same JSON protocol
over stdout and is launched by the `command:` escape hatch in
`code/configs/study/envthroughput.yaml`.

Keep it dependency-free beyond PufferLib itself and the standard library, and
keep the printed contract identical to `research.envbench.spec.CellResult`.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import time

RESULT_MARKER = "ENVBENCH_RESULT "

# Matched to research.envbench.atari where PufferLib can express it.
#
# It cannot express the observation geometry: its Atari path applies an integer
# `ResizeObservation(downscale=2)` to the raw 210x160 frame, giving 105x80, and
# does no max-pooling over the skipped frames. There is no knob for 84x84. So
# this provider is measured at its own contract (`atari-puffer-105x80`, recorded
# on every row) and is NOT directly comparable with the 84x84 providers: 105x80
# is 8400 pixels per frame against 84x84's 7056, so it carries ~19% more
# observation data per step, and it skips the max-pool. Frame-skip, stack depth
# and sticky actions are matched, so the gap is confined to the resize.
FRAME_SKIP = 4
STACK = 4
STICKY = 0.25
ACTION_POOL = 64


def blank(status: str, error: str = "") -> dict:
    return {
        "status": status,
        "sps": float("nan"),
        "fps": float("nan"),
        "iters": 0,
        "wall_s": float("nan"),
        "setup_s": float("nan"),
        "loadavg": float("nan"),
        "cpu_count": 0,
        "gpu_name": "",
        "provider_version": "",
        "obs_signature": "",
        "sim_seconds_per_agent_step": 0.0,
        "error": error[:200],
    }


def build(spec: dict):
    """Return (vecenv, n_actions, obs_array). Raises with a clear message."""
    import pufferlib.vector
    from pufferlib.environments import atari

    creator = atari.env_creator("pong")
    kwargs = {
        "framestack": STACK,
        "frameskip": FRAME_SKIP,
        "repeat_action_probability": STICKY,
        "obs_type": "grayscale",
    }
    backend = (
        pufferlib.vector.Multiprocessing
        if spec["options"].get("multiprocessing", True)
        else pufferlib.vector.Serial
    )
    vecenv = pufferlib.vector.make(
        creator,
        env_kwargs=kwargs,
        num_envs=spec["num_envs"],
        backend=backend,
    )
    obs, _ = vecenv.reset(seed=spec["seed"])
    return vecenv, vecenv.single_action_space.n, obs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True)
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--measure-seconds", type=float, default=10.0)
    parser.add_argument("--min-iters", type=int, default=20)
    parser.add_argument("--max-iters", type=int, default=100_000)
    args = parser.parse_args()

    spec = json.loads(args.spec)
    result = blank("failed")

    cpu_count = len(os.sched_getaffinity(0))
    loadavg = float("nan")
    with contextlib.suppress(OSError):
        loadavg = os.getloadavg()[0]
    result["cpu_count"] = cpu_count
    result["loadavg"] = loadavg

    vecenv = None
    try:
        from importlib.metadata import version

        import numpy as np

        result["provider_version"] = f"pufferlib {version('pufferlib')}"

        start = time.perf_counter()
        vecenv, n_actions, obs = build(spec)
        result["setup_s"] = time.perf_counter() - start

        obs = np.asarray(obs)
        per_env = obs.shape[1:]
        result["obs_signature"] = f"{obs.dtype}({','.join(str(int(d)) for d in per_env)})"

        rng = np.random.default_rng(spec["seed"])
        actions = rng.integers(0, n_actions, size=(ACTION_POOL, spec["num_envs"]), dtype=np.int32)

        perf = time.perf_counter
        index = 0
        begin = perf()
        while perf() - begin < args.warmup_seconds or index < 2:
            vecenv.step(actions[index % ACTION_POOL])
            index += 1

        begin = perf()
        iters = 0
        while True:
            vecenv.step(actions[index % ACTION_POOL])
            index += 1
            iters += 1
            elapsed = perf() - begin
            if elapsed >= args.measure_seconds and iters >= args.min_iters:
                break

        result["iters"] = iters
        result["wall_s"] = elapsed
        result["sps"] = spec["num_envs"] * iters / elapsed
        result["fps"] = result["sps"] * spec["action_repeat"]
        result["status"] = "ok"
    except ImportError as exc:
        # Preserve the machine facts gathered before the failure: a row that
        # says only "it broke" is less useful than one that says where.
        result = {**result, **blank("unavailable", f"{type(exc).__name__}: {exc}")}
        result["cpu_count"] = cpu_count
        result["loadavg"] = loadavg
    except Exception as exc:
        # PufferLib refuses more workers than physical cores by policy
        # ("num_workers (N) > hardware cores"). That is a provider constraint,
        # not a crash, so it is reported as `unsupported` like any other
        # configuration a provider declines to express.
        declined = type(exc).__name__ == "APIUsageError" or "hardware cores" in str(exc)
        status = "unsupported" if declined else "failed"
        result = {**result, **blank(status, f"{type(exc).__name__}: {exc}")}
        result["cpu_count"] = cpu_count
        result["loadavg"] = loadavg
    finally:
        if vecenv is not None:
            with contextlib.suppress(Exception):
                vecenv.close()

    print(RESULT_MARKER + json.dumps(result), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
