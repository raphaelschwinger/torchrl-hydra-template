#!/usr/bin/env python3
"""Expand sweep files into one TSV line per (job, seed).
`scripts/run_sweep.sh` (parallel, several GPUs, throughput) and
`scripts/run_measured_sweep.sh` (serial, one idle GPU, wall-clock) launch runs for
different reasons but need the same job table. Neither should own it, and bash
cannot read YAML -- so the table lives in `scripts/sweeps/*.yaml` and both
shell out to this.

    scripts/jobs.py --sweep scripts/sweeps/benchmarks.yaml
    scripts/jobs.py --sweep benchmarks.yaml --sweep dreamer_speedup.yaml
    scripts/jobs.py --sweep benchmarks.yaml --only dreamer,bbf --smoke

Output is `name<TAB>seed<TAB>overrides`, which is what both callers already
read with `IFS=$'\t'`.

Override order, and it matters because Hydra takes the *last* value for a key:

    experiment=<job.experiment>   common.overrides   job.overrides
        [--smoke only:]           common.smoke_overrides   job.smoke_overrides

So a job specialises the file's common block, and a smoke override beats the
real budget it is shrinking.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

#: Used when neither the CLI nor the sweep file says otherwise.
DEFAULT_SEEDS = [1, 2, 3]


def _as_list(node, field: str, where: str) -> list[str]:
    """Read a YAML list of override strings, or fail loudly."""
    if node is None:
        return []
    if not isinstance(node, list):
        raise SystemExit(
            f"{where}: `{field}` must be a list, got {type(node).__name__}"
        )
    return [str(item) for item in node]


def load_sweeps(paths: list[Path]) -> dict[str, dict]:
    """Merge several sweep files into one job table, keyed by job name.

    Duplicate names across files are an error rather than a last-one-wins
    merge: two sweep files quietly defining the same job would produce one run
    and two sets of expectations about what it measured.
    """
    jobs: dict[str, dict] = {}
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"no such sweep file: {path}")
        data = yaml.safe_load(path.read_text()) or {}
        runs = data.get("runs") or {}
        if not runs:
            raise SystemExit(f"{path}: no `runs:` block")

        common = data.get("common") or {}
        common_overrides = _as_list(
            common.get("overrides"), "overrides", f"{path}:common"
        )
        common_smoke = _as_list(
            common.get("smoke_overrides"), "smoke_overrides", f"{path}:common"
        )
        common_seeds = common.get("seeds")

        for name, job in runs.items():
            if name in jobs:
                raise SystemExit(
                    f"duplicate job {name!r} in {path} "
                    f"(already defined in {jobs[name]['sweep']})"
                )
            job = job or {}
            where = f"{path}:{name}"
            if "experiment" not in job:
                raise SystemExit(f"{where}: no `experiment:` key")
            seeds = job.get("seeds", common_seeds) or DEFAULT_SEEDS
            jobs[name] = {
                "sweep": path,
                "experiment": str(job["experiment"]),
                "seeds": [int(s) for s in seeds],
                "enabled": bool(job.get("enabled", True)),
                "overrides": common_overrides
                + _as_list(job.get("overrides"), "overrides", where),
                "smoke_overrides": common_smoke
                + _as_list(job.get("smoke_overrides"), "smoke_overrides", where),
            }
    return jobs


def selected(jobs: dict[str, dict], only: str) -> list[str]:
    """Job names to run, in sweep-file order.

    `--only` matches on substring, so `--only dreamer` picks up every Dreamer
    job across every sweep file. Naming a job explicitly also overrides its
    `enabled: false` -- that is how a job stays declared-but-skipped in the
    file (documenting what was left out) while still being one flag away.
    """
    if not only:
        return [name for name, job in jobs.items() if job["enabled"]]
    filters = [f for f in (part.strip() for part in only.split(",")) if f]
    return [name for name in jobs if any(f in name for f in filters)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep",
        action="append",
        type=Path,
        required=True,
        help="one sweep YAML; repeat to combine several files",
    )
    parser.add_argument(
        "--only", default="", help="comma-separated substrings of job names"
    )
    parser.add_argument(
        "--seeds", default="", help="comma-separated seeds, overriding the sweep files"
    )
    parser.add_argument(
        "--smoke", action="store_true", help="append the smoke overrides"
    )
    parser.add_argument(
        "--list", action="store_true", help="print job names only, one per line"
    )
    args = parser.parse_args()

    jobs = load_sweeps(args.sweep)
    names = selected(jobs, args.only)
    if args.only and not names:
        raise SystemExit(f"--only {args.only!r} matched no job in {len(jobs)} declared")

    if args.list:
        print("\n".join(names))
        return

    cli_seeds = (
        [int(s) for s in args.seeds.split(",") if s.strip()] if args.seeds else None
    )

    for name in names:
        job = jobs[name]
        overrides = [f"experiment={job['experiment']}"] + job["overrides"]
        if args.smoke:
            overrides += job["smoke_overrides"]
        for seed in cli_seeds or job["seeds"]:
            print(f"{name}\t{seed}\t{' '.join(overrides)}")


if __name__ == "__main__":
    # `raise SystemExit("message")` already prints to stderr and exits 1, which
    # is exactly the contract both shell callers check with `|| exit 1`.
    main()
