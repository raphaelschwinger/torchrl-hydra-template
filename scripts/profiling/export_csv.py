#!/usr/bin/env python3
"""Export profiling parquet to paperkit-compatible CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.paths import repo_root, results_dir


def default_export_path() -> Path:
    path = repo_root() / "logs" / "profiling" / "export" / "profile_breakdown.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=results_dir("profiling") / "profile.parquet",
        help="aggregated profiling table (default: logs/profiling/results/profile.parquet)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="destination CSV (default: logs/profiling/export/profile_breakdown.csv)",
    )
    args = parser.parse_args()
    output = args.output or default_export_path()

    frame = pd.read_parquet(args.input)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"wrote {output} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
