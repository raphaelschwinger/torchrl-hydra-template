#!/usr/bin/env python3
"""Export profiling parquet to the paper submodule CSV shape."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.paths import repo_root, results_dir


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
        default=repo_root() / "paper" / "data" / "figures" / "profile_breakdown.csv",
        help="destination CSV for paperkit",
    )
    args = parser.parse_args()

    frame = pd.read_parquet(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({len(frame)} rows)")


if __name__ == "__main__":
    main()
