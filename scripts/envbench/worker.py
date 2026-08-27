"""Run exactly one envbench cell and print its result as JSON."""

from __future__ import annotations

import argparse
import json
import sys

from src.envbench.spec import RESULT_MARKER, CellResult, CellSpec
from src.envbench.timing import run_cell
from src.utils.seeding import seed_everything


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, help="JSON-encoded CellSpec")
    parser.add_argument("--warmup-seconds", type=float, default=2.0)
    parser.add_argument("--measure-seconds", type=float, default=10.0)
    parser.add_argument("--min-iters", type=int, default=20)
    parser.add_argument("--max-iters", type=int, default=100_000)
    args = parser.parse_args(argv)

    try:
        spec = CellSpec.from_json(json.loads(args.spec))
    except Exception as exc:
        result = CellResult(status="failed", error=f"bad spec: {type(exc).__name__}: {exc}")
        print(RESULT_MARKER + json.dumps(result.to_json()), flush=True)
        return 0

    seed_everything(spec.seed)
    result = run_cell(
        spec,
        warmup_seconds=args.warmup_seconds,
        measure_seconds=args.measure_seconds,
        min_iters=args.min_iters,
        max_iters=args.max_iters,
    )
    print(RESULT_MARKER + json.dumps(result.to_json()), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
