#!/usr/bin/env python3
"""Evaluation CLI — fetch W&B runs and produce rliable plots.

Usage::

    # All finished runs tagged 'template', Atari100k benchmark
    python evaluation/run.py --benchmark atari100k --tag template \\
        --entity LatentLab --out reports/atari100k

    # Specific algorithms and games
    python evaluation/run.py --benchmark atari100k --tag template \\
        --algo DreamerV3 R2Dreamer \\
        --env ALE/Hero-v5 ALE/Pong-v5 \\
        --out reports/hero_pong

    # Include sample efficiency (requires history fetch — slower)
    python evaluation/run.py --benchmark atari100k --tag template \\
        --plots aggregate profiles improvement efficiency \\
        --out reports/full
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from .benchmarks import get_benchmark
from .benchmarks.base import BenchmarkSpec
from .data_sources import RunData, WandbDataSource
from .plots import aggregate, efficiency, improvement, profiles
from .plots.efficiency import build_efficiency_dict

ALL_PLOTS = ["aggregate", "profiles", "improvement", "efficiency"]
DEFAULT_PLOTS = ["aggregate", "profiles", "improvement"]


# ── Score matrix builder ──────────────────────────────────────────────────────


def build_score_dict(
    runs: list[RunData],
    benchmark: BenchmarkSpec,
    games: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Build rliable-ready score matrices from fetched RunData.

    Returns:
        {algo: np.ndarray of shape (num_runs, num_games)}.
        Rows are seeds; columns are games. NaN for missing seeds.
    """
    game_list = games or benchmark.games
    game_set = set(game_list)

    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for run in runs:
        if run.summary_score is None:
            continue
        if run.env_name not in game_set:
            continue
        prepared = benchmark.prepare(run.summary_score, run.env_name)
        grouped[run.algo_label][run.env_name].append(prepared)

    score_dict: dict[str, np.ndarray] = {}
    for algo, game_scores in grouped.items():
        rows: list[list[float]] = []
        for game in game_list:
            scores = game_scores.get(game)
            if scores:
                rows.append(scores)

        if not rows:
            print(f"  Skipping '{algo}': no valid scores.")
            continue

        max_seeds = max(len(r) for r in rows)
        # rliable convention: (num_runs, num_games) — rows=seeds, cols=games
        mat = np.full((max_seeds, len(rows)), np.nan)
        for g_idx, row in enumerate(rows):
            mat[: len(row), g_idx] = row

        n_seeds = int(np.sum(~np.isnan(mat[:, 0])))
        print(f"  {algo}: {len(rows)} game(s), up to {n_seeds} seed(s).")
        score_dict[algo] = mat

    return score_dict


# ── CLI ───────────────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch W&B runs and produce rliable evaluation plots.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--benchmark",
        required=True,
        help="Benchmark name (e.g. atari100k). See evaluation/benchmarks/.",
    )
    p.add_argument(
        "--tag",
        default="",
        help="W&B tag to filter on. Omit or pass '' to fetch all finished runs.",
    )
    p.add_argument(
        "--entity",
        required=True,
        help="W&B entity. Defaults to WANDB_ENTITY env var or login default.",
    )
    p.add_argument("--project", default="torchrl-hydra-template")
    p.add_argument(
        "--algo",
        nargs="+",
        default=None,
        help="Algorithm labels to include (default: all found).",
    )
    p.add_argument(
        "--env",
        nargs="+",
        default=None,
        help="Env IDs to include (default: benchmark's full game list).",
    )
    p.add_argument(
        "--plots",
        nargs="+",
        default=DEFAULT_PLOTS,
        choices=ALL_PLOTS + ["all"],
        help=f"Plots to generate (default: {DEFAULT_PLOTS}).",
    )
    p.add_argument("--out", required=True, help="Output directory for PNGs.")
    p.add_argument(
        "--reps",
        type=int,
        default=50_000,
        help="Bootstrap repetitions for aggregate/profiles/improvement.",
    )
    p.add_argument(
        "--efficiency-reps",
        type=int,
        default=2_000,
        help="Bootstrap repetitions for efficiency curve (fewer = faster).",
    )
    p.add_argument(
        "--tau-max",
        type=float,
        default=2.0,
        help="Upper τ for performance profile x-axis (default: 2.0).",
    )
    p.add_argument(
        "--eval-steps",
        type=int,
        nargs="+",
        default=None,
        help="Frame checkpoints for efficiency x-axis. "
        "Defaults to 8 evenly spaced steps up to budget_frames.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    plots_to_run = ALL_PLOTS if "all" in args.plots else args.plots
    fetch_history = "efficiency" in plots_to_run

    benchmark = get_benchmark(args.benchmark)
    games = args.env or benchmark.games

    source = WandbDataSource(
        tag=args.tag,
        entity=args.entity,
        project=args.project,
    )
    runs = source.fetch_runs(
        algos=args.algo,
        envs=games,
        metric=benchmark.score_metric,
        fetch_history=fetch_history,
        history_metric="episode/score",
    )

    if not runs:
        print("No runs matched — nothing to plot.", file=sys.stderr)
        return 1

    print(f"\nBuilding score matrices for benchmark '{benchmark.name}'...")
    score_dict = build_score_dict(runs, benchmark, games)

    if not score_dict:
        print("No valid score data — aborting.", file=sys.stderr)
        return 1

    n_games = next(iter(score_dict.values())).shape[1]
    if n_games < 3:
        print(
            f"\nNOTE: only {n_games} game(s) — aggregate IQM/gap are less meaningful.\n"
            "      Performance profile and PoI are still valid."
        )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating plots → {out_dir}/")

    if "aggregate" in plots_to_run:
        aggregate.plot(score_dict, out_dir, reps=args.reps)

    if "profiles" in plots_to_run:
        profiles.plot(score_dict, out_dir, reps=args.reps, tau_max=args.tau_max)

    if "improvement" in plots_to_run:
        improvement.plot(score_dict, out_dir, reps=args.reps)

    if "efficiency" in plots_to_run:
        steps = args.eval_steps or list(
            range(0, benchmark.budget_frames + 1, benchmark.budget_frames // 8)
        )
        eff_dict = build_efficiency_dict(runs, steps, benchmark.prepare)
        efficiency.plot(eff_dict, steps, out_dir, reps=args.efficiency_reps)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
