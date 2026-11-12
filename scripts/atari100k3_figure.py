#!/usr/bin/env python3
"""Atari100k-3 test and validation figures for the README, rliable-style.

The layout of the FastRL paper's `fig:results` (paper/figures/
atari100k3_results.{yml,py}): final scores of BBF (RR8, RR2), DreamerV3 and PPO,
5 seeds each, human-normalised, on

  * test -- the three Atari100k-3 games (Assault, Bank Heist, Road Runner);
  * val  -- the two validation games (Breakout, H.E.R.O.);

see the paper's app:subsets. Left: four aggregates over the set's games
(Median, IQM, Mean, Optimality Gap) with rliable's stratified-bootstrap 95% CIs
(Agarwal et al., 2021), which resample runs within each game. Right: per-game
mean HNS with the same intervals.

    # fetch from W&B, write docs/figures/{atari100k3,atari100k_val}.{csv,png}
    uv run --no-project --with wandb --with rliable --with matplotlib --with pandas \\
        python scripts/atari100k3_figure.py

    # redraw from the committed CSVs, no W&B
    uv run --no-project --with rliable --with matplotlib --with pandas \\
        python scripts/atari100k3_figure.py --from-csv

Runs come from the `test-atari100k` / `val-atari100k` sweeps
(scripts/sweeps/*_atari100k{,_val}.yaml), selected by `exp_name`, with the
paper's exclusions. Every arm must have exactly `SEEDS` finished runs per game,
or the script refuses to draw.
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FormatStrFormatter, MaxNLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "docs" / "figures"
ENTITY, PROJECT = "LatentLab", "torchrl-hydra-template"
SCORE_KEY = "eval/final_return_mean"
SEEDS = 5

# HNS = 100 * (score - random) / (human - random), with the Atari-5 reference
# scores (Aitchison et al., 2023), as in the paper.
SETS = {
    "test": {
        "out": "atari100k3",
        "header": "Aggregate over the three Atari100k-3 test games",
        "filters": {
            "tags": {"$all": ["test-atari100k"], "$nin": ["dreamer-stock", "bankheist-rerun"]},
            # an earlier launch of the BBF/Dreamer sweep (2026-09-28/29) is superseded
            "created_at": {"$gte": "2026-10-01T00:00:00Z"},
        },
        "suffix": "_test",
        "games": [
            {"label": "Assault", "env_id": "Assault-v5", "random": 222.39, "human": 742.0},
            {"label": "Bank Heist", "env_id": "BankHeist-v5", "random": 14.2, "human": 753.1},
            {"label": "Road Runner", "env_id": "RoadRunner-v5", "random": 11.5,
             "human": 7845.0},
        ],
    },
    "val": {
        "out": "atari100k_val",
        "header": "Aggregate over the two validation games",
        "filters": {
            "tags": {"$all": ["val-atari100k"]},
            "name": {"$nin": ["kawsctio"]},  # excluded in the paper's val figure too
        },
        "suffix": "_val",
        "games": [
            {"label": "Breakout", "env_id": "Breakout-v5", "random": 1.7, "human": 30.5},
            {"label": "H.E.R.O.", "env_id": "Hero-v5", "random": 1027.0, "human": 30826.4},
        ],
    },
}

# Top to bottom; exp_name is `<prefix><set suffix>`, e.g. bbf_rr8_test / bbf_rr8_val.
# Colours are the FastRL paper's palette (paperkit.plotting.COLORS).
ARMS = [
    {"label": "BBF (RR8)", "prefix": "bbf_rr8", "color": "#6c8ebf"},
    {"label": "BBF (RR2)", "prefix": "bbf", "color": "#6c8ebf"},
    {"label": "DreamerV3", "prefix": "dreamer", "color": "#d79b00"},
    {"label": "PPO", "prefix": "ppo", "color": "#82b366"},
]
METRICS = ["Median", "IQM", "Mean", "Optimality Gap"]
BOOTSTRAP = {"reps": 50_000, "ci": 0.95, "seed": 0}


def fetch(spec: dict) -> pd.DataFrame:
    """One row per (exp_name, env_id, seed) with the final score."""
    import wandb

    api = wandb.Api(timeout=120)
    filters = {
        "state": "finished",
        "config.trainer.total_frames": 100000,
        "config.exp_name": {"$in": [a["prefix"] + spec["suffix"] for a in ARMS]},
        **spec["filters"],
    }
    rows = [
        {
            "run_id": r.id,
            "exp_name": r.config.get("exp_name"),
            "env_id": r.config.get("env_id"),
            "seed": r.config.get("seed"),
            "score": r.summary.get(SCORE_KEY),
        }
        for r in api.runs(f"{ENTITY}/{PROJECT}", filters=filters)
    ]
    df = pd.DataFrame(rows, columns=["run_id", "exp_name", "env_id", "seed", "score"])
    return df.sort_values(["exp_name", "env_id", "seed"]).reset_index(drop=True)


def run_matrices(df: pd.DataFrame, spec: dict) -> dict[str, np.ndarray]:
    """{arm label: (seeds, games) HNS-% matrix}."""
    games = spec["games"]
    random = np.array([g["random"] for g in games])
    human = np.array([g["human"] for g in games])
    out = {}
    for arm in ARMS:
        exp_name = arm["prefix"] + spec["suffix"]
        cols = []
        for g in games:
            sel = df[(df.exp_name == exp_name) & (df.env_id == g["env_id"])]
            values = sel.sort_values("seed")["score"].to_numpy(dtype=float)
            if len(values) != SEEDS or np.isnan(values).any():
                raise SystemExit(
                    f"{exp_name}: {len(values)} finished run(s) with a score on "
                    f"{g['env_id']}, expected {SEEDS}. Are the sweeps done?"
                )
            cols.append(values)
        out[arm["label"]] = 100.0 * (np.stack(cols, axis=1) - random) / (human - random)
    return out


def estimate(matrix: np.ndarray, fn) -> tuple[float, tuple[float, float]]:
    """Point estimate and 95% CI via rliable's stratified bootstrap."""
    from rliable import library as rly

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="random_state is deprecated",
                                category=FutureWarning)
        # rliable's StratifiedBootstrap draws from numpy's global RNG and ignores
        # `random_state`; seed it so every render gives the same intervals.
        np.random.seed(BOOTSTRAP["seed"])  # noqa: NPY002
        _, cis = rly.get_interval_estimates(
            {"arm": matrix}, lambda m: np.array([fn(m)]), reps=BOOTSTRAP["reps"],
            confidence_interval_size=BOOTSTRAP["ci"],
            random_state=np.random.RandomState(BOOTSTRAP["seed"]),
        )
    return float(fn(matrix)), (float(cis["arm"][0, 0]), float(cis["arm"][1, 0]))


def draw(runs: dict[str, np.ndarray], spec: dict) -> plt.Figure:
    from rliable import metrics

    games = spec["games"]
    aggregates = {
        "Median": metrics.aggregate_median,
        "IQM": metrics.aggregate_iqm,
        "Mean": metrics.aggregate_mean,
        # HNS is in percent, so human level -- the optimality-gap threshold -- is 100.
        "Optimality Gap": lambda m: metrics.aggregate_optimality_gap(m, gamma=100.0),
    }
    panels = [(m, aggregates[m]) for m in METRICS] + [
        (g["label"], lambda m, i=i: float(np.mean(m[:, i]))) for i, g in enumerate(games)
    ]

    plt.rcParams.update({"font.family": "sans-serif", "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e5e5e5", "grid.linewidth": 0.6})
    n = len(panels)
    width_in, height_in = 1.5 + 1.2 * n, 2.35
    fig = plt.figure(figsize=(width_in, height_in), layout="none")
    inch_y, inch_x = 1.0 / height_in, 1.0 / width_in
    left_edge, right_edge = 1.05 * inch_x, 1.0 - 0.1 * inch_x
    bottom, top = 0.5 * inch_y, 1.0 - 0.75 * inch_y
    gap, group_gap = 0.12 * inch_x, 0.4 * inch_x
    w = (right_edge - left_edge - (n - 2) * gap - group_gap) / n
    lefts, x = [], left_edge
    for i in range(n):
        lefts.append(x)
        x += w + (group_gap if i == len(METRICS) - 1 else gap)
    ys = np.arange(len(ARMS))[::-1]

    first = None
    for index, ((title, fn), x0) in enumerate(zip(panels, lefts, strict=True)):
        ax = fig.add_axes((x0, bottom, w, top - bottom), sharey=first)
        first = first or ax
        for y, arm in zip(ys, ARMS, strict=True):
            point, ci = estimate(runs[arm["label"]], fn)
            ax.hlines(y, ci[0], ci[1], color=arm["color"], linewidth=5, alpha=0.35, zorder=2)
            ax.plot(point, y, marker="s", markersize=5, color=arm["color"],
                    linestyle="none", zorder=3)
        ax.set_title(title.replace(" ", "\n") if len(title) > 10 else title, fontsize=9)
        ax.grid(axis="y", visible=False)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=9)
        ax.tick_params(axis="x", labelsize=8)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3, min_n_ticks=2))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.margins(x=0.15)
        if index == 0:
            ax.set_yticks(ys)
            ax.set_yticklabels([a["label"] for a in ARMS])
        else:
            plt.setp(ax.get_yticklabels(), visible=False)
        ax.set_ylim(ys.min() - 0.6, ys.max() + 0.6)

    split = len(METRICS)
    for text, x0, x1 in [
        (spec["header"], lefts[0], lefts[split - 1] + w),
        ("Per game (mean over seeds)", lefts[split], lefts[-1] + w),
    ]:
        fig.text((x0 + x1) / 2, 1.0 - 0.05 * inch_y, text, ha="center", va="top",
                 fontsize=9, color="#444444")
        rule = 1.0 - 0.28 * inch_y
        fig.add_artist(Line2D([x0, x1], [rule, rule], color="#999999", linewidth=0.6))

    fig.text((left_edge + right_edge) / 2, 0.06 * inch_y,
             f"Human-normalised score (%) – {SEEDS} seeds, 95% stratified-bootstrap CI",
             ha="center", va="bottom", fontsize=9)
    return fig


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--set", choices=[*SETS, "all"], default="all")
    p.add_argument("--from-csv", action="store_true",
                   help="redraw from docs/figures/<name>.csv instead of fetching")
    args = p.parse_args()

    for name in SETS if args.set == "all" else [args.set]:
        spec = SETS[name]
        out = FIG_DIR / spec["out"]
        if args.from_csv:
            df = pd.read_csv(out.with_suffix(".csv"))
        else:
            df = fetch(spec)
            out.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(out.with_suffix(".csv"), index=False)
            print(f"wrote {out.with_suffix('.csv')} ({len(df)} runs)")
        fig = draw(run_matrices(df, spec), spec)
        fig.savefig(out.with_suffix(".png"), dpi=200, facecolor="white")
        plt.close(fig)
        print(f"wrote {out.with_suffix('.png')}")


if __name__ == "__main__":
    main()
