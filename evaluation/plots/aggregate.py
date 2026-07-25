"""Aggregate metrics bar chart: IQM, Mean, Median, Optimality Gap."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rliable import library as rly
from rliable import metrics, plot_utils


def plot(
    score_dict: dict[str, np.ndarray],
    out_dir: Path,
    *,
    reps: int = 50_000,
) -> None:
    """Save aggregate_metrics.png to out_dir.

    Args:
        score_dict: {algo: (num_runs, num_games)} prepared score matrix.
        out_dir: directory to save the PNG.
        reps: bootstrap repetitions.
    """
    METRIC_NAMES = ["IQM", "Mean", "Median", "Optimality Gap"]

    def agg_func(scores: np.ndarray) -> np.ndarray:
        return np.array(
            [
                metrics.aggregate_iqm(scores),
                metrics.aggregate_mean(scores),
                metrics.aggregate_median(scores),
                metrics.aggregate_optimality_gap(scores),
            ]
        )

    estimates, cis = rly.get_interval_estimates(score_dict, agg_func, reps=reps)

    algos = list(score_dict.keys())

    fig, axes = plot_utils.plot_interval_estimates(
        estimates,
        cis,
        metric_names=METRIC_NAMES,
        algorithms=algos,
        xlabel="Human Normalized Score",
        xlabel_y_coordinate=-0.16,
    )

    path = out_dir / "aggregate_metrics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
