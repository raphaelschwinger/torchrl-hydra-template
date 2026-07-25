"""Probability of improvement P(base > other) bar chart."""
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
    reps: int = 2_000,
) -> None:
    """Save probability_of_improvement.png to out_dir.

    Compares the first algorithm in score_dict against all others.
    Skipped silently when score_dict has fewer than two algorithms.

    Args:
        score_dict: {algo: (num_runs, num_games)} prepared score matrix.
        out_dir: directory to save the PNG.
        reps: bootstrap repetitions.
    """
    algos = list(score_dict.keys())
    if len(algos) < 2:
        print("Skipping probability_of_improvement: need at least 2 algorithms.")
        return

    base = algos[0]
    pairs: dict[str, tuple] = {}
    for other in algos[1:]:
        sa, sb = score_dict[base], score_dict[other]
        n_games = min(sa.shape[1], sb.shape[1])
        pairs[f"{base},{other}"] = (sa[:, :n_games], sb[:, :n_games])

    probs, prob_cis = rly.get_interval_estimates(
        pairs, metrics.probability_of_improvement, reps=reps
    )

    ax = plot_utils.plot_probability_of_improvement(probs, prob_cis)
    ax.set_title(f"P({base} > other) — 95% Bootstrap CI", fontsize=11)

    fig = ax.get_figure()
    path = out_dir / "probability_of_improvement.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
