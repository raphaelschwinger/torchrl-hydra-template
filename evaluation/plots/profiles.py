"""Performance profiles (τ-curves)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from rliable import library as rly
from rliable import plot_utils


def plot(
    score_dict: dict[str, np.ndarray],
    out_dir: Path,
    *,
    reps: int = 50_000,
    tau_max: float = 2.0,
    n_tau: int = 201,
) -> None:
    """Save performance_profiles.png to out_dir.

    Args:
        score_dict: {algo: (num_runs, num_games)} prepared score matrix.
        out_dir: directory to save the PNG.
        reps: bootstrap repetitions (10k is enough for curve shape).
        tau_max: upper bound of τ. Atari100k: 2.0; Atari200M: 8.0.
        n_tau: grid points — 201 gives a smooth curve.
    """
    thresholds = np.linspace(0.0, tau_max, n_tau)

    profiles, profile_cis = rly.create_performance_profile(
        score_dict, thresholds, reps=reps
    )

    colors = dict(zip(score_dict.keys(), sns.color_palette("colorblind")))

    fig, ax = plt.subplots(figsize=(7, 5))
    plot_utils.plot_performance_profiles(
        profiles,
        thresholds,
        performance_profile_cis=profile_cis,
        colors=colors,
        xlabel="",
        ax=ax,
    )
    ax.set_title("Performance Profile — 95% Stratified Bootstrap CI", fontsize=13)
    fig.subplots_adjust(bottom=0.12)
    fig.text(0.5, 0.02, r"Human Normalized Score $(\tau)$", ha="center", va="bottom", fontsize=11)

    path = out_dir / "performance_profiles.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
