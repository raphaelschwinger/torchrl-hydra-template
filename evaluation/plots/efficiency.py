"""Sample efficiency curve: IQM vs training frames."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rliable import library as rly
from rliable import metrics, plot_utils

from ..data_sources.base import RunData


def build_efficiency_dict(
    runs: list[RunData],
    eval_steps: list[int],
    prepare,
) -> dict[str, np.ndarray]:
    """Build {algo: (num_runs, num_games, num_checkpoints)} from run histories.

    For each checkpoint in eval_steps, each run's score is the last observed
    score at or before that frame. Runs with no history are skipped.
    """
    grouped: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = (
        defaultdict(lambda: defaultdict(list))
    )
    for run in runs:
        if not run.history:
            continue
        frames = np.array([f for f, _ in run.history])
        scores = np.array([prepare(s, run.env_name) for _, s in run.history])
        grouped[run.algo_label][run.env_name].append((frames, scores))

    if not grouped:
        return {}

    all_games = sorted({g for gd in grouped.values() for g in gd})
    steps = np.array(eval_steps)
    n_steps = len(steps)
    n_games = len(all_games)

    eff_dict: dict[str, np.ndarray] = {}
    for algo, game_data in grouped.items():
        max_seeds = max(len(v) for v in game_data.values())
        # rliable shape: (num_runs, num_games, num_checkpoints)
        mat = np.full((max_seeds, n_games, n_steps), np.nan)
        for g_idx, game in enumerate(all_games):
            for seed_idx, (frames, scores) in enumerate(game_data.get(game, [])):
                for s_idx, step in enumerate(steps):
                    mask = frames <= step
                    if mask.any():
                        mat[seed_idx, g_idx, s_idx] = scores[mask][-1]
        eff_dict[algo] = mat

    return eff_dict


def plot(
    eff_dict: dict[str, np.ndarray],
    eval_steps: list[int],
    out_dir: Path,
    *,
    reps: int = 2_000,
) -> None:
    """Save sample_efficiency.png to out_dir.

    Args:
        eff_dict: {algo: (num_runs, num_games, num_checkpoints)} from build_efficiency_dict.
        eval_steps: frame checkpoints matching the last axis of eff_dict arrays.
        out_dir: directory to save the PNG.
        reps: bootstrap repetitions.
    """
    if not eff_dict:
        print("Skipping sample_efficiency: no history data.")
        return

    steps = np.array(eval_steps)

    iqm_func = lambda scores: np.array(
        [metrics.aggregate_iqm(scores[..., i]) for i in range(scores.shape[-1])]
    )

    iqm_scores, iqm_cis = rly.get_interval_estimates(eff_dict, iqm_func, reps=reps)

    fig, ax = plt.subplots(figsize=(8, 5))
    plot_utils.plot_sample_efficiency_curve(
        steps,
        iqm_scores,
        iqm_cis,
        algorithms=list(eff_dict.keys()),
        xlabel="Environment Frames",
        ylabel="IQM (normalized score)",
        ax=ax,
    )
    ax.set_title("Sample Efficiency — IQM with 95% Stratified Bootstrap CI", fontsize=13)

    path = out_dir / "sample_efficiency.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
