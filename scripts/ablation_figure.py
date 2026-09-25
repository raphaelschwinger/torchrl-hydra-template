#!/usr/bin/env python3
"""Sequential-ablation figure (score + runtime) built from W&B runs.

Draws the "staircase" ablation layout of Rauch et al. (2025), *Can Masked
Autoencoders Also Listen to Birds?*, Figure 3 — each row adds one modification
to the row above, the bar shows the absolute metric, and a superscript shows the
change against the previous row — with a second panel so a modification's effect
on **score** and on **runtime** are read off the same rows.

Three layers, deliberately separated:

1. ``scripts/ablation/<spec>.yaml`` — hand-written. Declares the sections, the
   row order, the row labels, and per row a W&B query. Nothing here is inferred.
2. ``--fetch`` — resolves those queries against W&B and writes a tidy cache at
   ``scripts/ablation/<spec>.csv``: one line per (arm, game, seed).
3. rendering — reads the CSV only, never the network. Iterating on layout is
   therefore free, and the committed CSV reproduces the figure offline.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = REPO_ROOT / "scripts" / "ablation"
DEFAULT_OUT_DIR = REPO_ROOT / "logs" / "analysis"
PUBLISH_DIR = REPO_ROOT / "docs" / "figures"

# Confidence whiskers sit behind the numbers they qualify, so they are
# drawn lighter than the bars and the text.
CI_ALPHA = 0.45


def rel(path: Path) -> str:
    """Repo-relative path when possible — `--out` may point anywhere."""
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


# --------------------------------------------------------------------- model


@dataclass
class Arm:
    """One row of the figure."""

    label: str
    section: str
    color: str
    filters: dict
    highlight: bool = False
    # One entry per run, filled from the cache at render time. The game and seed
    # are kept alongside the value because a stratified bootstrap resamples
    # *within* each game — collapsing to a flat list too early would throw that
    # structure away.
    samples: list[dict] = field(default_factory=list)

    @property
    def key(self) -> str:
        """Stable identifier used as the CSV join key."""
        return f"{self.section}/{self.label}"

    @property
    def n_runs(self) -> int:
        return len(self.samples)

    def values(self, field_name: str) -> list[float]:
        return [s[field_name] for s in self.samples if s[field_name] is not None]

    def by_game(self, field_name: str) -> dict[str, list[float]]:
        """Values grouped by game, for the stratified resample."""
        grouped: dict[str, list[float]] = {}
        for sample in self.samples:
            if sample[field_name] is not None:
                grouped.setdefault(sample["game"], []).append(sample[field_name])
        return grouped


def load_spec(path: Path) -> dict:
    with path.open() as fh:
        spec = yaml.safe_load(fh)
    for required in ("entity", "project", "sections"):
        if required not in spec:
            sys.exit(f"{path}: missing required key '{required}'")
    return spec


def build_arms(spec: dict) -> list[Arm]:
    # Constraints every row shares (`state: finished`, a project-wide date
    # bound, ...). A row's own keys win, so one row can opt out of a default.
    defaults = spec.get("default_filters", {}) or {}
    arms: list[Arm] = []
    for section in spec["sections"]:
        color = section.get("color", "#e8802a")
        for entry in section.get("arms", []):
            arms.append(
                Arm(
                    label=entry["label"],
                    section=section["name"],
                    color=entry.get("color", color),
                    filters={**defaults, **entry.get("filters", {})},
                    highlight=bool(entry.get("highlight", False)),
                )
            )
    if not arms:
        sys.exit("spec declares no arms")
    return arms


# --------------------------------------------------------------------- fetch


def fetch(spec: dict, arms: list[Arm], cache_path: Path) -> None:
    """Resolve every arm's W&B query and write the tidy cache."""
    import wandb

    api = wandb.Api()
    path = f"{spec['entity']}/{spec['project']}"
    score_metric = spec["score"]["metric"]
    runtime_metric = spec["runtime"]["metric"]
    runtime_aggregate = spec["runtime"].get("aggregate", "summary")

    rows: list[dict] = []
    print(f"fetching from {path}\n")
    for arm in arms:
        if not arm.filters:
            # No constraint means "every run in the project" — never what a row
            # wants, and it would silently fill the cache with unrelated runs.
            print(f"! {arm.label:44s} SKIPPED: empty filters")
            continue
        try:
            runs = list(api.runs(path, filters=arm.filters))
        except Exception as exc:  # noqa: BLE001 - surfaced, not swallowed
            print(f"  {arm.label:44s} QUERY FAILED: {exc}")
            continue

        kept, skipped = 0, 0
        for run in runs:
            # A crashed run still reports `finished` if the trainer closed the
            # logger in its `finally`, so length is checked at render time via
            # the seed count rather than trusted here.
            score = run.summary.get(score_metric)
            if score is None:
                skipped += 1
                continue
            runtime = _runtime_value(run, runtime_metric, runtime_aggregate)
            rows.append(
                {
                    "arm": arm.key,
                    "label": arm.label,
                    "section": arm.section,
                    "game": run.config.get("env_id", "unknown"),
                    "seed": run.config.get("seed", ""),
                    "run_id": run.id,
                    "state": run.state,
                    "score": float(score),
                    "runtime": "" if runtime is None else float(runtime),
                }
            )
            kept += 1

        games = {r["game"] for r in rows if r["arm"] == arm.key}
        detail = f"{kept} run(s)" + (
            f", {skipped} without {score_metric}" if skipped else ""
        )
        if games:
            detail += f"  [{', '.join(sorted(games))}]"
        marker = "  " if kept else "! "
        print(f"{marker}{arm.label:44s} {detail}")

    if not rows:
        # Every row came back empty. Overwriting here would destroy a cache that
        # is still good -- including a hand-written one, as `_birdmae_check` is.
        print("\nno runs matched any row; cache left untouched")
        return

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "arm",
        "label",
        "section",
        "game",
        "seed",
        "run_id",
        "state",
        "score",
        "runtime",
    ]
    with cache_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\ncached {len(rows)} run(s) -> {rel(cache_path)}")


def _runtime_value(run, metric: str, aggregate: str) -> float | None:
    """Per-run runtime figure: last logged value, or the median of the series."""
    if aggregate == "summary":
        value = run.summary.get(metric)
        return float(value) if value is not None else None

    try:
        history = run.history(keys=[metric], pandas=False, samples=2_000)
    except Exception as exc:  # noqa: BLE001
        print(f"    warning: history for {run.id} unavailable ({exc}); using summary")
        value = run.summary.get(metric)
        return float(value) if value is not None else None

    values = [row[metric] for row in history if row.get(metric) is not None]
    return float(np.median(values)) if values else None


def load_cache(cache_path: Path, arms: list[Arm], spec: dict) -> None:
    """Populate each arm's score/runtime samples from the cache."""
    if not cache_path.exists():
        sys.exit(f"no cache at {rel(cache_path)} — run with --fetch first")

    normalization = spec["score"].get("normalization", "none")
    games = spec["score"].get("games", {})
    by_key = {arm.key: arm for arm in arms}

    with cache_path.open() as fh:
        for row in csv.DictReader(fh):
            arm = by_key.get(row["arm"])
            if arm is None:  # row belongs to an arm since removed from the spec
                continue
            score = float(row["score"])
            if normalization == "human":
                ref = games.get(row["game"])
                if ref is None:
                    sys.exit(
                        f"{row['game']} has no random/human entry under `score.games`"
                    )
                score = 100.0 * (score - ref["random"]) / (ref["human"] - ref["random"])
            arm.samples.append(
                {
                    "game": row["game"],
                    "seed": row["seed"],
                    "score": score,
                    "runtime": float(row["runtime"]) if row["runtime"] else None,
                }
            )


# ----------------------------------------------------------------- aggregate


def _iqm(values: np.ndarray) -> float:
    """Interquartile mean — the mean after trimming 25% from each tail.

    Identical to ``rliable.metrics.aggregate_iqm`` (and ``scipy.stats.trim_mean``
    at 0.25) but without requiring a rectangular runs x games matrix, so an arm
    with a missing seed still aggregates instead of erroring.
    """
    ordered = np.sort(values)
    cut = int(len(ordered) * 0.25)
    trimmed = (
        ordered[cut : len(ordered) - cut] if len(ordered) - 2 * cut > 0 else ordered
    )
    return float(np.mean(trimmed))


REDUCERS = {
    "iqm": _iqm,
    "mean": lambda v: float(np.mean(v)),
    "median": lambda v: float(np.median(v)),
}


def reduce_samples(values: list[float], how: str) -> float | None:
    if not values:
        return None
    reducer = REDUCERS.get(how)
    if reducer is None:
        sys.exit(f"unknown reduce '{how}' (expected one of {sorted(REDUCERS)})")
    return reducer(np.asarray(values, dtype=float))


def interval(
    arm: Arm, field_name: str, how: str, reps: int, seed: int = 0
) -> tuple[float, float] | None:
    """95% CI for an arm's aggregate, by bootstrap over runs.

    Uses ``rliable``'s StratifiedBootstrap when the arm's runs form a complete
    runs x games matrix — the estimator openrlbenchmark uses elsewhere in this
    repo, which resamples seeds *within* each game so one lucky game cannot
    dominate. Ragged coverage (a crashed seed, uneven games) has no rectangular
    matrix to hand it, so it falls back to an ordinary percentile bootstrap over
    the pooled values and says so. With a single game the two coincide: there is
    only one stratum.

    Returns None when there are too few runs for a bootstrap to mean anything.
    """
    grouped = arm.by_game(field_name)
    counts = {len(v) for v in grouped.values()}
    n_total = sum(len(v) for v in grouped.values())
    if n_total < 3:
        return None

    reducer = REDUCERS[how]
    rng = np.random.default_rng(seed)

    if len(counts) == 1 and len(grouped) > 1:
        matrix = np.array([grouped[g] for g in sorted(grouped)]).T  # (runs, games)
        try:
            from rliable import library as rly

            _, cis = rly.get_interval_estimates(
                {"arm": matrix},
                lambda scores: np.array([reducer(scores.flatten())]),
                reps=reps,
            )
            return float(cis["arm"][0, 0]), float(cis["arm"][1, 0])
        except ImportError:
            print("  note: rliable unavailable; using a pooled percentile bootstrap")

    pooled = np.asarray(arm.values(field_name), dtype=float)
    draws = [
        reducer(rng.choice(pooled, size=len(pooled), replace=True)) for _ in range(reps)
    ]
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


# -------------------------------------------------------------- broken axis


def piecewise_scale(compress_below: float, compress_fraction: float, hi: float):
    """Forward/inverse transforms for the reference figure's broken x-axis.

    Everything below ``compress_below`` occupies ``compress_fraction`` of the
    panel width; the rest expands over the remainder. This is what makes a
    sub-point delta visible next to a bar that starts near zero.
    """
    lo = float(compress_below)
    frac = float(compress_fraction)
    span = max(hi - lo, 1e-9)

    def forward(x):
        x = np.asarray(x, dtype=float)
        return np.where(
            x <= lo,
            frac * np.clip(x, 0.0, None) / max(lo, 1e-9),
            frac + (1.0 - frac) * (x - lo) / span,
        )

    def inverse(y):
        y = np.asarray(y, dtype=float)
        return np.where(
            y <= frac,
            y * max(lo, 1e-9) / max(frac, 1e-9),
            lo + (y - frac) * span / max(1.0 - frac, 1e-9),
        )

    return forward, inverse


def _nice_step(span: float, count: int) -> float:
    """Round tick interval — 1, 2, 2.5 or 5 times a power of ten."""
    raw = max(span, 1e-9) / max(count, 1)
    magnitude = 10.0 ** np.floor(np.log10(raw))
    for multiple in (1.0, 2.0, 2.5, 5.0, 10.0):
        if raw <= multiple * magnitude:
            return multiple * magnitude
    return 10.0 * magnitude


def _nice_ticks(lo: float, hi: float, count: int) -> list[float]:
    step = _nice_step(hi - lo, count)
    start = np.ceil(lo / step) * step
    return [float(t) for t in np.arange(start, hi + step * 0.5, step) if lo <= t <= hi]


def axis_ticks(compress_below: float | None, lo: float, hi: float) -> list[float]:
    """Sparse round ticks under the break, dense round ticks above it."""
    if compress_below is None:
        return _nice_ticks(lo, hi, 6)
    below = [t for t in _nice_ticks(0, compress_below, 3) if t > 0]
    above = _nice_ticks(compress_below, hi, 5)
    return sorted(set(round(t, 6) for t in below + above))


# -------------------------------------------------------------------- render


def render(spec: dict, arms: list[Arm], out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_cfg = spec.get("figure", {})
    panels_cfg = fig_cfg.get("panels", {})
    delta_reference = fig_cfg.get("delta_reference", "previous")

    panel_data = {}
    for panel_key, default_reduce in (("score", "iqm"), ("runtime", "median")):
        how = spec[panel_key].get("reduce", default_reduce)
        reps = int(spec[panel_key].get("bootstrap_reps", 2_000))
        want_ci = bool(spec[panel_key].get("confidence_intervals", False))
        panel_data[panel_key] = (
            [reduce_samples(a.values(panel_key), how) for a in arms],
            [interval(a, panel_key, how, reps) if want_ci else None for a in arms],
        )

    sections = _section_spans(arms)
    height = fig_cfg.get("row_height", 0.52) * len(arms) + 0.55 * len(sections) + 2.0
    fig, ax = plt.subplots(figsize=(fig_cfg.get("width", 13.0), height))

    # Rows top to bottom; each section header consumes one slot of padding.
    y_of: dict[int, float] = {}
    header_y: dict[str, float] = {}
    cursor = 0.0
    for name, indices in sections:
        header_y[name] = cursor
        cursor += 1.15
        for idx in indices:
            y_of[idx] = cursor
            cursor += 1.0
        cursor += 0.45
    total = cursor

    # One [0, 1] mapping per half, each with its own break point.
    halves = {}
    for panel_key, sign in (("score", -1.0), ("runtime", +1.0)):
        values, intervals = panel_data[panel_key]
        halves[panel_key] = _half_scale(
            values, intervals, panels_cfg.get(panel_key, {}), sign
        )

    placements = []
    for panel_key in ("score", "runtime"):
        values, intervals = panel_data[panel_key]
        placements += _draw_half(
            ax=ax,
            arms=arms,
            values=values,
            intervals=intervals,
            y_of=y_of,
            half=halves[panel_key],
            cfg=panels_cfg.get(panel_key, {}),
            delta_reference=delta_reference,
            show_missing=fig_cfg.get("show_missing", True),
            label_rows=(panel_key == "score"),
            show_counts=fig_cfg.get("show_seed_counts", True),
            meta_position=fig_cfg.get("meta_position", "inline"),
        )

    ax.set_xlim(-1.14, 1.14)
    ax.set_ylim(total, -0.95)
    ax.set_yticks([])
    for side in ("top", "right", "left", "bottom"):
        # The x-axis is drawn as an arrow instead of a spine.
        ax.spines[side].set_visible(False)

    _draw_spine(ax, sections, y_of, header_y, total, top_y=-0.80)
    _draw_axis(ax, halves, panels_cfg, fig_cfg)

    fig.suptitle(fig_cfg.get("title", ""), fontsize=13, fontweight="bold", y=0.99)
    caption = fig_cfg.get("caption")
    if caption:
        fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=8.5, wrap=True)

    fig.tight_layout(rect=(0, 0.045, 1, 0.97))
    _resolve_overlaps(fig, [(ax, placements)])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(out_path.with_suffix(suffix), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {rel(out_path.with_suffix('.png'))} (+ .pdf)")


def _half_scale(values, intervals, cfg, sign):
    """Map one half's data onto [0, 1], honouring its axis break.

    Returns a dict with the mapping function, the sign that sends it left or
    right of the spine, and the tick positions the bottom axis should carry.
    """
    present = [v for v in values if v is not None]
    if not present:
        return {
            "sign": sign,
            "hi": 1.0,
            "to_x": lambda v: 0.0,
            "ticks": [],
            "empty": True,
        }

    ceiling = max(present + [ci[1] for ci in intervals if ci is not None])
    hi = ceiling * 1.06
    compress = cfg.get("compress_below", None)
    if compress == "auto":
        compress = min(present) * 0.92
    if compress is not None and (compress <= 0 or compress >= hi):
        compress = None

    if compress is None:

        def fraction(v):
            return float(np.clip(v, 0.0, None) / hi)

    else:
        forward, _ = piecewise_scale(compress, cfg.get("compress_fraction", 0.25), hi)

        def fraction(v):
            return float(forward(np.asarray(float(v))))

    return {
        "sign": sign,
        "hi": hi,
        "to_x": lambda v: sign * fraction(v),
        "ticks": [t for t in axis_ticks(compress, 0, hi) if 0 < t <= hi],
        "empty": False,
    }


#: Bird-MAE's row markers, written at the end of a label in the spec: † for a
#: row that changes what the learner computes, + / ↑ / ↓ for new component,
#: parameter up, parameter down.
MARKER_CHARS = "†↑↓+"


def _split_markers(label: str) -> tuple[str, str]:
    """Separate a label's trailing markers from its name.

    `meta_position: spine` writes the name on the score side of the spine and
    the markers on the runtime side, so the two halves meet in a clean column
    instead of the name carrying its punctuation into the bar.
    """
    name = label.rstrip()
    markers: list[str] = []
    while name and name[-1] in MARKER_CHARS:
        markers.append(name[-1])
        name = name[:-1].rstrip()
    return name, " ".join(reversed(markers))


def _draw_half(
    *,
    ax,
    arms,
    values,
    intervals,
    y_of,
    half,
    cfg,
    delta_reference,
    show_missing,
    label_rows,
    show_counts,
    meta_position="inline",
) -> list:
    if half["empty"]:
        return []

    to_x = half["to_x"]
    sign = half["sign"]
    value_fmt = cfg.get("value_format", "{:.2f}")
    delta_fmt = cfg.get("delta_format", "{:+.2f}")
    # Text hugs the spine on the inner side and the bar tip on the outer side;
    # which screen direction that is flips with the half.
    inner_ha = "right" if sign < 0 else "left"
    outer_ha = "left" if sign < 0 else "right"
    pad = 0.012
    # `spine`: the name stays on the score side and its markers and seed count
    # cross to the runtime side, so both meet at the spine instead of the name
    # trailing punctuation into its bar. `inline` keeps them in the label.
    split_meta = meta_position == "spine"

    placements: list[tuple] = []
    last_present: float | None = None
    first_present: float | None = None

    for idx, (arm, value, ci) in enumerate(zip(arms, values, intervals)):
        y = y_of[idx]
        if value is None:
            if show_missing:
                ax.barh(
                    y,
                    sign * 0.985,
                    height=0.82,
                    color="none",
                    edgecolor="#cccccc",
                    linestyle=":",
                    lw=1.0,
                )
                if label_rows:
                    name = (
                        _split_markers(arm.label)[0] if split_meta else arm.label
                    )
                    ax.text(
                        sign * pad,
                        y,
                        f"{name}   (no runs)",
                        va="center",
                        ha=inner_ha,
                        fontsize=9.5,
                        color="#aaaaaa",
                    )
            continue

        x_end = to_x(value)
        ax.barh(y, x_end, height=0.82, color=arm.color, edgecolor="none", zorder=3)
        weight = "bold" if arm.highlight else "normal"

        # The row is named once, on the score half, against the spine.
        label_art = None
        if label_rows:
            # Seed count rides with the name: coverage is uneven across arms and
            # a one-seed row must not read like a three-seed one. Under
            # `meta_position: spine` it rides the other side of the spine
            # instead, drawn by the runtime half below.
            if split_meta:
                text = _split_markers(arm.label)[0]
            else:
                text = f"{arm.label}   n={arm.n_runs}" if show_counts else arm.label
            label_art = ax.text(
                sign * pad,
                y,
                text,
                va="center",
                ha=inner_ha,
                fontsize=9.5,
                fontweight=weight,
                color="#1a1a1a",
                zorder=7,
            )
        if split_meta and not label_rows:
            markers = _split_markers(arm.label)[1]
            meta = " ".join(
                part
                for part in (markers, f"n={arm.n_runs}" if show_counts else "")
                if part
            )
            if meta:
                ax.text(
                    sign * pad,
                    y,
                    meta,
                    va="center",
                    ha=inner_ha,
                    fontsize=8.5,
                    color="#4a4a4a",
                    zorder=7,
                )
        value_art = ax.text(
            x_end - sign * pad,
            y,
            value_fmt.format(value),
            va="center",
            ha=outer_ha,
            fontsize=10,
            fontweight=weight,
            color="#1a1a1a",
            zorder=7,
        )
        if label_art is not None:
            placements.append((label_art, value_art, x_end, sign))

        if ci is not None:
            # Ride the bar's lower half rather than its centre line: with no
            # halo behind the text, a centred whisker strikes through the value.
            y_ci = y + 0.26
            lo_x, hi_x = to_x(ci[0]), to_x(ci[1])
            ax.plot(
                [lo_x, hi_x],
                [y_ci, y_ci],
                color="#1a1a1a",
                lw=1.2,
                alpha=CI_ALPHA,
                solid_capstyle="butt",
                zorder=6,
            )
            for bound in (lo_x, hi_x):
                ax.plot(
                    [bound, bound],
                    [y_ci - 0.11, y_ci + 0.11],
                    color="#1a1a1a",
                    lw=1.2,
                    alpha=CI_ALPHA,
                    zorder=6,
                )

        reference = first_present if delta_reference == "first" else last_present
        if reference is not None and abs(value - reference) > 1e-9:
            ax.text(
                x_end,
                y - 0.52,
                delta_fmt.format(value - reference),
                va="center",
                ha=outer_ha,
                fontsize=8,
                fontweight="bold",
                color="#1a1a1a",
                zorder=4,
            )
        if first_present is None:
            first_present = value
        last_present = value

    return placements


SPINE_COLOR = "#1a1a1a"
SPINE_LW = 1.8


def _arrow(ax, start, end):
    """Sleek open arrowhead terminating an axis, in data coordinates."""
    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=11,
            lw=SPINE_LW,
            color=SPINE_COLOR,
            shrinkA=0,
            shrinkB=0,
            zorder=5,
            clip_on=False,
            joinstyle="miter",
        )
    )


def _draw_spine(ax, sections, y_of, header_y, total, top_y) -> None:
    """One rule down the middle, broken where a section title sits in the gap.

    The gap is only just wider than the title: too much air and the rule stops
    reading as one continuous axis. The top of the rule carries an arrowhead,
    so it terminates the way the x-axis does rather than just stopping.
    """
    gap = 0.28  # half-height of the break each title is written into
    cuts = []
    for name, indices in sections:
        centre = header_y[name] + 0.12
        ax.text(
            0,
            centre,
            name,
            fontsize=10.5,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=8,
            clip_on=False,
        )
        cuts.append((centre - gap, centre + gap))

    # Arrowhead on the first segment; plain segments for the rest.
    _arrow(ax, (0, cuts[0][0]), (0, top_y))
    top = cuts[0][1]
    for lo, hi in cuts[1:]:
        if lo > top:
            ax.plot(
                [0, 0],
                [top, lo],
                color=SPINE_COLOR,
                lw=SPINE_LW,
                solid_capstyle="butt",
                zorder=5,
                clip_on=False,
            )
        top = hi
    # ... and on down to the axis, so the rule terminates on the x-axis itself.
    ax.plot(
        [0, 0],
        [top, total],
        color=SPINE_COLOR,
        lw=SPINE_LW,
        solid_capstyle="butt",
        zorder=5,
        clip_on=False,
    )


def _draw_axis(ax, halves, panels_cfg, fig_cfg) -> None:
    """One bottom axis, ticked in each half's own units."""
    positions, labels = [], []
    for panel_key in ("score", "runtime"):
        half = halves[panel_key]
        for tick in half["ticks"]:
            positions.append(half["to_x"](tick))
            labels.append(_tick_label(tick))
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    # Both halves grow outward from the spine, so the axis gets a head at each
    # end rather than the single origin-to-max head of a normal axis.
    y_axis = ax.get_ylim()[0]
    _arrow(ax, (0, y_axis), (-1.10, y_axis))
    _arrow(ax, (0, y_axis), (1.10, y_axis))

    # Two axis titles, one per half, since the two sides measure different things.
    for panel_key, x in (("score", 0.25), ("runtime", 0.75)):
        ax.text(
            x,
            -0.055,
            panels_cfg.get(panel_key, {}).get("label", ""),
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=10,
        )


def _tick_label(value: float) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:g}"


def _section_spans(arms: list[Arm]) -> list[tuple[str, list[int]]]:
    spans: list[tuple[str, list[int]]] = []
    for idx, arm in enumerate(arms):
        if spans and spans[-1][0] == arm.section:
            spans[-1][1].append(idx)
        else:
            spans.append((arm.section, [idx]))
    return spans


def _resolve_overlaps(fig, placements) -> None:
    """Move a bar's value label outward when the bar is too short to hold both.

    A short bar leaves no room for the row name and its number side by side, and
    they overprint. The name is the part that must stay put, so the number moves
    past the bar tip -- or past the end of a name that already overruns it.
    """
    from matplotlib.transforms import ScaledTranslation

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax, entries in placements:
        for label_art, value_art, x_end, sign in entries:
            label_box = label_art.get_window_extent(renderer=renderer)
            value_box = value_art.get_window_extent(renderer=renderer)
            overlaps = (
                label_box.x0 < value_box.x1 if sign < 0 else label_box.x1 > value_box.x0
            )
            if not overlaps:
                continue
            tip_px = ax.transData.transform((x_end, 0))[0]
            edge_px = label_box.x0 if sign < 0 else label_box.x1
            target_px = min(tip_px, edge_px) if sign < 0 else max(tip_px, edge_px)
            target_x = ax.transData.inverted().transform((target_px, 0))[0]
            offset = ScaledTranslation(sign * 4 / 72, 0, fig.dpi_scale_trans)
            value_art.set_ha("right" if sign < 0 else "left")
            value_art.set_position((target_x, value_art.get_position()[1]))
            value_art.set_transform(ax.transData + offset)


# ----------------------------------------------------------------------- cli


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--spec",
        default="dreamjepa",
        help="spec name under scripts/ablation/ (without .yaml)",
    )
    parser.add_argument(
        "--fetch", action="store_true", help="re-query W&B and refresh the CSV cache"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="output directory for the figure",
    )
    parser.add_argument(
        "--publish", action="store_true", help="also copy the PNG into docs/figures/"
    )
    args = parser.parse_args()

    spec_path = SPEC_DIR / f"{args.spec}.yaml"
    if not spec_path.exists():
        available = ", ".join(sorted(p.stem for p in SPEC_DIR.glob("*.yaml"))) or "none"
        sys.exit(f"no spec at {spec_path} (available: {available})")

    spec = load_spec(spec_path)
    arms = build_arms(spec)
    cache_path = SPEC_DIR / f"{args.spec}.csv"

    if args.fetch:
        fetch(spec, arms, cache_path)

    load_cache(cache_path, arms, spec)
    missing = [a.label for a in arms if not a.samples]
    if missing:
        print(f"\n{len(missing)} row(s) with no runs: {', '.join(missing)}")

    out_path = args.out / f"ablation_{args.spec}"
    render(spec, arms, out_path)

    if args.publish:
        PUBLISH_DIR.mkdir(parents=True, exist_ok=True)
        target = PUBLISH_DIR / f"ablation_{args.spec}.png"
        target.write_bytes(out_path.with_suffix(".png").read_bytes())
        print(f"published {rel(target)}")


if __name__ == "__main__":
    main()
