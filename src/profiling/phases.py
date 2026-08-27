"""The phase taxonomy the profile figure is drawn from.

One vocabulary, shared by the trainer that records the phases, the task that
aggregates them and the figure script that draws them, so a bucket cannot be
renamed in one place and silently vanish from another.

Each phase maps onto one of the three optimisation axes of the paper's
Section 3, except evaluation -- which is accounted separately because at a
100k-step budget the 100 final episodes are not free -- and the residual.

Importing this module must stay free of torch, torchrl and the vendored
template: the figure script runs in CI, on a machine with no GPU and no RL
stack installed.
"""

from __future__ import annotations

ROOT = "run"

ENVIRONMENT = "environment"
DATA = "data"
NETWORK = "network"
EVALUATION = "evaluation"
OVERHEAD = "overhead"

#: Phase -> optimisation axis of `sec:optimisations`.
PHASE_AXIS: dict[str, str] = {
    "env_step": ENVIRONMENT,
    "rollout_inference": NETWORK,
    "replay_insert": DATA,
    "replay_sample": DATA,
    "replay_writeback": DATA,
    "gradient_step": NETWORK,
    "eval_periodic": EVALUATION,
    "eval_final": EVALUATION,
    "other": OVERHEAD,
}

#: Fixed draw order. The legend must not reshuffle between rebuilds, or the
#: committed PDF churns for reasons that have nothing to do with the numbers.
#: The order follows the pipeline: collect, move, learn, measure, overhead.
PHASE_ORDER: tuple[str, ...] = (
    "env_step",
    "rollout_inference",
    "replay_insert",
    "replay_sample",
    "replay_writeback",
    "gradient_step",
    "eval_periodic",
    "eval_final",
    "other",
)

#: Human-readable labels for the figure.
PHASE_LABELS: dict[str, str] = {
    "env_step": "Environment step",
    "rollout_inference": "Rollout inference",
    "replay_insert": "Replay insert",
    "replay_sample": "Replay sample + transfer",
    "replay_writeback": "Replay write-back",
    "gradient_step": "Gradient step",
    "eval_periodic": "Evaluation (periodic)",
    "eval_final": "Evaluation (final)",
    "other": "Other",
}

#: Phases that open a region containing nested `env_step` / `rollout_inference`
#: leaves. Everything under one of these collapses into it, so evaluation is
#: reported as one cost rather than smeared across the training buckets.
CONTAINER_PHASES: frozenset[str] = frozenset({"eval_periodic", "eval_final"})

RESIDUAL_PHASE = "other"


def collapse(phase_path: str) -> str:
    """Map a recorded phase path onto the bucket the figure reports.

    ``run/collect/env_step``      -> ``env_step``
    ``run/eval_final/env_step``   -> ``eval_final``
    ``run/eval_final``            -> ``eval_final``
    """
    parts = [p for p in phase_path.split("/") if p]
    if parts and parts[0] == ROOT:
        parts = parts[1:]
    for part in parts:
        if part in CONTAINER_PHASES:
            return part
    return parts[-1] if parts else RESIDUAL_PHASE


def axis(phase: str) -> str:
    """Optimisation axis for a collapsed phase; unknown phases are overhead."""
    return PHASE_AXIS.get(phase, OVERHEAD)
