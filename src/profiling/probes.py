"""Where each algorithm's data-pipeline and gradient-step phases attach.

The vendored template is upstream and not ours to instrument, so the profiling
trainer wraps *instance* attributes of the objects the algorithm has already
built. That keeps the instrumentation on our side of the vendor boundary and
survives `make vendor-update`, at the price of naming private methods here --
which is why every entry cites the line it hangs off, and why a missing hook is
recorded rather than raised.

The table is deliberately per algorithm. The environment and the policy are
generic (the trainer finds them through the public `train_env` and
`get_policy()` surface), but "where does this algorithm put its replay traffic
and its gradient step" has no generic answer: BBF hand-writes `_store` /
`_sample` / `_update`, while DreamerV3 delegates to a `Buffer` object and a
world model.

Importing this module pulls in nothing -- it is names, not code.
"""

from __future__ import annotations

#: algorithm class name -> ((dotted attribute path, phase name), ...)
#:
#: The dotted path is resolved against the *algorithm instance*, after
#: `algorithm.setup()` has run, since that is when the buffer and the networks
#: exist.
PROBES: dict[str, tuple[tuple[str, str], ...]] = {
    # src/algorithms/bbf/bbf.py: step() calls _store(), then _sample() and
    # _update() once per gradient step. _sample() carries the host->device copy
    # (`sample.to(self.device)`), so it is the data-pipeline bucket.
    "BBFAlgorithm": (
        ("_store", "replay_insert"),
        ("_sample", "replay_sample"),
        ("_update", "gradient_step"),
    ),
    # src/algorithms/dreamer/dreamer.py: step() calls replay_buffer.add_transition(),
    # then per update replay_buffer.sample() -> model.update() -> replay_buffer.update().
    # Buffer.sample() is where the pinned-memory / non_blocking transfer lives
    # (src/algorithms/dreamer/buffer.py), and Buffer.update() writes the posterior
    # latents back into storage -- buffer traffic, not a gradient.
    "DreamerAlgorithm": (
        ("replay_buffer.add_transition", "replay_insert"),
        ("replay_buffer.sample", "replay_sample"),
        ("replay_buffer.update", "replay_writeback"),
        ("model.update", "gradient_step"),
    ),
}

#: Used when an algorithm has no entry above. Enough to keep the sweep running
#: and to make the gap visible in the table (the gradient bucket comes out
#: empty and the residual absorbs it) rather than to pretend it was measured.
FALLBACK: tuple[tuple[str, str], ...] = (
    ("replay_buffer.extend", "replay_insert"),
    ("replay_buffer.sample", "replay_sample"),
)


def probes_for(algorithm) -> tuple[tuple[str, str], ...]:
    """Probe table for an algorithm instance, by class name."""
    return PROBES.get(type(algorithm).__name__, FALLBACK)


def resolve(root, dotted: str):
    """Split ``a.b.c`` into (the object holding ``c``, ``"c"``).

    Returns ``(None, attr)`` when any link in the chain is missing.
    """
    parts = dotted.split(".")
    obj = root
    for part in parts[:-1]:
        obj = getattr(obj, part, None)
        if obj is None:
            return None, parts[-1]
    return obj, parts[-1]
