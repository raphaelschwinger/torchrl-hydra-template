"""Process-wide toggles for the Dreamer speed optimisations.

``configure()`` is called once per ``DreamerV3.__init__`` (before any
submodule is constructed — ``static_pad`` is read at ``nn.Module.__init__``
time) and fully replaces the previous flag state from defaults, so two models
built in the same process (tests, Hydra multirun) never leak flags from one
config into the other.

``static_pad``, ``dedup_value``, ``cudnn_benchmark``, ``tf32`` and
``foreach_laprop`` are exact optimisations — flipping them off reproduces
bit-identical numerics, just slower (``foreach_laprop`` verified by
``tests/test_laprop.py``). ``amp`` is the one that changes numerics; it picks
the mixed-precision scheme:

- ``bf16`` — bfloat16 autocast, no gradient scaling (official DreamerV3).
- ``fp16`` — float16 autocast + ``GradScaler`` (r2dreamer parity). fp16's
  5-bit exponent underflows on small gradients, so the loss scaling is
  required, not optional.
- ``off``  — no autocast at all; the model runs in full float32.

``off`` is a plain switch here only because the norm layers keep f32 weights
and upcast internally (``networks.RMSNormF32``). Constructing them in bf16
instead — as this port did before ``d4debb1`` — would couple the norm dtype to
this flag and make ``off`` a multi-file change.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path


#: Accepted values of ``PerfFlags.amp``.
AMP_MODES = ("bf16", "fp16", "off")


@dataclasses.dataclass
class PerfFlags:
    static_pad: bool = True
    dedup_value: bool = True
    cudnn_benchmark: bool = True
    tf32: bool = True
    amp: str = "bf16"
    foreach_laprop: bool = True


# Module-level singleton read by networks.py / rssm.py / model/dreamerv3.py /
# LaProp construction. Mutated only via configure().
flags = PerfFlags()


def configure(cfg=None) -> PerfFlags:
    """Set the global flags from a Hydra/OmegaConf ``perf`` sub-config (or a
    plain dict). Always starts from defaults — a missing key keeps its
    default rather than a previous model's override.
    """
    global flags
    base = PerfFlags()
    if cfg is not None:
        data = dict(cfg)
        known = set(dataclasses.asdict(base))
        overrides = {}
        for key, value in data.items():
            if key not in known:
                continue
            overrides[key] = _parse_amp(value) if key == "amp" else bool(value)
        base = dataclasses.replace(base, **overrides)
    flags = base
    return flags


def _parse_amp(value) -> str:
    """Normalise an ``amp`` override to one of ``AMP_MODES``.

    A bare ``off`` in a YAML *file* is parsed as the boolean ``False`` (YAML
    1.1 reads off/on as bools), while the Hydra CLI passes the string "off"
    through unchanged — so both spellings reach here and both must work.
    ``on``/``true`` is rejected rather than guessed: it does not say whether
    bf16 or fp16 was meant.
    """
    if value is False:
        return "off"
    if value is True:
        raise ValueError(
            "perf.amp must be one of bf16 / fp16 / off, not a bare true/on "
            "(YAML reads those as booleans, and they do not say which "
            "autocast dtype you want)."
        )
    name = str(value).lower()
    if name not in AMP_MODES:
        raise ValueError(f"unknown perf.amp {value!r}; choose from {list(AMP_MODES)}")
    return name


def configure_env() -> None:
    """Set the process-wide env vars Dreamer's torch.compile path wants.

    Must be called before the *first* CUDA call anywhere in the process —
    ``PYTORCH_CUDA_ALLOC_CONF`` is parsed once, lazily, the first time the
    CUDA allocator is touched, and re-setting it later is a silent no-op.
    That first touch is ``seed_everything()``'s ``torch.cuda.manual_seed_all``
    in ``src/train.py``, which runs before any algorithm/model is
    constructed — so this cannot live in ``DreamerV3.__init__`` and must be
    called from the entry point instead, ahead of ``seed_everything()``.

    ``setdefault``: an explicit shell or devcontainer override always wins over this default.

    - ``TORCHINDUCTOR_CACHE_DIR``: persists compiled kernels (esp.
      ``max-autotune``'s ~20-candidate-per-kernel search) across runs and
      container restarts instead of the default ephemeral ``/tmp``.
    - ``PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True``: lets CUDA memory
      segments grow instead of requiring a new contiguous block, reducing
      fragmentation-driven OOMs — the crash mode ``max-autotune``'s varied
      per-candidate scratch-buffer sizes can trigger.

          Deliberately does not touch ``TORCHINDUCTOR_CACHE_DIR`` — left at
    torchrl's own default (``/tmp/torchinductor_<user>``, on this box's local
    overlay filesystem, not the NFS-mounted project dir). It already persists
    fine across ordinary runs within one container lifetime; a `setdefault`
    here would also lose the race against torchrl's own import-time default
    anyway, since importing this function pulls in the full
    ``src.algorithms.dreamer`` package (-> ``dreamer.py`` -> every model file
    -> ``torchrl``) before this function's body ever runs.
    """

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
