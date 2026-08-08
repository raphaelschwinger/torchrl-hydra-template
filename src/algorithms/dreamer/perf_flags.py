"""Process-wide toggles for the Dreamer speed optimisations.

``configure()`` is called once per ``DreamerV3.__init__`` (before any
submodule is constructed — ``static_pad`` is read at ``nn.Module.__init__``
time) and fully replaces the previous flag state from defaults, so two models
built in the same process (tests, Hydra multirun) never leak flags from one
config into the other.

``static_pad``, ``dedup_value``, ``cudnn_benchmark``, ``tf32`` and
``foreach_laprop`` are exact optimisations — flipping them off reproduces
bit-identical numerics, just slower (``foreach_laprop`` verified by
``tests/test_laprop.py``). ``bf16_autocast`` changes numerics: ``False``
restores r2dreamer's float16-autocast + ``GradScaler`` scheme.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path


@dataclasses.dataclass
class PerfFlags:
    static_pad: bool = True
    dedup_value: bool = True
    cudnn_benchmark: bool = True
    tf32: bool = True
    bf16_autocast: bool = True
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
        overrides = {k: bool(v) for k, v in data.items() if k in known}
        base = dataclasses.replace(base, **overrides)
    flags = base
    return flags


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
