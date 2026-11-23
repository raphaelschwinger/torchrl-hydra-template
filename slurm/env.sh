# Source me from a Slurm job, from inside the extracted code tree:
#
#   cd "$CODE" && source slurm/env.sh
#
# Gives the job a ready Python env in seconds instead of a fresh `uv sync` into
# the job-private /tmp (5-11 min, ~2 GB of wheels, wiped at job end).
#
# Everything lives on the node-local, persistent per-user cache the cluster
# provides at /scratch/node-cache/$UID (mode 700):
#
#   uv-cache/          uv's wheel cache
#   uv-python/         uv-managed interpreters
#   venvs/<hash>/      one venv per (uv.lock, .python-version), shared by every
#                      job on this node that pins the same lock
#   venvs/<hash>.ok    stamp written after a complete build
#   inductor/, triton/ torch.compile caches (only the first Dreamer job per node
#                      pays the cold max-autotune compile)
#
# The cache and the venvs sit on the same disk, so UV_LINK_MODE=hardlink makes
# installing from the cache a hardlink per file rather than a copy.
#
# The venv holds the dependencies only (`--no-install-project`); the project is
# put on PYTHONPATH from the job's own extracted commit, so jobs at different
# commits but with the same lock share one venv.

set -euo pipefail

NODE_CACHE="${NODE_CACHE:-/scratch/node-cache/$(id -u)}"
if [[ ! -d "$NODE_CACHE" || ! -w "$NODE_CACHE" ]]; then
  echo "env.sh: $NODE_CACHE missing or not writable; falling back to \$TMPDIR" >&2
  NODE_CACHE="${TMPDIR:-/tmp}/node-cache"
fi
mkdir -p "$NODE_CACHE"/{uv-cache,uv-python,venvs,inductor,triton}

export UV_CACHE_DIR="$NODE_CACHE/uv-cache"
export UV_PYTHON_INSTALL_DIR="$NODE_CACHE/uv-python"
export UV_LINK_MODE=hardlink
export UV_PYTHON_PREFERENCE=only-managed
export UV_NO_PROGRESS=1

LOCK_HASH="$(cat uv.lock .python-version 2>/dev/null | sha256sum | cut -c1-12)"
VENV="$NODE_CACHE/venvs/$LOCK_HASH"
export UV_PROJECT_ENVIRONMENT="$VENV"

UV="${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}"

t0=$SECONDS
(
  # Concurrent jobs on this node wait here while the first one builds.
  flock -x 9
  if [[ -f "$VENV.ok" ]]; then
    echo "env.sh: reusing venv $VENV"
  else
    # No stamp means no build or a half-finished one (preempted mid-sync):
    # start clean rather than trust whatever is there.
    echo "env.sh: building venv $VENV"
    rm -rf "$VENV"
    "$UV" sync --frozen --no-install-project
    touch "$VENV.ok"
  fi
) 9>"$VENV.lock"
echo "env.sh: env ready in $((SECONDS - t0))s"

export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

# Compile caches: a pure cache, so results are unaffected.
export TORCHINDUCTOR_CACHE_DIR="$NODE_CACHE/inductor"
export TRITON_CACHE_DIR="$NODE_CACHE/triton"

set +u
