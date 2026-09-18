#!/usr/bin/env bash
#
# Run a wall-clock sweep on one named GPU, under a host-memory budget.
#
#   GPU=2 ./scripts/measured_sweep.sh scripts/profiling/task.py study=profiling
#   GPU=2 ./scripts/measured_sweep.sh scripts/envbench/task.py study=envthroughput quick=true
#
set -euo pipefail

if [ $# -lt 1 ]; then
  echo "usage: $0 <task-script> [hydra overrides...]" >&2
  exit 2
fi
TASK="$1"; shift

GPU="${GPU:-0}"
RSS_LIMIT_GIB="${RSS_LIMIT_GIB:-128}"
THREADS="${THREADS:-32}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"

if command -v nvidia-smi >/dev/null 2>&1; then
  read -r used util < <(
    nvidia-smi --id="$GPU" --query-gpu=memory.used,utilization.gpu \
               --format=csv,noheader,nounits | tr -d ','
  )
  if [ "${FORCE:-0}" != "1" ] && { [ "$used" -gt 1024 ] || [ "$util" -gt 5 ]; }; then
    echo "GPU $GPU is busy: ${used} MiB used, ${util}% utilisation." >&2
    echo "A wall-clock measurement on a shared card is not a measurement." >&2
    echo "Free the card, pick another GPU=, or set FORCE=1 to override." >&2
    exit 1
  fi
  echo "GPU $GPU: ${used} MiB used, ${util}% utilisation"
fi

export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export PROJECT_ROOT="$REPO_ROOT"
export PYTHONUNBUFFERED=1
export HYDRA_FULL_ERROR=1

echo "sweep: $TASK on GPU $GPU (visible as 0), RSS ceiling ${RSS_LIMIT_GIB} GiB, ${THREADS} threads${*:+ | overrides: $*}"

exec uv run python -u "$TASK" \
  device_index=0 \
  "physical_device_index=$GPU" \
  "rss_limit_gib=$RSS_LIMIT_GIB" \
  "$@"
