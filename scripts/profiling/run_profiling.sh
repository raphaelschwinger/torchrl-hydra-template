#!/usr/bin/env bash
# Per-phase wall-clock profile. See scripts/measured_sweep.sh for GPU policy.
#
#   GPU=2 ./scripts/profiling/run_profiling.sh
#   GPU=2 ./scripts/profiling/run_profiling.sh runs_filter=[bbf]
#   GPU=2 ./scripts/profiling/run_profiling.sh quick=true
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/../measured_sweep.sh" \
  scripts/profiling/task.py study=profiling "$@"
