#!/usr/bin/env bash
# Environment throughput sweep.
#
#   ./scripts/envbench/run_envbench.sh quick=true
#   ./scripts/envbench/run_envbench.sh providers_filter=[gym_sync]
set -euo pipefail
exec "$(dirname "${BASH_SOURCE[0]}")/../measured_sweep.sh" \
  scripts/envbench/task.py study=envthroughput "$@"
