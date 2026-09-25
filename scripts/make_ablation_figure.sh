#!/usr/bin/env bash
#
# Sequential-ablation figure (score + runtime) from W&B runs.
#
# Unlike scripts/make_figures.sh — which shells out to openrlbenchmark's `rlops`
# for cross-algorithm comparisons — this one is our own plotting code, because
# an ablation staircase is not something rlops draws

set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

PYTHON=".venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="python3"

case "${1:-}" in
  -h|--help) sed -n '2,36p' "$0"; exit 0 ;;
esac

exec "$PYTHON" scripts/ablation_figure.py "$@"
