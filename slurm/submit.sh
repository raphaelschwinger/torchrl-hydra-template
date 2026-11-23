#!/usr/bin/env bash
# Submit one Slurm job per (run, seed) of one or more sweep files.
#
#   slurm/submit.sh                                  # both full Atari-100k sweeps
#   slurm/submit.sh --sweep slurm/sweeps/bbf_atari100k_full.yaml
#   slurm/submit.sh --only bbf_pong,dreamer_pong --seeds 1 --smoke
#   slurm/submit.sh --dry-run
#
# Jobs run the current HEAD commit, which must be pushed (so the W&B commit
# resolves). Cells with a done marker, or with a job of the same name already
# queued or running, are skipped, so re-running this tops up what is missing.
#
# Outputs: $OUT_ROOT/<tag>[/smoke]/{runs,logs,done}/  (default OUT_ROOT=~/runs)

set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"

SWEEPS=()
ONLY="" SEEDS="" SMOKE=0 DRY_RUN=0 GRES="" PARTITION=""
OUT_ROOT="${OUT_ROOT:-$HOME/runs}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sweep)     SWEEPS+=("$2"); shift 2 ;;
    --only)      ONLY="$2"; shift 2 ;;
    --seeds)     SEEDS="$2"; shift 2 ;;
    --smoke)     SMOKE=1; shift ;;
    --dry-run)   DRY_RUN=1; shift ;;
    --gres)      GRES="$2"; shift 2 ;;
    --partition) PARTITION="$2"; shift 2 ;;
    --out)       OUT_ROOT="$2"; shift 2 ;;
    -h|--help)   sed -n '2,14p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 1 ;;
  esac
done
[[ ${#SWEEPS[@]} -eq 0 ]] && SWEEPS=(slurm/sweeps/bbf_atari100k_full.yaml slurm/sweeps/dreamer_atari100k_full.yaml)

PYTHON=python3
SWEEP_ARGS=()
for f in "${SWEEPS[@]}"; do SWEEP_ARGS+=(--sweep "$f"); done
TAG="$("$PYTHON" scripts/jobs.py "${SWEEP_ARGS[@]}" --print-tag)"
[[ -n "$TAG" ]] || { echo "sweep declares no common.tag" >&2; exit 1; }

COMMIT="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "WARNING: uncommitted changes; jobs run $COMMIT without them" >&2
fi
if [[ -z "$(git branch -r --contains "$COMMIT" 2>/dev/null)" ]]; then
  echo "WARNING: $COMMIT is not on any remote branch; push it so W&B can resolve it" >&2
fi

SWEEP_DIR="$OUT_ROOT/$TAG"
JOB_PREFIX=""
TAGS="slurm,$TAG"
if ((SMOKE)); then
  SWEEP_DIR="$SWEEP_DIR/smoke"
  JOB_PREFIX="smoke-"
  TAGS="$TAGS,smoke"
fi
((DRY_RUN)) || mkdir -p "$SWEEP_DIR"/{runs,logs,done}

QUEUED="$(squeue -u "$USER" -h -o '%j' 2>/dev/null || true)"

SBATCH_OPTS=()
[[ -n "$GRES" ]] && SBATCH_OPTS+=(--gres="$GRES")
[[ -n "$PARTITION" ]] && SBATCH_OPTS+=(--partition="$PARTITION")

submitted=0 skipped=0
while IFS=$'\t' read -r name seed overrides; do
  cell="${name}-seed${seed}"
  job="${JOB_PREFIX}${name}-s${seed}"
  if [[ -f "$SWEEP_DIR/done/$cell.done" ]]; then
    echo "SKIP  $job (done)"; skipped=$((skipped + 1)); continue
  fi
  if grep -qxF "$job" <<< "$QUEUED"; then
    echo "SKIP  $job (already queued)"; skipped=$((skipped + 1)); continue
  fi
  extra=()
  ((SMOKE)) && extra+=(logger.0.mode=offline)
  # shellcheck disable=SC2206  # overrides are space-separated Hydra args
  args=($overrides "${extra[@]}")
  cmd=(sbatch --parsable --job-name="$job" --output="$SWEEP_DIR/logs/$cell.log"
       "${SBATCH_OPTS[@]}" slurm/train.sbatch
       "$REPO" "$COMMIT" "$SWEEP_DIR" "$name" "$seed" "$TAGS" "${args[@]}")
  if ((DRY_RUN)); then
    printf '%q ' "${cmd[@]}"; echo
  else
    id="$("${cmd[@]}")"
    echo "SUBMIT $job -> $id"
  fi
  submitted=$((submitted + 1))
done < <("$PYTHON" scripts/jobs.py "${SWEEP_ARGS[@]}" \
           ${ONLY:+--only "$ONLY"} ${SEEDS:+--seeds "$SEEDS"} \
           $( ((SMOKE)) && echo --smoke ))

echo "commit=$COMMIT  tag=$TAG  dir=$SWEEP_DIR  $( ((DRY_RUN)) && echo would-submit || echo submitted)=$submitted  skipped=$skipped"
