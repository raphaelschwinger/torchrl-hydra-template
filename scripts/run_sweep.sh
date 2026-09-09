#!/usr/bin/env bash
#
# Benchmark sweep: 6 experiments x 3 seeds, load-balanced across GPUs.
#
# Workers pull from a shared queue rather than getting a fixed slice, because
# the jobs differ in cost by more than an order of magnitude (BBF's 100k steps
# at replay-ratio 2 against PPO's 100k steps at 8 envs). A static split would
# leave one GPU idle for hours.
#
# Resumable: every finished run drops a marker in $SWEEP_DIR/done/, and reruns
# skip it. Kill the script, restart it, and it picks up where it left off.
#
# The jobs live in scripts/sweeps/*.yaml, not in this file, and are read by
# scripts/jobs.py -- which scripts/run_measured_sweep.sh reads too, so a job defined
# once can be run either way. `--sweep` repeats to combine several files.
#
#   ./scripts/run_sweep.sh --dry-run           # print the 18 commands
#   ./scripts/run_sweep.sh --smoke             # tiny budgets, validates specs
#   ./scripts/run_sweep.sh                     # the real sweep on GPUs 2,3
#   ./scripts/run_sweep.sh --only bbf,tdmpc2   # subset by job name
#   ./scripts/run_sweep.sh --gpus 0,1,2,3      # more workers
#   ./scripts/run_sweep.sh --tag template-v2   # extra W&B tag for this run
#   ./scripts/run_sweep.sh --sweep a.yaml --sweep b.yaml
#
set -uo pipefail
# Overrides are deliberately word-split into argv, but they contain bracket
# syntax (`trainer.devices=[0]`, `logger.0.tags=[template]`) that bash would
# treat as a character-class glob. Disable globbing rather than quote, since
# quoting would collapse each override string into a single argument.
set -f

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

GPUS="2,3"
# Empty, not "1,2,3": the sweep file's own `seeds:` is the default, and only an
# explicit --seeds should override it.
SEEDS=""
ONLY=""
DRY_RUN=0
SMOKE=0
TAG=""
SWEEPS=()
SWEEP_DIR="${SWEEP_DIR:-logs/sweeps}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)     GPUS="$2"; shift 2 ;;
    --seeds)    SEEDS="$2"; shift 2 ;;
    --only)     ONLY="$2"; shift 2 ;;
    --tag)      TAG="$2"; shift 2 ;;
    --sweep)    SWEEPS+=("$2"); shift 2 ;;
    --dry-run)  DRY_RUN=1; shift ;;
    --smoke)    SMOKE=1; SWEEP_DIR="${SWEEP_DIR}/smoke"; shift ;;
    -h|--help)  sed -n '2,25p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

[[ ${#SWEEPS[@]} -eq 0 ]] && SWEEPS=("scripts/sweeps/benchmarks.yaml")

if [[ $SMOKE -eq 1 ]]; then
  EXTRA_ARGS="logger.0.mode=offline"
else
  EXTRA_ARGS=""
fi

# ------------------------------------------------------------------- queue
mkdir -p "$SWEEP_DIR"/{done,logs,runs}
QUEUE="$SWEEP_DIR/queue.txt"
JOBS_TSV="$SWEEP_DIR/jobs.tsv"
: > "$QUEUE"

IFS=',' read -ra GPU_LIST <<< "$GPUS"

PYTHON=".venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="python3"

# The full expansion, before the done/ filter -- written once and used for both
# the queue and the end-of-run summary, so the two can never disagree about
# which runs this sweep was supposed to cover.
SWEEP_ARGS=()
for f in "${SWEEPS[@]}"; do SWEEP_ARGS+=(--sweep "$f"); done
"$PYTHON" scripts/jobs.py "${SWEEP_ARGS[@]}" \
  ${ONLY:+--only "$ONLY"} ${SEEDS:+--seeds "$SEEDS"} \
  $( ((SMOKE)) && echo --smoke ) > "$JOBS_TSV" || exit 1

# Tags accumulate: `parallel` says which runner produced the run, the sweep's
# `common.tag` says which sweep it belongs to, `--tag` is this invocation's own.
# Smoke cells take neither the sweep's tag nor `--tag`'s place in a selector:
# they get `smoke` instead, so a stray one can never reach a results table.
add_tag() {
  local t=$1
  [[ -z "$t" ]] && return 0
  case ",$RUN_TAGS," in *",$t,"*) return 0 ;; esac
  RUN_TAGS="${RUN_TAGS:+$RUN_TAGS,}$t"
}
RUN_TAGS=""
add_tag "parallel"
if ((SMOKE)); then
  add_tag "smoke"
else
  add_tag "$("$PYTHON" scripts/jobs.py "${SWEEP_ARGS[@]}" --print-tag)"
fi
add_tag "$TAG"

queued=0
skipped=0
while IFS=$'\t' read -r name seed overrides; do
  if [[ -f "$SWEEP_DIR/done/${name}-seed${seed}.done" ]]; then
    skipped=$((skipped + 1))
    continue
  fi
  printf '%s\t%s\t%s\n' "$name" "$seed" "$overrides" >> "$QUEUE"
  queued=$((queued + 1))
done < "$JOBS_TSV"

echo "queued=$queued  already-done=$skipped  gpus=${GPU_LIST[*]}  sweeps=${SWEEPS[*]}  dir=$SWEEP_DIR"
[[ $SMOKE -eq 1 ]] && echo "MODE: smoke (tiny budgets — results are meaningless, specs are not)"
[[ $queued -eq 0 ]] && { echo "nothing to do"; exit 0; }

build_cmd() {  # name seed overrides gpu
  local name=$1 seed=$2 overrides=$3
  echo "python src/train.py $overrides" \
       "trainer.seed=$seed" \
       "trainer.devices=[0]" \
       "logger.0.tags=[$RUN_TAGS]" \
       $EXTRA_ARGS \
       "hydra.run.dir=$SWEEP_DIR/runs/${name}-seed${seed}" \
       "paths.output_dir=$SWEEP_DIR/runs/${name}-seed${seed}"
}

if [[ $DRY_RUN -eq 1 ]]; then
  i=0
  while IFS=$'\t' read -r name seed overrides; do
    gpu="${GPU_LIST[$((i % ${#GPU_LIST[@]}))]}"
    echo
    echo "# [$name seed=$seed] -> GPU $gpu (actual GPU is assigned at run time by whichever worker is free)"
    echo "CUDA_VISIBLE_DEVICES=$gpu $(build_cmd "$name" "$seed" "$overrides")"
    i=$((i + 1))
  done < "$QUEUE"
  exit 0
fi

# ------------------------------------------------------------------ workers
pop_job() {
  (
    flock 9
    local line
    line=$(head -n 1 "$QUEUE" 2>/dev/null)
    if [[ -n "$line" ]]; then
      sed -i '1d' "$QUEUE"
      printf '%s' "$line"
    fi
  ) 9>"$QUEUE.lock"
}

worker() {
  local gpu=$1
  while :; do
    local job
    job="$(pop_job)"
    [[ -z "$job" ]] && break

    local name seed overrides
    IFS=$'\t' read -r name seed overrides <<< "$job"

    local tag="${name}-seed${seed}"
    local log="$SWEEP_DIR/logs/${tag}.log"
    echo "[gpu $gpu] START  $tag  ($(date +%H:%M:%S))"

    local start=$SECONDS
    # `trainer.devices=[0]` plus CUDA_VISIBLE_DEVICES: the process sees exactly
    # one GPU, so device index 0 always means the right card and nothing can
    # leak onto a neighbour's.
    if CUDA_VISIBLE_DEVICES="$gpu" $(build_cmd "$name" "$seed" "$overrides") > "$log" 2>&1; then
      touch "$SWEEP_DIR/done/${tag}.done"
      echo "[gpu $gpu] OK     $tag  ($(( (SECONDS - start) / 60 ))m)"
    else
      echo "[gpu $gpu] FAILED $tag  ($(( (SECONDS - start) / 60 ))m)  -> $log"
      tail -n 15 "$log" | sed "s/^/[gpu $gpu]   | /"
    fi
  done
  echo "[gpu $gpu] drained"
}

trap 'echo; echo "interrupted — killing workers"; kill 0; exit 130' INT TERM

started=$(date +%s)
for gpu in "${GPU_LIST[@]}"; do
  worker "$gpu" &
done
wait

# ----------------------------------------------------------------- summary
echo
echo "=== summary (elapsed $(( ($(date +%s) - started) / 60 ))m) ==="
total=0; ok=0
while IFS=$'\t' read -r name seed _; do
  total=$((total + 1))
  if [[ -f "$SWEEP_DIR/done/${name}-seed${seed}.done" ]]; then
    ok=$((ok + 1))
  else
    echo "  MISSING  ${name}-seed${seed}  -> $SWEEP_DIR/logs/${name}-seed${seed}.log"
  fi
done < "$JOBS_TSV"
echo "  $ok/$total complete"
[[ $ok -eq $total ]] || exit 1
