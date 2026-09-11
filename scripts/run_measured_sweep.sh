#!/usr/bin/env bash
#
# Wall-clock sweep: the same jobs as run_sweep.sh, run one at a time on
# one idle GPU, with the timings recorded.
#
# This is the serial twin of scripts/run_sweep.sh. That script exists to
# get many runs *finished*, so it drives several GPUs at once; this one exists
# to measure how *long* a run takes, which the same parallelism would destroy —
# two cells on one host contend for CPU, page cache and PCIe even on separate
# cards. Both read the same scripts/sweeps/*.yaml through scripts/jobs.py,
# so a job is defined once and can be run either way.
#
#   GPU=2 ./scripts/run_measured_sweep.sh --sweep scripts/sweeps/dreamer_optimisations_ablation.yaml
#   GPU=2 ./scripts/run_measured_sweep.sh --only tf32 --seeds 1
#   GPU=2 ./scripts/run_measured_sweep.sh --dry-run
#
# Resumable, like run_sweep.sh: finished cells drop a marker in
# $MEASURED_DIR/done/ and are skipped on a rerun.
#
# Timings land in $MEASURED_DIR/timings.tsv, one line per cell, alongside the card
# and host state at that cell's start — the wrapper's idle check below runs once
# at the beginning, so those columns are what tells you the machine was still
# quiet at hour six.
#
# Options:
#   --sweep FILE    one sweep YAML from scripts/sweeps/; repeat to combine
#   --only SUBSTR   comma-separated substrings of job names
#   --seeds LIST    comma-separated seeds, overriding the sweep files
#   --tag NAME      extra W&B tag; added alongside "measured" and the
#                   sweep file's own `common.tag`
#   --online        log to W&B live instead of offline (see below)
#   --no-sync       keep the offline runs local; never upload
#   --dry-run       print the commands and exit
#
# Environment:
#   GPU        physical GPU index to pin to           (default 0)
#   THREADS    OMP/MKL thread cap                     (default 32, see below)
#   FORCE=1    run even if the GPU is not idle
#   MEASURED_DIR   output directory                     (default logs/measured;
#                  the sweep's tag is appended as a subdirectory)
#
set -uo pipefail
# Overrides are word-split into argv and contain bracket syntax
# (`trainer.devices=[0]`) that bash would treat as a character-class glob.
set -f

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

GPU="${GPU:-0}"
# A thread cap rather than none, so the recorded load average stays meaningful:
# uncapped, a many-core box runs a thread per core and "was the machine quiet?"
# stops being answerable from the number. Raise it if a cell is CPU-bound.
THREADS="${THREADS:-32}"
MEASURED_DIR="${MEASURED_DIR:-logs/measured}"

ONLY=""
SEEDS=""
TAG=""
ONLINE=0
NO_SYNC=0
DRY_RUN=0
SWEEPS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sweep)    SWEEPS+=("$2"); shift 2 ;;
    --only)     ONLY="$2"; shift 2 ;;
    --seeds)    SEEDS="$2"; shift 2 ;;
    --tag)      TAG="$2"; shift 2 ;;
    --online)   ONLINE=1; shift ;;
    --no-sync)  NO_SYNC=1; shift ;;
    --dry-run)  DRY_RUN=1; shift ;;
    -h|--help)  sed -n '2,39p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

[[ ${#SWEEPS[@]} -eq 0 ]] && SWEEPS=("scripts/sweeps/benchmarks.yaml")

# --- one card, and only one ---------------------------------------------------
# CUDA_VISIBLE_DEVICES rather than `trainer.devices=[N]` alone: with a single
# card visible, a stray `.cuda()` anywhere in the stack cannot land on a
# neighbour's GPU. Inside the process that card is index 0, which is why every
# cell still passes `trainer.devices=[0]`; the physical index is recorded in
# timings.tsv instead.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS="$THREADS"
export MKL_NUM_THREADS="$THREADS"
export PYTHONUNBUFFERED=1

gpu_state() {  # -> "<mib> <util>", or "nan nan" without nvidia-smi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --id="$GPU" --query-gpu=memory.used,utilization.gpu \
               --format=csv,noheader,nounits | tr -d ',' | tr '\n' ' '
  else
    echo "nan nan"
  fi
}

# --- refuse to measure wall-clock on a contended card -------------------------
read -r used util < <(gpu_state)
if [[ "$used" != "nan" ]] && ((DRY_RUN == 0)); then
  if [[ "${FORCE:-0}" != "1" ]] && { [[ "$used" -gt 1024 ]] || [[ "$util" -gt 5 ]]; }; then
    echo "GPU $GPU is busy: ${used} MiB used, ${util}% utilisation." >&2
    echo "A wall-clock measurement on a shared card is not a measurement." >&2
    echo "Free the card, pick another GPU=, or set FORCE=1 to override." >&2
    exit 1
  fi
fi

# --- is the host quiet, not just the card? ------------------------------------
# The GPU check above is a gate; this is only a warning, because a busy host is
# often someone else's short job rather than a reason to refuse. But wall-clock
# on a loaded box measures the neighbours: `nproc`-ish load means every cell is
# competing for the CPU that feeds the GPU.
load_now="$(cut -d' ' -f1 /proc/loadavg)"
# `--all` because plain `nproc` honours OMP_NUM_THREADS, which this script has
# already exported — without it the load is compared against the thread cap
# rather than the machine, and a quiet 384-core box warns at every start.
cpus="$(nproc --all 2>/dev/null || echo 0)"
if [[ "$cpus" -gt 0 ]] && (( ${load_now%.*} > cpus )); then
  echo "WARNING: load average ${load_now} on ${cpus} CPUs — the host is busy." >&2
  echo "         Timings will include other people's work. loadavg_start is" >&2
  echo "         recorded per cell so you can check this after the fact." >&2
fi

# --- W&B off the training thread ----------------------------------------------
# Offline by default: `wandb.log` hands off to a separate process rather than
# blocking, so the cost is small, but Dreamer's video logging is not — and a
# sweep whose point is the timing should not have to argue about it. The runs
# still reach W&B: each is synced as soon as its own cell's clock stops, with
# its original timestamps, so a day-long sweep publishes as it goes instead of
# holding everything to the last row. `--online` opts back in when you want live
# curves more than you want a clean number; `--no-sync` keeps them local.
if ((ONLINE)); then
  MODE_ARGS=""
else
  MODE_ARGS="logger.0.mode=offline"
fi

# ------------------------------------------------------------------- job list
PYTHON=".venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="python3"

SWEEP_ARGS=()
for f in "${SWEEPS[@]}"; do SWEEP_ARGS+=(--sweep "$f"); done
SWEEP_TAG="$("$PYTHON" scripts/jobs.py "${SWEEP_ARGS[@]}" --print-tag)"

# One directory per sweep, named by its tag: cells are keyed `<row>-seed<N>`,
# and two sweeps both declaring a `baseline` row would otherwise write the same
# directory and skip each other's cells as already done.
MEASURED_DIR="${MEASURED_DIR}${SWEEP_TAG:+/$SWEEP_TAG}"
mkdir -p "$MEASURED_DIR"/{done,logs,runs}
JOBS_TSV="$MEASURED_DIR/jobs.tsv"
TIMINGS="$MEASURED_DIR/timings.tsv"

"$PYTHON" scripts/jobs.py "${SWEEP_ARGS[@]}" \
  ${ONLY:+--only "$ONLY"} ${SEEDS:+--seeds "$SEEDS"} > "$JOBS_TSV" || exit 1

# Tags accumulate: the runner's own, the sweep's `common.tag`, and `--tag`.
add_tag() {
  local t=$1
  [[ -z "$t" ]] && return 0
  case ",$RUN_TAGS," in *",$t,"*) return 0 ;; esac
  RUN_TAGS="${RUN_TAGS:+$RUN_TAGS,}$t"
}
RUN_TAGS=""
add_tag "measured"
add_tag "$SWEEP_TAG"
add_tag "$TAG"

# Upload whatever offline runs sit under $1. Called after each cell so a sweep
# that runs for a day puts its results on W&B as it goes rather than holding
# them hostage to the last row, and once more at the end to catch stragglers.
# `wandb sync` on an already-uploaded directory is a no-op, which is what makes
# both the repeat call and a resumed sweep safe.
sync_runs() {  # dir
  ((ONLINE == 0 && NO_SYNC == 0)) || return 0
  local wandb_bin=".venv/bin/wandb"
  [[ -x "$wandb_bin" ]] || wandb_bin="$(command -v wandb 2>/dev/null)"
  [[ -n "$wandb_bin" ]] || return 0
  set +f
  local runs=("$1"/wandb/offline-run-*)
  set -f
  [[ -e "${runs[0]}" ]] || return 0
  local run
  for run in "${runs[@]}"; do
    "$wandb_bin" sync "$run" >/dev/null 2>&1 || echo "  sync failed: $run" >&2
  done
}

# `logger.0.save_dir` is pinned to the cell directory alongside the two hydra
# path overrides. Without it the logger falls back to `${paths.log_dir}/wandb/`,
# which is derived from the project root rather than from `paths.output_dir` —
# so every cell's offline run lands in one shared logs/wandb/ and `sync_runs`,
# which globs inside the cell directory, matches nothing and returns quietly.
build_cmd() {  # name seed overrides
  local name=$1 seed=$2 overrides=$3
  echo "python src/train.py $overrides" \
       "trainer.seed=$seed" \
       "trainer.accelerator=gpu" \
       "trainer.devices=[0]" \
       "logger.0.tags=[$RUN_TAGS]" \
       $MODE_ARGS \
       "hydra.run.dir=$MEASURED_DIR/runs/${name}-seed${seed}" \
       "paths.output_dir=$MEASURED_DIR/runs/${name}-seed${seed}" \
       "logger.0.save_dir=$MEASURED_DIR/runs/${name}-seed${seed}"
}

if ((DRY_RUN)); then
  echo "GPU $GPU: ${used} MiB used, ${util}% utilisation | ${THREADS} threads | tags=$RUN_TAGS"
  while IFS=$'\t' read -r name seed overrides; do
    echo
    echo "# [$name seed=$seed]"
    echo "CUDA_VISIBLE_DEVICES=$GPU $(build_cmd "$name" "$seed" "$overrides")"
  done < "$JOBS_TSV"
  exit 0
fi

gpu_name="$(nvidia-smi --id="$GPU" --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "sweep: $(wc -l < "$JOBS_TSV") cells on GPU $GPU ($gpu_name), ${THREADS} threads, tags=$RUN_TAGS"
echo "GPU $GPU: ${used} MiB used, ${util}% utilisation"
((ONLINE == 0)) && echo "W&B: offline — each run uploaded as its cell finishes"

if [[ ! -s "$TIMINGS" ]]; then
  printf 'sweep\tname\tseed\twall_seconds\tstatus\tgpu_mem_start_mib\tgpu_util_start_pct\tloadavg_start\tphysical_gpu\tgpu_name\n' > "$TIMINGS"
fi

trap 'echo; echo "interrupted"; exit 130' INT TERM

started=$(date +%s)
total=0; ok=0

# Snapshot the job list before the first cell start
mapfile -t JOB_LINES < "$JOBS_TSV"

for job_line in "${JOB_LINES[@]}"; do
  IFS=$'\t' read -r name seed overrides <<< "$job_line"
  total=$((total + 1))
  cell="${name}-seed${seed}"
  if [[ -f "$MEASURED_DIR/done/${cell}.done" ]]; then
    echo "SKIP   $cell (already done)"
    ok=$((ok + 1))
    continue
  fi

  # Card and host state at *this* cell's start. The guard above ran once, before
  # the sweep; these columns are how a neighbour arriving at hour six shows up.
  read -r mem_start util_start < <(gpu_state)
  load_start="$(cut -d' ' -f1 /proc/loadavg)"

  log="$MEASURED_DIR/logs/${cell}.log"
  echo "START  $cell  ($(date +%H:%M:%S))  gpu ${mem_start}MiB/${util_start}%  load ${load_start}"

  # Nanoseconds and integer arithmetic rather than `bc`, which is not installed
  # in the devcontainer. Bash integers are 64-bit, so an epoch in ns fits.
  start_ns=$(date +%s%N)
  if $(build_cmd "$name" "$seed" "$overrides") > "$log" 2>&1; then
    status="ok"
    touch "$MEASURED_DIR/done/${cell}.done"
    ok=$((ok + 1))
  else
    status="failed"
  fi
  wall_ms=$(( ($(date +%s%N) - start_ns) / 1000000 ))
  wall="$(( wall_ms / 1000 )).$(( (wall_ms % 1000) / 100 ))"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${SWEEP_TAG:--}" "$name" "$seed" "$wall" "$status" "$mem_start" "$util_start" \
    "$load_start" "$GPU" "$gpu_name" >> "$TIMINGS"

  if [[ "$status" == "ok" ]]; then
    printf '%-6s %s  %s s\n' "OK" "$cell" "$wall"
  else
    printf '%-6s %s  %s s  -> %s\n' "FAILED" "$cell" "$wall" "$log"
    tail -n 15 "$log" | sed 's/^/  | /'
  fi

  # After this cell's clock has stopped and before the next one starts, so the
  # upload is never inside a measurement.
  sync_runs "$MEASURED_DIR/runs/${cell}"
done

# ----------------------------------------------------------------- summary
echo
echo "=== summary (elapsed $(( ($(date +%s) - started) / 60 ))m) ==="
echo "  $ok/$total complete"
echo "  timings: $TIMINGS"

# --- catch any run the per-cell sync missed -----------------------------------
if ((ONLINE == 0 && NO_SYNC == 0)); then
  if [[ ! -x .venv/bin/wandb ]] && ! command -v wandb >/dev/null 2>&1; then
    echo "  wandb CLI not found — sync manually with:"
    echo "    wandb sync $MEASURED_DIR/runs/*/wandb/offline-run-*"
  else
    set +f
    for run_dir in "$MEASURED_DIR"/runs/*/; do
      set -f
      sync_runs "${run_dir%/}"
      set +f
    done
    set -f
    echo "  W&B: synced (per cell, and again just now for stragglers)"
  fi
fi
[[ $ok -eq $total ]] || exit 1
