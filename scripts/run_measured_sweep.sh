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
#   GPU=2 ./scripts/run_measured_sweep.sh --sweep scripts/sweeps/dreamer_speedup.yaml
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
#   --tag NAME      W&B tag for this sweep            (default: measured)
#   --online        log to W&B live instead of offline (see below)
#   --no-sync       keep the offline runs local; do not upload at the end
#   --dry-run       print the commands and exit
#
# Environment:
#   GPU        physical GPU index to pin to           (default 0)
#   THREADS    OMP/MKL thread cap                     (default 32, see below)
#   FORCE=1    run even if the GPU is not idle
#   MEASURED_DIR   output directory                     (default logs/measured)
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
RUN_TAG="${TAG:-measured}"

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
if [[ "$used" != "nan" ]]; then
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
cpus="$(nproc 2>/dev/null || echo 0)"
if [[ "$cpus" -gt 0 ]] && (( ${load_now%.*} > cpus )); then
  echo "WARNING: load average ${load_now} on ${cpus} CPUs — the host is busy." >&2
  echo "         Timings will include other people's work. loadavg_start is" >&2
  echo "         recorded per cell so you can check this after the fact." >&2
fi

# --- W&B off the training thread ----------------------------------------------
# Offline by default: `wandb.log` hands off to a separate process rather than
# blocking, so the cost is small, but Dreamer's video logging is not — and a
# sweep whose point is the timing should not have to argue about it. The runs
# still reach W&B: they are synced at the end of the sweep, which uploads them
# with their original timestamps, so nothing is lost by staying offline while
# the clock is running. `--online` opts back in when you want live curves more
# than you want a clean number; `--no-sync` keeps them local.
if ((ONLINE)); then
  MODE_ARGS=""
else
  MODE_ARGS="logger.0.mode=offline"
fi

# ------------------------------------------------------------------- job list
mkdir -p "$MEASURED_DIR"/{done,logs,runs}
JOBS_TSV="$MEASURED_DIR/jobs.tsv"
TIMINGS="$MEASURED_DIR/timings.tsv"

PYTHON=".venv/bin/python"
[[ -x "$PYTHON" ]] || PYTHON="python3"

SWEEP_ARGS=()
for f in "${SWEEPS[@]}"; do SWEEP_ARGS+=(--sweep "$f"); done
"$PYTHON" scripts/jobs.py "${SWEEP_ARGS[@]}" \
  ${ONLY:+--only "$ONLY"} ${SEEDS:+--seeds "$SEEDS"} > "$JOBS_TSV" || exit 1

build_cmd() {  # name seed overrides
  local name=$1 seed=$2 overrides=$3
  echo "python src/train.py $overrides" \
       "trainer.seed=$seed" \
       "trainer.accelerator=gpu" \
       "trainer.devices=[0]" \
       "logger.0.tags=[$RUN_TAG]" \
       $MODE_ARGS \
       "hydra.run.dir=$MEASURED_DIR/runs/${name}-seed${seed}" \
       "paths.output_dir=$MEASURED_DIR/runs/${name}-seed${seed}"
}

if ((DRY_RUN)); then
  echo "GPU $GPU: ${used} MiB used, ${util}% utilisation | ${THREADS} threads | tag=$RUN_TAG"
  while IFS=$'\t' read -r name seed overrides; do
    echo
    echo "# [$name seed=$seed]"
    echo "CUDA_VISIBLE_DEVICES=$GPU $(build_cmd "$name" "$seed" "$overrides")"
  done < "$JOBS_TSV"
  exit 0
fi

gpu_name="$(nvidia-smi --id="$GPU" --query-gpu=name --format=csv,noheader 2>/dev/null || echo unknown)"
echo "sweep: $(wc -l < "$JOBS_TSV") cells on GPU $GPU ($gpu_name), ${THREADS} threads, tag=$RUN_TAG"
echo "GPU $GPU: ${used} MiB used, ${util}% utilisation"
((ONLINE == 0)) && echo "W&B: offline — uploaded automatically when the sweep finishes"

if [[ ! -s "$TIMINGS" ]]; then
  printf 'name\tseed\twall_seconds\tstatus\tgpu_mem_start_mib\tgpu_util_start_pct\tloadavg_start\tphysical_gpu\tgpu_name\n' > "$TIMINGS"
fi

trap 'echo; echo "interrupted"; exit 130' INT TERM

started=$(date +%s)
total=0; ok=0
while IFS=$'\t' read -r name seed overrides; do
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

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$name" "$seed" "$wall" "$status" "$mem_start" "$util_start" \
    "$load_start" "$GPU" "$gpu_name" >> "$TIMINGS"

  if [[ "$status" == "ok" ]]; then
    printf '%-6s %s  %s s\n' "OK" "$cell" "$wall"
  else
    printf '%-6s %s  %s s  -> %s\n' "FAILED" "$cell" "$wall" "$log"
    tail -n 15 "$log" | sed 's/^/  | /'
  fi
done < "$JOBS_TSV"

# ----------------------------------------------------------------- summary
echo
echo "=== summary (elapsed $(( ($(date +%s) - started) / 60 ))m) ==="
echo "  $ok/$total complete"
echo "  timings: $TIMINGS"

# --- upload the offline runs --------------------------------------------------
# After the clock has stopped, so the upload cannot land in any cell's timing.
# Each run directory is synced separately: `wandb sync` on a directory that has
# already been uploaded is a no-op, which is what makes this safe to rerun after
# a resumed sweep.
if ((ONLINE == 0 && NO_SYNC == 0)); then
  echo
  if ! command -v wandb >/dev/null 2>&1 && [[ ! -x .venv/bin/wandb ]]; then
    echo "  wandb CLI not found — sync manually with:"
    echo "    wandb sync $MEASURED_DIR/runs/*/wandb/offline-run-*"
  else
    WANDB=".venv/bin/wandb"
    [[ -x "$WANDB" ]] || WANDB="wandb"
    # `set -f` is on, so expand the glob explicitly rather than relying on it.
    set +f
    offline_runs=("$MEASURED_DIR"/runs/*/wandb/offline-run-*)
    set -f
    if [[ ! -e "${offline_runs[0]}" ]]; then
      echo "  no offline runs to sync under $MEASURED_DIR/runs/"
    else
      echo "  syncing ${#offline_runs[@]} run(s) to W&B..."
      for run in "${offline_runs[@]}"; do
        "$WANDB" sync "$run" || echo "  sync failed: $run" >&2
      done
    fi
  fi
fi
[[ $ok -eq $total ]] || exit 1
