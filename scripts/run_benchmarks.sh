#!/usr/bin/env bash
#
# Benchmark sweep: 6 experiments x 3 seeds, load-balanced across GPUs.
#
# Workers pull from a shared queue rather than getting a fixed slice, because
# the jobs differ in cost by more than an order of magnitude (BBF's 100k steps
# at replay-ratio 2 against PPO's 100k steps at 8 envs). A static split would
# leave one GPU idle for hours.
#
# Resumable: every finished run drops a marker in $BENCH_DIR/done/, and reruns
# skip it. Kill the script, restart it, and it picks up where it left off.
#
#   ./scripts/run_benchmarks.sh --dry-run           # print the 18 commands
#   ./scripts/run_benchmarks.sh --smoke             # tiny budgets, validates specs
#   ./scripts/run_benchmarks.sh                     # the real sweep on GPUs 2,3
#   ./scripts/run_benchmarks.sh --only bbf,tdmpc2   # subset by job name
#   ./scripts/run_benchmarks.sh --gpus 0,1,2,3      # more workers
#   ./scripts/run_benchmarks.sh --tag template-v2   # own W&B tag for this sweep
#
set -uo pipefail
# Overrides are deliberately word-split into argv, but they contain bracket
# syntax (`trainer.devices=[0]`, `logger.0.tags=[template]`) that bash would
# treat as a character-class glob. Disable globbing rather than quote, since
# quoting would collapse each override string into a single argument.
set -f

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

GPUS="2,3"
SEEDS="1,2,3"
ONLY=""
DRY_RUN=0
SMOKE=0
TAG=""
BENCH_DIR="${BENCH_DIR:-logs/benchmarks}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)     GPUS="$2"; shift 2 ;;
    --seeds)    SEEDS="$2"; shift 2 ;;
    --only)     ONLY="$2"; shift 2 ;;
    --tag)      TAG="$2"; shift 2 ;;
    --dry-run)  DRY_RUN=1; shift ;;
    --smoke)    SMOKE=1; BENCH_DIR="${BENCH_DIR}/smoke"; shift ;;
    -h|--help)  sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

# ---------------------------------------------------------------- job table
#
# Format: NAME | HYDRA OVERRIDES | SMOKE-ONLY OVERRIDES
#
# No protocol overrides here: every experiment's committed `evaluation` config
# already reports the same thing as the others in its comparison group — the
# three Atari-100k runs evaluate every 10k steps and finish with 100 episodes,
# the three cheetah-run runs evaluate every 10k steps with the deterministic
# policy. See configs/evaluation/ and the `evaluation:` block of each
# experiment. Anything a job needs beyond that belongs in its experiment file,
# not in this table.
#
# Note `ppo/dmc` still needs accelerator=gpu — it is the one experiment that
# does not select `override /trainer: gpu`.
#
# The Dreamer smoke overrides pull `world_model_video_log_every` down to 8
# frames on purpose: at the committed 50k cadence a smoke run never reaches the
# video path, which is exactly how a decoder-key crash on DMC Proprio shipped
# past a green `--smoke`. The agent video is disabled there instead — it rolls
# out hundreds of env steps and adds nothing a smoke run needs.
#
JOBS=(
  "ppo_atari100k_jamesbond|experiment=ppo/ale environment.task=Jamesbond|trainer.total_frames=256 trainer.num_envs=1 trainer.log_every_n_steps=64 algorithm.frames_per_batch=64 algorithm.mini_batch_size=32 algorithm.num_epochs=2 algorithm.anneal_frames=256"

  "bbf_atari100k_jamesbond|experiment=bbf/atari100k environment.task=Jamesbond|trainer.total_frames=40 trainer.log_every_n_steps=8 algorithm.min_replay_history=8 algorithm.batch_size=2 algorithm.replay_ratio=1 algorithm.replay_capacity=200 algorithm.max_update_horizon=3 algorithm.min_update_horizon=1 algorithm.spr_depth=2 algorithm.width_scale=1 algorithm.hidden_dim=64 algorithm.reset_interval=12 algorithm.eps_annealing_frames=8"

  "dreamer_atari100k_jamesbond|experiment=dreamer/atari100k environment.task=Jamesbond|trainer.total_frames=20 trainer.log_every_n_steps=10 algorithm.world_model_video_log_every=8 algorithm.agent_video_log_every=0 model.deter=64 model.hidden=64 model.discrete=8 model.depth=8 model.units=64 algorithm.buffer_config.batch_size=16 algorithm.buffer_config.batch_length=8 algorithm.buffer_config.max_size=500 algorithm.dreamer_config.compile=false algorithm.dreamer_config.imag_horizon=3"

  "ppo_dmc_cheetah_run|experiment=ppo/dmc environment.task=cheetah-run trainer.accelerator=gpu|trainer.total_frames=256 trainer.log_every_n_steps=64 algorithm.frames_per_batch=64 algorithm.mini_batch_size=32 algorithm.num_epochs=2 algorithm.anneal_frames=256 evaluation.every_n_steps=0"

  "tdmpc2_dmc_cheetah_run|experiment=tdmpc2/dmc environment.task=cheetah-run|trainer.total_frames=120 trainer.log_every_n_steps=40 algorithm.compile=false algorithm.frames_per_batch=40 algorithm.init_random_frames=40 algorithm.pretrain_updates=1 algorithm.num_updates=1 algorithm.batch_size=4 algorithm.buffer_size=1000 algorithm.latent_dim=64 algorithm.enc_dim=32 algorithm.mlp_dim=32 algorithm.num_q=2 algorithm.num_samples=32 algorithm.num_elites=4 algorithm.num_pi_trajs=2 algorithm.iterations=1 evaluation.every_n_steps=0 evaluation.final_num_episodes=1 checkpoint.enabled=false"

  "dreamer_dmc_cheetah_run|experiment=dreamer/dmc environment.task=cheetah-run|trainer.total_frames=40 trainer.log_every_n_steps=10 algorithm.world_model_video_log_every=8 algorithm.agent_video_log_every=0 model.deter=64 model.hidden=64 model.discrete=8 model.units=64 algorithm.buffer_config.batch_size=16 algorithm.buffer_config.batch_length=8 algorithm.buffer_config.max_size=500 algorithm.dreamer_config.compile=false algorithm.dreamer_config.imag_horizon=3 evaluation.every_n_steps=0 evaluation.final_num_episodes=1"
)

# Applied to every job in --smoke. Keeps a validation pass to seconds and,
# critically, keeps it off the W&B server: `WANDB_MODE=offline` does NOT work
# here, because configs/logger/wandb.yaml passes an explicit `mode: online` to
# wandb.init(), which takes precedence over the environment variable. Smoke
# runs also carry a `smoke` tag rather than `template`, so a stray online run
# can never reach scripts/update_algo_results.py.
SMOKE_COMMON="evaluation.every_n_steps=0 evaluation.final_num_episodes=2 checkpoint.enabled=false"

if [[ $SMOKE -eq 1 ]]; then
  RUN_TAG="${TAG:-smoke}"
  EXTRA_ARGS="logger.0.mode=offline"
else
  # `--tag` exists because W&B tags are the only thing separating one sweep from
  # the next: runs from an older evaluation protocol stay `finished` and keep
  # their tag forever, and rlops cannot tell two protocols apart. Give a sweep
  # its own tag and `scripts/make_figures.sh --tag <name>` compares only it.
  RUN_TAG="${TAG:-template}"
  EXTRA_ARGS=""
fi

# ------------------------------------------------------------------- queue
mkdir -p "$BENCH_DIR"/{done,logs,runs}
QUEUE="$BENCH_DIR/queue.txt"
: > "$QUEUE"

IFS=',' read -ra GPU_LIST <<< "$GPUS"
IFS=',' read -ra SEED_LIST <<< "$SEEDS"

queued=0
skipped=0
for job in "${JOBS[@]}"; do
  name="${job%%|*}"
  rest="${job#*|}"
  overrides="${rest%%|*}"
  smoke_overrides="${rest#*|}"

  if [[ -n "$ONLY" ]]; then
    match=0
    IFS=',' read -ra filters <<< "$ONLY"
    for f in "${filters[@]}"; do [[ "$name" == *"$f"* ]] && match=1; done
    [[ $match -eq 0 ]] && continue
  fi

  # SMOKE_COMMON before the per-job overrides so a job can still specialise;
  # both come after $overrides so they beat the real protocol settings (the
  # Atari jobs otherwise inherit a 100-episode final evaluation, which on
  # Jamesbond is up to 450k env steps of untrained play).
  [[ $SMOKE -eq 1 ]] && overrides="$overrides $SMOKE_COMMON $smoke_overrides"

  for seed in "${SEED_LIST[@]}"; do
    if [[ -f "$BENCH_DIR/done/${name}-seed${seed}.done" ]]; then
      skipped=$((skipped + 1))
      continue
    fi
    printf '%s\t%s\t%s\n' "$name" "$seed" "$overrides" >> "$QUEUE"
    queued=$((queued + 1))
  done
done

echo "queued=$queued  already-done=$skipped  gpus=${GPU_LIST[*]}  seeds=${SEED_LIST[*]}  dir=$BENCH_DIR"
[[ $SMOKE -eq 1 ]] && echo "MODE: smoke (tiny budgets — results are meaningless, specs are not)"
[[ $queued -eq 0 ]] && { echo "nothing to do"; exit 0; }

build_cmd() {  # name seed overrides gpu
  local name=$1 seed=$2 overrides=$3
  echo "python src/train.py $overrides" \
       "trainer.seed=$seed" \
       "trainer.devices=[0]" \
       "logger.0.tags=[$RUN_TAG]" \
       $EXTRA_ARGS \
       "hydra.run.dir=$BENCH_DIR/runs/${name}-seed${seed}" \
       "paths.output_dir=$BENCH_DIR/runs/${name}-seed${seed}"
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
    local log="$BENCH_DIR/logs/${tag}.log"
    echo "[gpu $gpu] START  $tag  ($(date +%H:%M:%S))"

    local start=$SECONDS
    # `trainer.devices=[0]` plus CUDA_VISIBLE_DEVICES: the process sees exactly
    # one GPU, so device index 0 always means the right card and nothing can
    # leak onto a neighbour's.
    if CUDA_VISIBLE_DEVICES="$gpu" $(build_cmd "$name" "$seed" "$overrides") > "$log" 2>&1; then
      touch "$BENCH_DIR/done/${tag}.done"
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
for job in "${JOBS[@]}"; do
  name="${job%%|*}"
  for seed in "${SEED_LIST[@]}"; do
    [[ -n "$ONLY" ]] && { m=0; IFS=',' read -ra fs <<< "$ONLY"; for f in "${fs[@]}"; do [[ "$name" == *"$f"* ]] && m=1; done; [[ $m -eq 0 ]] && continue; }
    total=$((total + 1))
    if [[ -f "$BENCH_DIR/done/${name}-seed${seed}.done" ]]; then
      ok=$((ok + 1))
    else
      echo "  MISSING  ${name}-seed${seed}  -> $BENCH_DIR/logs/${name}-seed${seed}.log"
    fi
  done
done
echo "  $ok/$total complete"
[[ $ok -eq $total ]] || exit 1
