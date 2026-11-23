#!/bin/bash
# One training run per job. Submitted by slurm/submit.sh, which passes:
#
#   sbatch [--output ... --gres ...] slurm/train.sbatch \
#       REPO COMMIT SWEEP_DIR NAME SEED TAGS override...
#
# The code is extracted from the pinned COMMIT, so a queued or requeued job is
# immune to later edits of the checkout. On preemption Slurm requeues the job
# and it restarts from scratch (the replay buffer is not checkpointed); the
# killed attempt stays in W&B as crashed, and the `restartN` tag tells attempts
# apart.
#
#SBATCH --partition=scavenger
#SBATCH --qos=scavenger
#SBATCH --gres=gpu:rtx_pro_6000:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --open-mode=append

set -euo pipefail

REPO=$1 COMMIT=$2 SWEEP_DIR=$3 NAME=$4 SEED=$5 TAGS=$6
shift 6
CELL="${NAME}-seed${SEED}"
RUN_DIR="$SWEEP_DIR/runs/$CELL"
RESTART="${SLURM_RESTART_COUNT:-0}"

echo "===== $(date -Is)  job=$SLURM_JOB_ID  restart=$RESTART  host=$(hostname)"
echo "cell: $CELL  commit: $COMMIT"
nvidia-smi -L || true

if [[ -f "$SWEEP_DIR/done/$CELL.done" ]]; then
  echo "already done ($SWEEP_DIR/done/$CELL.done); nothing to do"
  exit 0
fi

CODE="$(mktemp -d "${TMPDIR:-/tmp}/code.XXXXXX")"
trap 'rm -rf "$CODE"' EXIT
git -C "$REPO" archive "$COMMIT" | tar -x -C "$CODE"
cd "$CODE"
git -C "$REPO" log -1 --format='%H %s' "$COMMIT"

# shellcheck source=slurm/env.sh
source slurm/env.sh
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

# W&B reads the commit from the git repo; the extracted tree has none.
export WANDB_GIT_COMMIT="$COMMIT"
export WANDB_GIT_REMOTE_URL="$(git -C "$REPO" remote get-url origin 2>/dev/null || true)"

mkdir -p "$RUN_DIR"
set -x
python src/train.py "$@" \
  "trainer.seed=$SEED" \
  "trainer.devices=[0]" \
  "logger.0.tags=[${TAGS},restart${RESTART}]" \
  "hydra.run.dir=$RUN_DIR" \
  "paths.output_dir=$RUN_DIR"
set +x

touch "$SWEEP_DIR/done/$CELL.done"
echo "===== $(date -Is)  done: $CELL"
