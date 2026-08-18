#!/usr/bin/env bash
#
# Comparison figures + rliable aggregates, straight from W&B.
#
# This is a thin wrapper around openrlbenchmark's own `rlops` CLI
# (https://github.com/openrlbenchmark/openrlbenchmark) — no plotting code of our
# own. It works without adapters because the logging contract already matches
# what rlops expects: top-level `env_id` / `exp_name` / `seed`, one row per
# episode on `charts/episodic_return`, and `global_step` as a real column.
# `tests/test_evaluation_contract.py` is what keeps that true.
#
#   ./scripts/make_figures.sh                    # every group, tag `template`
#   ./scripts/make_figures.sh --group atari100k  # one group
#   ./scripts/make_figures.sh --tag template-v2  # a different W&B tag
#   ./scripts/make_figures.sh --metric charts/episodic_return   # override metric
#   ./scripts/make_figures.sh --tag template-v2 --publish       # + copy into docs/figures/
#   ./scripts/make_figures.sh --full --publish   # publish-quality bootstrap CIs
#
# rliable's Stratified Bootstrap CIs default (in openrlbenchmark) to only 10
# reps per estimator — fine for iterating on layout, too few to trust the
# interval widths. `--full` switches to openrlbenchmark's own recommended
# values (sample-efficiency 50000, performance-profile/interval-estimates
# 2000 each); use it whenever a figure is headed for docs/figures/. The three
# `--bootstrap-reps-*` flags override individually if you need something else.
#
# Output: logs/analysis/<group>{,_aggregate,_performance_profile,
# _sample_efficiency,_sample_walltime_efficiency}.{png,pdf,svg} plus a
# markdown/csv score table. `--publish` copies the PNGs and tables into
# docs/figures/, which is what docs/evaluation.md embeds — regenerate the committed
# figures with `--tag <sweep tag> --publish`.
#
# READ THE CAVEAT BEFORE USING THE NUMBERS: rlops averages the last
# `--metric-last-n-average-window` (100) logged points of
# `charts/episodic_return`, whatever stream `canonical_source` pointed it at.
# Runs recorded under different evaluation protocols are therefore NOT
# comparable, and rlops cannot detect that. Two failure modes seen in practice:
#
#   * a run whose canonical episodes all sit at one step (periodic eval off)
#     renders as a flat horizontal line across the whole x-axis, because the
#     single point is back-filled — it looks like a curve and is not one;
#   * a run that crashed early still shows as `finished` on W&B (the trainer
#     closes the logger in a `finally`), so a truncated run silently joins the
#     comparison. Check the seed/step counts rlops prints before trusting a plot.
#
set -uo pipefail
# `?` and `&` in the filter strings must reach rlops literally.
set -f

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1

ENTITY="${WANDB_ENTITY:-LatentLab}"
PROJECT="${WANDB_PROJECT:-torchrl-hydra-template}"
TAG="template"
GROUP=""
METRIC=""
OUT_DIR="logs/analysis"
PUBLISH_DIR="docs/figures"
PUBLISH=0
VENV=".venv-openrlbenchmark"
# openrlbenchmark's own quick-test defaults; --full switches to its
# recommended values (see comments in openrlbenchmark/rlops.py's RliableConfig).
SAMPLE_EFFICIENCY_REPS=10
PERFORMANCE_PROFILE_REPS=10
INTERVAL_ESTIMATES_REPS=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    --entity)   ENTITY="$2"; shift 2 ;;
    --project)  PROJECT="$2"; shift 2 ;;
    --tag)      TAG="$2"; shift 2 ;;
    --group)    GROUP="$2"; shift 2 ;;
    --metric)   METRIC="$2"; shift 2 ;;
    --out)      OUT_DIR="$2"; shift 2 ;;
    --publish)  PUBLISH=1; shift ;;
    --full)
      SAMPLE_EFFICIENCY_REPS=50000
      PERFORMANCE_PROFILE_REPS=2000
      INTERVAL_ESTIMATES_REPS=2000
      shift ;;
    --bootstrap-reps-sample-efficiency)  SAMPLE_EFFICIENCY_REPS="$2"; shift 2 ;;
    --bootstrap-reps-performance-profile) PERFORMANCE_PROFILE_REPS="$2"; shift 2 ;;
    --bootstrap-reps-interval-estimates)  INTERVAL_ESTIMATES_REPS="$2"; shift 2 ;;
    -h|--help)  sed -n '2,43p' "$0"; exit 0 ;;
    *) echo "unknown option: $1" >&2; exit 1 ;;
  esac
done

# ------------------------------------------------------------------ toolchain
# openrlbenchmark pulls seaborn / expt / older numpy pins. Installing it into
# the training venv risks moving torch's numpy out from under it, so it gets its
# own environment; `uv` is already a dependency of the devcontainer.
if [[ ! -x "$VENV/bin/python" ]]; then
  echo "building $VENV (one-off)"
  uv venv --python 3.10 "$VENV" || exit 1
  uv pip install --python "$VENV/bin/python" --quiet openrlbenchmark || exit 1
fi

mkdir -p "$OUT_DIR"

# -------------------------------------------------------------------- groups
#
# Format: NAME | ENV-IDS | NORMALIZATION | METRIC | EXPERIMENTS (exp_name:label,...)
#
# `exp_name` is the top-level config key (= the algorithm config's name), which
# is what rlops matches on via `cen=exp_name`. `atari` normalization uses
# openrlbenchmark's built-in human/random table, keyed `Jamesbond-v5` — the
# reason `env_id` carries no `ALE/` prefix.
#
# The metric is `charts/eval_episodic_return`, not the canonical
# `charts/episodic_return`, so a figure always compares the *same measurement*
# across algorithms. `canonical_source` is a per-experiment decision — legitimate
# either way (see configs/evaluation/) — but it means the canonical key is eval
# rollouts for BBF and DreamerV3 and the training stream for ppo/ale. Plotting
# those together would put a stochastic-policy curve next to two
# deterministic-protocol ones. On DMC the two keys are identical, since every
# experiment there is already eval-canonical.
#
# NOT `GROUPS`: bash reserves that name for the caller's group IDs and silently
# ignores the assignment, so every field would parse as a gid.
FIG_GROUPS=(
  "atari100k|Jamesbond-v5|atari|charts/eval_episodic_return|dreamer:DreamerV3,bbf:BBF,ppo:PPO"
  "dmc|cheetah-run|maxmin|charts/eval_episodic_return|dreamer:DreamerV3,tdmpc2:TD-MPC2,ppo:PPO"
)

status=0
for spec in "${FIG_GROUPS[@]}"; do
  name="${spec%%|*}";      rest="${spec#*|}"
  env_ids="${rest%%|*}";   rest="${rest#*|}"
  norm="${rest%%|*}";      rest="${rest#*|}"
  metric="${rest%%|*}";    experiments="${rest#*|}"
  [[ -n "$METRIC" ]] && metric="$METRIC"

  [[ -n "$GROUP" && "$GROUP" != "$name" ]] && continue

  # 'dreamer:DreamerV3,bbf:BBF' -> 'dreamer?tag=template&cl=DreamerV3' ...
  args=()
  IFS=',' read -ra pairs <<< "$experiments"
  for pair in "${pairs[@]}"; do
    args+=("${pair%%:*}?tag=${TAG}&cl=${pair#*:}")
  done

  echo "=== $name  (env_ids=$env_ids  norm=$norm  metric=$metric  tag=$TAG)"
  "$VENV/bin/python" -m openrlbenchmark.rlops \
    --filters "?we=${ENTITY}&wpn=${PROJECT}&ceik=env_id&cen=exp_name&metric=${metric}" \
      "${args[@]}" \
    --env-ids $env_ids \
    --no-check-empty-runs \
    --pc.ncols 1 --pc.ncols-legend 3 \
    --rliable \
    --rc.score-normalization-method "$norm" \
    --rc.normalized-score-threshold 8.0 \
    --rc.sample-efficiency-plots \
    --rc.performance-profile-plots \
    --rc.aggregate-metrics-plots \
    --rc.sample-efficiency-num-bootstrap-reps "$SAMPLE_EFFICIENCY_REPS" \
    --rc.performance-profile-num-bootstrap-reps "$PERFORMANCE_PROFILE_REPS" \
    --rc.interval-estimates-num-bootstrap-reps "$INTERVAL_ESTIMATES_REPS" \
    --output-filename "${OUT_DIR}/${name}" \
    --scan-history || status=1
done

echo
echo "figures written to ${OUT_DIR}/"

# ------------------------------------------------------------------- publish
# `logs/` is gitignored, so the README's figures need a copy that is not. Only
# the PNGs and the score tables move: rlops also emits PDF and SVG of every
# panel, which are ~40x the size and are not what a README renders.
if [[ $PUBLISH -eq 1 ]]; then
  # `set -f` above keeps the rlops filter strings intact; the copy below is the
  # one place that actually wants globbing.
  set +f
  mkdir -p "$PUBLISH_DIR"
  for spec in "${FIG_GROUPS[@]}"; do
    name="${spec%%|*}"
    [[ -n "$GROUP" && "$GROUP" != "$name" ]] && continue
    cp "$OUT_DIR/${name}"*.png "$PUBLISH_DIR/" || status=1
    cp "$OUT_DIR/${name}.md" "$PUBLISH_DIR/" || status=1
  done
  published=$(ls "$PUBLISH_DIR"/*.png 2>/dev/null | wc -l)
  set -f
  echo "published ${published} PNGs + tables to ${PUBLISH_DIR}/"
fi

exit $status
