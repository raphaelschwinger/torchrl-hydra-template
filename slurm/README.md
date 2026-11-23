# Slurm (vcai cluster)

One training run per Slurm job, on preemptible `scavenger` GPUs. Sweeps are
the same YAML format as `scripts/sweeps/` and are expanded by `scripts/jobs.py`.

```bash
slurm/submit.sh --dry-run                       # print the 104 sbatch commands
slurm/submit.sh --smoke --only bbf_pong,dreamer_pong --seeds 1   # tiny budgets, W&B offline
slurm/submit.sh                                 # full Atari-100k: 26 games x 2 seeds x {BBF, DreamerV3}
slurm/submit.sh --sweep slurm/sweeps/bbf_atari100k_full.yaml --gres gpu:b200:1
```

- **Commit.** Jobs run the commit at `HEAD` when you submit: each job extracts
  it with `git archive` into its private `$TMPDIR`, so later edits or checkouts
  never reach queued jobs. Push it first so W&B can resolve the commit.
- **Outputs.** `~/runs/<tag>[/smoke]/{runs,logs,done}/`, one `<name>-seed<N>`
  cell each. `submit.sh` skips cells with a `done/` marker or with a job of the
  same name (`<name>-s<N>`) already in `squeue`, so re-running it tops up
  failed or missing cells.
- **GPU.** `--gres=gpu:rtx_pro_6000:1` by default. Name a type: the untyped
  `gpu:1` can land on a MIG slice. Full GPUs: `rtx_pro_6000` (athena, iris,
  nereus), `b200` (apollo, hephaestus, poseidon).
- **Preemption.** `scavenger` requeues preempted jobs (`--requeue`). A requeued
  job restarts its run from scratch, because the replay buffer is not
  checkpointed. The killed attempt stays in W&B as `crashed`. Every run is
  tagged `restart<SLURM_RESTART_COUNT>`, so attempts can be told apart. The
  log file is appended to, not overwritten.

## Environment (`env.sh`)

A fresh `uv sync` into the job-private `/tmp` costs 5–11 min and ~2 GB of
downloads per job. `env.sh` keeps everything on the node-local persistent cache
`/scratch/node-cache/$UID`:

| path | contents |
|---|---|
| `uv-cache/`, `uv-python/` | uv wheel cache and managed interpreters |
| `venvs/<sha256(uv.lock + .python-version)[:12]>/` | deps-only venv shared by all jobs on the node with that lock; `.ok` stamp marks a complete build |
| `inductor/`, `triton/` | `torch.compile` caches: only the first Dreamer job per node pays the cold max-autotune compile |

The first job on a node builds the venv under `flock`. Concurrent jobs wait on
the lock, and later jobs reuse the venv in seconds. The cache and the venvs sit
on one disk, so `UV_LINK_MODE=hardlink` links files out of the cache instead of
copying them. The project itself goes on `PYTHONPATH` from the extracted commit,
not into the venv. A build cut off by preemption has no stamp and is rebuilt.
