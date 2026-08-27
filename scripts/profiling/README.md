# Profiling study

Where does the wall-clock of one baseline Atari-100k run go, per algorithm?

**MEASURED STUDY** — results are wall-clock timings, not byte-reproducible. Refresh
the paper figure by re-running the sweep and exporting CSV:

```bash
GPU=2 ./scripts/profiling/run_profiling.sh quick=true   # smoke (~2 min)
GPU=2 ./scripts/profiling/run_profiling.sh                # full sweep (hours)
uv run python scripts/profiling/export_csv.py             # → paper/data/figures/profile_breakdown.csv
cd paper && make plots
```

Launch through `run_profiling.sh` (idle-GPU check, RSS ceiling, thread cap). Each
cell is a fresh subprocess of `src/train.py` with
`trainer._target_=src.trainers.profiling.ProfilingStepTrainer`.

Aggregated output: `logs/profiling/results/profile.parquet` (gitignored).
