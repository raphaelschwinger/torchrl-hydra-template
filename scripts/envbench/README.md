# Environment throughput study

Measures how fast each environment provider can be stepped (random actions, no learner).

**MEASURED STUDY** — wall-clock timings, not byte-reproducible.

```bash
uv sync --extra envs                    # optional CPU providers (Linux x86_64)
./scripts/envbench/run_envbench.sh quick=true
./scripts/envbench/run_envbench.sh
uv run python scripts/envbench/export_csv.py   # → paper/data/figures/env_throughput.csv
cd paper && make plots
```

Aggregated output: `logs/envthroughput/results/throughput.parquet` (gitignored).

JAX/MJX providers require a separate venv: `uv sync --extra envsjax` and
`.venv-jax/bin/python` (see `configs/study/envthroughput.yaml`).
