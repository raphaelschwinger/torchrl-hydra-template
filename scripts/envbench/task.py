"""Environment-provider throughput sweep driver.

MEASURED STUDY — wall-clock timings are not byte-reproducible.
See ``scripts/envbench/README.md``.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.envbench.spec import RESULT_MARKER, CellResult, CellSpec
from src.utils.environment import environment_record
from src.utils.paths import repo_root, results_dir

log = logging.getLogger(__name__)

RESULT_DECIMALS = 3

COLUMNS = [
    "suite",
    "task",
    "provider",
    "label",
    "substrate",
    "parallelism",
    "device",
    "obs_contract",
    "obs_signature",
    "num_envs",
    "action_repeat",
    "sim_seconds_per_agent_step",
    "status",
    "n_seeds",
    "sps_mean",
    "sps_std",
    "fps_mean",
    "fps_std",
    "setup_s_mean",
    "wall_s_mean",
    "loadavg_mean",
    "cpu_count",
    "gpu_name",
    "provider_version",
    "error",
]


def cell_specs(cfg: DictConfig) -> list[tuple[CellSpec, dict[str, Any]]]:
    seeds = [int(s) for s in (cfg.seeds[:1] if cfg.quick else cfg.seeds)]
    wanted = OmegaConf.to_container(cfg.providers_filter) if cfg.providers_filter else None

    specs: list[tuple[CellSpec, dict[str, Any]]] = []
    for seed in seeds:
        for name, provider in cfg.providers.items():
            if wanted is not None and name not in wanted:
                continue
            suite = cfg.suites[provider.suite]
            grid = provider.get("num_envs", None) or cfg.num_envs
            grid = [int(n) for n in grid]
            if cfg.quick:
                grid = grid[: cfg.quick_grid_size]
            for num_envs in grid:
                spec = CellSpec(
                    suite=provider.suite,
                    task=str(suite.task),
                    provider=str(name),
                    kind=str(provider.kind),
                    num_envs=num_envs,
                    seed=seed,
                    action_repeat=int(suite.action_repeat),
                    obs_contract=str(provider.get("obs_contract", None) or suite.obs_contract),
                    parallelism=str(provider.parallelism),
                    device=str(provider.device),
                    sim_seconds_per_agent_step=float(suite.sim_seconds_per_agent_step),
                    options=_plain(provider.get("options", None)),
                )
                meta = {
                    "label": str(provider.label),
                    "substrate": str(provider.substrate),
                    "command": _command_for(cfg, provider),
                    "env": _plain(provider.get("env", None)),
                }
                specs.append((spec, meta))
    return specs


def _plain(node: Any) -> dict[str, Any]:
    if node is None:
        return {}
    if isinstance(node, dict):
        return dict(node)
    return dict(OmegaConf.to_container(node, resolve=True))


def _command_for(cfg: DictConfig, provider: DictConfig) -> list[str]:
    override = _plain_list(provider.get("command", None))
    if override:
        return [_resolve(part) for part in override]

    executable = provider.get("python_executable", None)
    interpreter = _resolve(str(executable)) if executable else sys.executable
    worker = repo_root() / "scripts" / "envbench" / "worker.py"
    return [interpreter, str(worker)]


def _resolve(part: str) -> str:
    if "/" in part and not part.startswith("/"):
        return str(repo_root() / part)
    return part


def _plain_list(node: Any) -> list[str]:
    if node is None:
        return []
    if isinstance(node, list):
        return [str(x) for x in node]
    return [str(x) for x in OmegaConf.to_container(node, resolve=True)]


def run_cell_subprocess(
    spec: CellSpec,
    meta: dict[str, Any],
    cfg: DictConfig,
    log_dir: Path,
) -> dict[str, Any]:
    command = [
        *meta["command"],
        "--spec",
        json.dumps(spec.to_json()),
        "--warmup-seconds",
        str(cfg.warmup_seconds if not cfg.quick else cfg.quick_warmup_seconds),
        "--measure-seconds",
        str(cfg.measure_seconds if not cfg.quick else cfg.quick_measure_seconds),
        "--min-iters",
        str(cfg.min_iters),
        "--max-iters",
        str(cfg.max_iters),
    ]

    environment = dict(os.environ)
    environment.update({"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})
    environment.update({str(k): str(v) for k, v in meta["env"].items()})
    environment["PYTHONUNBUFFERED"] = "1"

    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=cfg.cell_timeout_seconds,
            check=False,
            env=environment,
            cwd=repo_root(),
        )
        stdout, stderr, returncode = completed.stdout, completed.stderr, completed.returncode
        timed_out = False
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout or ""
        stderr = exc.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        returncode = -1
        timed_out = True

    (log_dir / f"{spec.cell_id}.log").write_text(
        f"$ {' '.join(command)}\n\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}\n"
    )

    result = _parse_result(stdout)
    if result is None:
        result = CellResult(status="timeout" if timed_out else "failed")
        tail = [line for line in stderr.strip().splitlines() if line.strip()]
        result.error = (
            f"cell exceeded {cfg.cell_timeout_seconds}s"
            if timed_out
            else (tail[-1][:200] if tail else f"no result line (exit {returncode})")
        )

    row = {**spec.to_json(), **result.to_json()}
    row["label"] = meta["label"]
    row["substrate"] = meta["substrate"]
    row.pop("options", None)
    row.pop("kind", None)
    return row


def _parse_result(stdout: str) -> CellResult | None:
    for line in reversed(stdout.splitlines()):
        if line.startswith(RESULT_MARKER):
            return CellResult(**json.loads(line[len(RESULT_MARKER) :]))
    return None


def aggregate(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    keys = ["suite", "task", "provider", "num_envs"]

    constants = [
        "label",
        "substrate",
        "parallelism",
        "device",
        "obs_contract",
        "action_repeat",
        "cpu_count",
    ]
    meta = frame.groupby(keys, as_index=False)[constants].first()

    aggregations = {
        "n_seeds": ("sps", "count"),
        "sps_mean": ("sps", "mean"),
        "sps_std": ("sps", "std"),
        "fps_mean": ("fps", "mean"),
        "fps_std": ("fps", "std"),
        "setup_s_mean": ("setup_s", "mean"),
        "wall_s_mean": ("wall_s", "mean"),
        "loadavg_mean": ("loadavg", "mean"),
        "sim_seconds_per_agent_step": ("sim_seconds_per_agent_step", "first"),
        "obs_signature": ("obs_signature", "first"),
        "provider_version": ("provider_version", "first"),
        "gpu_name": ("gpu_name", "first"),
    }
    successes = frame[frame.status == "ok"]
    if len(successes):
        stats = successes.groupby(keys, as_index=False).agg(**aggregations)
    else:
        stats = pd.DataFrame(columns=[*keys, *aggregations])

    summary = meta.merge(stats, on=keys, how="left")

    status = frame.groupby(keys, as_index=False).agg(
        all_ok=("status", lambda s: bool((s == "ok").all())),
        worst=("status", lambda s: sorted(set(s) - {"ok"})[0] if set(s) - {"ok"} else "ok"),
        error=("error", lambda s: next((e for e in s if e), "")),
    )
    summary = summary.merge(status, on=keys, how="left")
    summary["status"] = np.where(summary["all_ok"], "ok", summary["worst"])
    summary = summary.drop(columns=["all_ok", "worst"])

    for column in ("obs_signature", "provider_version", "gpu_name", "error"):
        summary[column] = summary[column].fillna("")
    summary["n_seeds"] = summary["n_seeds"].fillna(0).astype("int32")
    summary["num_envs"] = summary["num_envs"].astype("int32")
    summary["action_repeat"] = summary["action_repeat"].astype("int32")
    summary["cpu_count"] = summary["cpu_count"].astype("int32")

    floats = summary.select_dtypes("float").columns
    summary[floats] = summary[floats].round(RESULT_DECIMALS)
    summary = summary.sort_values(keys, ignore_index=True)
    return summary[COLUMNS]


@hydra.main(version_base="1.3", config_path="../../configs", config_name="envbench")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    log.info("config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    run_dir = Path(cfg.paths.output_dir)
    log_dir = run_dir / "cells"
    log_dir.mkdir(parents=True, exist_ok=True)

    specs = cell_specs(cfg)
    log.info("running %d cells", len(specs))

    rows: list[dict[str, Any]] = []
    for index, (spec, meta) in enumerate(specs, start=1):
        row = run_cell_subprocess(spec, meta, cfg, log_dir)
        rows.append(row)
        if row["status"] == "ok":
            log.info(
                "[%3d/%3d] %-34s %8.0f steps/s  %10.0f frames/s",
                index,
                len(specs),
                spec.cell_id,
                row["sps"],
                row["fps"],
            )
        else:
            log.warning(
                "[%3d/%3d] %-34s %s: %s",
                index,
                len(specs),
                spec.cell_id,
                row["status"],
                row["error"],
            )

    raw = pd.DataFrame(rows)
    raw.to_parquet(run_dir / "cells_raw.parquet", index=False, compression="snappy")

    summary = aggregate(rows)
    target = results_dir(cfg.study_name) / f"{cfg.results_name}.parquet"
    summary.to_parquet(target, index=False, compression="snappy")
    log.info("wrote %s (%d rows)", target, len(summary))

    record = environment_record()
    record["cells"] = len(rows)
    record["ok_cells"] = int((raw.status == "ok").sum())
    (results_dir(cfg.study_name) / "environment.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )

    best = summary[summary.status == "ok"]
    if len(best):
        peak = best.loc[best.groupby(["suite", "provider"])["fps_mean"].idxmax()]
        for row in peak.sort_values(["suite", "fps_mean"], ascending=[True, False]).itertuples():
            log.info(
                "peak  %-6s %-18s %10.0f frames/s @ N=%d",
                row.suite,
                row.provider,
                row.fps_mean,
                row.num_envs,
            )


if __name__ == "__main__":
    main()
