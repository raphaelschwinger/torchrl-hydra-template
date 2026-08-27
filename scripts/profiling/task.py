"""Wall-clock profiling study driver.

MEASURED STUDY — wall-clock timings are not byte-reproducible.
See ``scripts/profiling/README.md``.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from src.profiling.phases import PHASE_ORDER, RESIDUAL_PHASE, ROOT, axis, collapse
from src.utils.environment import environment_record
from src.utils.measured import launch_cell
from src.utils.paths import repo_root, results_dir

log = logging.getLogger(__name__)

RESULT_DECIMALS = 3

COLUMNS = [
    "algorithm",
    "label",
    "task",
    "seed",
    "experiment",
    "total_frames",
    "phase",
    "axis",
    "self_seconds",
    "share",
    "calls",
    "agent_steps",
    "total_wall_seconds",
    "uninstrumented_wall_seconds",
    "profiling_overhead",
    "setup_seconds",
    "sync_cuda",
    "peak_rss_gib",
    "gpu_mem_used_start_mib",
    "gpu_util_start_pct",
    "loadavg_start",
    "status",
    "gpu_index",
    "physical_gpu_index",
    "gpu_name",
    "torch_version",
    "torchrl_version",
    "loadavg",
    "cpu_count",
    "error",
]

NO_PHASE = ""
STATUSES = frozenset({"ok", "failed", "timeout", "oom"})


@dataclass(frozen=True)
class CellSpec:
    algorithm: str
    label: str
    task: str
    seed: int
    experiment: str
    total_frames: int
    profiled: bool
    overrides: tuple[str, ...] = field(default_factory=tuple)

    @property
    def cell_id(self) -> str:
        suffix = "prof" if self.profiled else "plain"
        return f"{self.algorithm}.{self.task}.s{self.seed}.{suffix}"


def _plain_list(node) -> list[str]:
    if node is None:
        return []
    return [str(item) for item in OmegaConf.to_container(node, resolve=True)]


def cell_specs(cfg: DictConfig) -> list[CellSpec]:
    keep = cfg.get("runs_filter", None)
    keep = set(_plain_list(keep)) if keep else None
    specs: list[CellSpec] = []
    for seed in cfg.seeds:
        for name, run in cfg.runs.items():
            if keep is not None and name not in keep:
                continue
            total_frames = int(cfg.quick_total_frames) if cfg.quick else int(run.total_frames)
            common = dict(
                algorithm=str(name),
                label=str(run.label),
                task=str(cfg.task),
                seed=int(seed),
                experiment=str(run.experiment),
                total_frames=total_frames,
                overrides=tuple(_plain_list(run.get("overrides"))),
            )
            specs.append(CellSpec(profiled=True, **common))
            if cfg.get("overhead_run", False):
                specs.append(CellSpec(profiled=False, **common))
    return specs


def _command(spec: CellSpec, cfg: DictConfig, run_dir: Path) -> list[str]:
    root = repo_root()
    profile_json = run_dir / "profile_phases.json"
    command = [
        sys.executable,
        str(root / str(cfg.train_script)),
        f"experiment={spec.experiment}",
        f"environment.task={spec.task}",
        f"trainer.seed={spec.seed}",
        f"trainer.total_frames={spec.total_frames}",
        "trainer.accelerator=gpu",
        f"trainer.devices=[{int(cfg.device_index)}]",
        "trainer._target_=src.trainers.profiling.ProfilingStepTrainer",
        f"+profiling.enabled={str(bool(spec.profiled)).lower()}",
        f"+profiling.sync_cuda={str(bool(cfg.sync_cuda)).lower()}",
        f"+profiling.output={profile_json}",
        f"hydra.run.dir={run_dir}",
    ]
    command += _plain_list(cfg.get("common_overrides"))
    command += list(spec.overrides)
    if cfg.quick:
        command += _plain_list(cfg.get("quick_overrides"))
    return command


def run_cell_subprocess(spec: CellSpec, cfg: DictConfig, log_dir: Path) -> dict[str, Any]:
    run_dir = log_dir / spec.cell_id
    run_dir.mkdir(parents=True, exist_ok=True)
    outcome = launch_cell(
        _command(spec, cfg, run_dir),
        log_dir / f"{spec.cell_id}.log",
        timeout_seconds=float(cfg.cell_timeout_seconds),
        rss_limit_gib=cfg.get("rss_limit_gib", None),
        gpu_index=int(cfg.get("physical_device_index", cfg.device_index)),
    )

    profile: dict[str, Any] = {}
    status, error = outcome["status"], outcome["error"]
    profile_json = run_dir / "profile_phases.json"
    if profile_json.exists():
        try:
            profile = json.loads(profile_json.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            status = "failed" if status == "ok" else status
            error = error or f"unreadable profile: {exc}"
    elif status == "ok":
        status = "failed"
        error = "no profile written"

    return {**outcome, "spec": spec, "status": status, "error": error[:200], "profile": profile}


def load_cell(spec: CellSpec, cells_dir: Path) -> dict[str, Any]:
    profile_json = cells_dir / spec.cell_id / "profile_phases.json"
    if not profile_json.exists():
        return {"spec": spec, "status": "failed", "error": "no profile written", "profile": {}}
    try:
        profile = json.loads(profile_json.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        return {
            "spec": spec,
            "status": "failed",
            "error": f"unreadable profile: {exc}"[:200],
            "profile": {},
        }
    steps = int(profile.get("agent_steps", 0) or 0)
    if steps < spec.total_frames:
        return {
            "spec": spec,
            "status": "failed",
            "error": f"run stopped at {steps} of {spec.total_frames} agent steps",
            "profile": profile,
        }
    return {"spec": spec, "status": "ok", "error": "", "profile": profile}


def rows_for(
    cell: dict[str, Any], reference: dict[str, Any] | None, cfg: DictConfig
) -> list[dict[str, Any]]:
    spec: CellSpec = cell["spec"]
    profile = cell["profile"] or {}
    env = profile.get("environment", {}) or {}
    total = float(profile.get("total_wall_seconds", float("nan")) or float("nan"))
    plain = float("nan")
    if reference is not None:
        plain = float(
            (reference.get("profile") or {}).get("total_wall_seconds", float("nan"))
            or float("nan")
        )

    base = {
        "algorithm": spec.algorithm,
        "label": spec.label,
        "task": spec.task,
        "seed": spec.seed,
        "experiment": spec.experiment,
        "total_frames": spec.total_frames,
        "agent_steps": int(profile.get("agent_steps", 0) or 0),
        "total_wall_seconds": total,
        "uninstrumented_wall_seconds": plain,
        "profiling_overhead": (total / plain - 1.0) if plain and plain == plain else float("nan"),
        "setup_seconds": float(profile.get("setup_seconds", float("nan")) or float("nan")),
        "sync_cuda": bool(profile.get("sync_cuda", False)),
        "peak_rss_gib": float(cell.get("peak_rss_gib", float("nan"))),
        "gpu_mem_used_start_mib": float(cell.get("gpu_mem_used_start_mib", float("nan"))),
        "gpu_util_start_pct": float(cell.get("gpu_util_start_pct", float("nan"))),
        "loadavg_start": float(env.get("loadavg_start", float("nan"))),
        "status": cell["status"],
        "gpu_index": int(env.get("gpu_index", -1)),
        "physical_gpu_index": int(cfg.get("physical_device_index", cfg.device_index)),
        "gpu_name": str(env.get("gpu_name", "")),
        "torch_version": str(env.get("torch", "")),
        "torchrl_version": str(env.get("torchrl", "")),
        "loadavg": float(env.get("loadavg", float("nan"))),
        "cpu_count": int(env.get("cpu_count", 0) or 0),
        "error": cell["error"],
    }

    phases = profile.get("phases") or {}
    if cell["status"] != "ok" or not phases:
        return [
            {
                **base,
                "phase": NO_PHASE,
                "axis": NO_PHASE,
                "self_seconds": float("nan"),
                "share": float("nan"),
                "calls": 0,
            }
        ]

    calls = profile.get("calls") or {}
    seconds: dict[str, float] = {}
    counts: dict[str, int] = {}
    for path, value in phases.items():
        if path != ROOT and not path.startswith(f"{ROOT}/"):
            continue
        bucket = collapse(path)
        if bucket == "run":
            bucket = RESIDUAL_PHASE
        seconds[bucket] = seconds.get(bucket, 0.0) + float(value)
        counts[bucket] = counts.get(bucket, 0) + int(calls.get(path, 0))

    residual = total - sum(seconds.values())
    seconds[RESIDUAL_PHASE] = seconds.get(RESIDUAL_PHASE, 0.0) + residual
    counts.setdefault(RESIDUAL_PHASE, 0)

    rows = []
    for phase in PHASE_ORDER:
        if phase not in seconds:
            continue
        value = seconds[phase]
        rows.append(
            {
                **base,
                "phase": phase,
                "axis": axis(phase),
                "self_seconds": value,
                "share": value / total if total else float("nan"),
                "calls": counts.get(phase, 0),
            }
        )
    unknown = sorted(set(seconds) - set(PHASE_ORDER))
    if unknown:
        raise ValueError(f"phase(s) with no entry in PHASE_ORDER: {unknown}")
    return rows


def aggregate(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=COLUMNS)
    if frame.empty:
        return frame
    order = pd.Categorical(frame["phase"], categories=[NO_PHASE, *PHASE_ORDER], ordered=True)
    frame = frame.assign(_phase_order=order).sort_values(
        ["algorithm", "task", "seed", "_phase_order"], ignore_index=True
    )
    frame = frame.drop(columns="_phase_order")
    for column in (
        "seed",
        "total_frames",
        "agent_steps",
        "calls",
        "gpu_index",
        "physical_gpu_index",
        "cpu_count",
    ):
        frame[column] = frame[column].fillna(0).astype("int32")
    for column in (
        "label",
        "experiment",
        "phase",
        "axis",
        "status",
        "gpu_name",
        "torch_version",
        "torchrl_version",
        "error",
    ):
        frame[column] = frame[column].fillna("")
    floats = frame.select_dtypes("float").columns
    frame[floats] = frame[floats].round(RESULT_DECIMALS)
    return frame[COLUMNS]


def raw_frame(cells: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for cell in cells:
        spec: CellSpec = cell["spec"]
        profile = cell["profile"] or {}
        for path, value in (profile.get("phases") or {}).items():
            rows.append(
                {
                    **{k: v for k, v in asdict(spec).items() if k != "overrides"},
                    "status": cell["status"],
                    "phase_path": path,
                    "phase": collapse(path),
                    "self_seconds": float(value),
                    "total_seconds": float((profile.get("totals") or {}).get(path, 0.0)),
                    "calls": int((profile.get("calls") or {}).get(path, 0)),
                }
            )
    return pd.DataFrame(rows)


@hydra.main(version_base="1.3", config_path="../../configs", config_name="profiling")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    log.info("config:\n%s", OmegaConf.to_yaml(cfg, resolve=True))

    run_dir = Path(cfg.paths.output_dir)
    log_dir = run_dir / "cells"
    log_dir.mkdir(parents=True, exist_ok=True)

    reuse = cfg.get("reaggregate_from", None)
    reuse_dir = Path(str(reuse)) if reuse else None
    if reuse_dir is not None and not reuse_dir.is_absolute():
        reuse_dir = repo_root() / reuse_dir

    specs = cell_specs(cfg)
    cells: list[dict[str, Any]] = []
    for index, spec in enumerate(specs, start=1):
        log.info("[%2d/%2d] %s", index, len(specs), spec.cell_id)
        if reuse_dir is not None:
            cell = load_cell(spec, reuse_dir / "cells")
        else:
            cell = run_cell_subprocess(spec, cfg, log_dir)
        cells.append(cell)
        wall = (cell["profile"] or {}).get("total_wall_seconds", float("nan"))
        if cell["status"] == "ok":
            log.info("         %8.1f s wall", wall)
        else:
            log.warning("         %s: %s", cell["status"], cell["error"])

    references = {
        (c["spec"].algorithm, c["spec"].seed): c
        for c in cells
        if not c["spec"].profiled and c["status"] == "ok"
    }
    rows: list[dict[str, Any]] = []
    for cell in cells:
        if not cell["spec"].profiled:
            continue
        key = (cell["spec"].algorithm, cell["spec"].seed)
        rows.extend(rows_for(cell, references.get(key), cfg))

    raw = raw_frame(cells)
    if not raw.empty:
        raw.to_parquet(run_dir / "phases_raw.parquet", index=False, compression="snappy")

    summary = aggregate(rows)
    destination = results_dir(cfg.study_name) / f"{cfg.results_name}.parquet"
    summary.to_parquet(destination, index=False, compression="snappy")
    log.info("wrote %s (%d rows)", destination, len(summary))

    record = environment_record()
    record["cells"] = len(cells)
    record["ok_cells"] = int(sum(c["status"] == "ok" for c in cells))
    record["device_index"] = int(cfg.device_index)
    record["physical_device_index"] = int(cfg.get("physical_device_index", cfg.device_index))
    record["rss_limit_gib"] = float(cfg.get("rss_limit_gib", float("nan")) or float("nan"))
    record["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    record["sync_cuda"] = bool(cfg.sync_cuda)
    (results_dir(cfg.study_name) / "environment.json").write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n"
    )

    for algorithm, group in summary[summary.status == "ok"].groupby("algorithm"):
        top = group.sort_values("share", ascending=False).iloc[0]
        log.info(
            "%-12s %8.1f s total, largest bucket %s at %.0f%%",
            algorithm,
            float(top.total_wall_seconds),
            top.phase,
            100 * float(top.share),
        )


if __name__ == "__main__":
    main()
