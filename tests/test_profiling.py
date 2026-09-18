"""Unit tests for wall-clock phase profiling (timer, taxonomy, probes, study task)."""

from __future__ import annotations

import json
import time

import pytest
from hydra import compose, initialize_config_dir

from scripts.profiling.task import (
    COLUMNS,
    NO_PHASE,
    RESULT_DECIMALS,
    STATUSES,
    CellSpec,
    aggregate,
    cell_specs,
    load_cell,
    rows_for,
)
from src.profiling.phases import (
    PHASE_AXIS,
    PHASE_LABELS,
    PHASE_ORDER,
    RESIDUAL_PHASE,
    axis,
    collapse,
)
from src.profiling.probes import PROBES, resolve
from src.profiling.timer import PhaseTimer
from src.utils.paths import repo_root, study_dir

CONFIG_DIR = repo_root() / "configs"

with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
    CFG = compose(config_name="profiling")


@pytest.fixture(scope="module")
def cfg():
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(config_name="profiling")


def test_config_composes(cfg):
    assert cfg.study_name == "profiling"
    assert study_dir(cfg.study_name).is_dir()
    assert cfg.seeds
    assert cfg.runs
    assert cfg.task


def test_every_run_points_at_a_resolvable_experiment(cfg):
    root = repo_root() / "configs"
    for name, run in cfg.runs.items():
        path = root / "experiment" / f"{run.experiment}.yaml"
        assert path.exists(), f"{name}: no config for {run.experiment}"


def test_train_script_exists(cfg):
    assert (repo_root() / str(cfg.train_script)).is_file()


def test_the_dreamer_baseline_disables_every_tweak(cfg):
    import yaml

    path = repo_root() / "configs" / "experiment" / f"{cfg.runs.dreamerv3.experiment}.yaml"
    baseline = yaml.safe_load(path.read_text())["algorithm"]
    assert baseline["dreamer_config"]["compile"] is False
    assert baseline["dreamer_config"]["perf"]["amp"] == "off"
    assert baseline["buffer_config"]["pin_memory"] is False


def test_cells_pair_each_profiled_run_with_a_reference(cfg):
    specs = cell_specs(cfg)
    profiled = [s for s in specs if s.profiled]
    plain = [s for s in specs if not s.profiled]
    assert len(profiled) == len(cfg.runs) * len(cfg.seeds)
    assert len(plain) == len(profiled)
    assert len({s.cell_id for s in specs}) == len(specs)


def test_every_phase_has_an_axis_and_a_label():
    for phase in PHASE_ORDER:
        assert phase in PHASE_AXIS
        assert phase in PHASE_LABELS
    assert set(PHASE_AXIS) == set(PHASE_ORDER)


def test_every_probe_phase_is_in_the_taxonomy():
    for probes in PROBES.values():
        for _, phase in probes:
            assert phase in PHASE_ORDER, phase


def test_collapse_folds_evaluation_and_keeps_training_leaves():
    assert collapse("run/env_step") == "env_step"
    assert collapse("run/gradient_step/replay_sample") == "replay_sample"
    assert collapse("run/eval_final/env_step") == "eval_final"
    assert collapse("run/eval_periodic/rollout_inference") == "eval_periodic"
    assert collapse("run/eval_final") == "eval_final"
    assert axis("nonsense") == PHASE_AXIS[RESIDUAL_PHASE]


def test_self_time_partitions_the_root():
    timer = PhaseTimer()
    with timer.phase("run"):
        with timer.phase("collect"), timer.phase("env_step"):
            time.sleep(0.01)
        with timer.phase("gradient_step"):
            time.sleep(0.01)
    leaves = timer.leaves()
    assert sum(leaves.values()) == pytest.approx(timer.total_seconds["run"], rel=1e-9)


def test_resolve_walks_a_dotted_path():
    class Leaf:
        def sample(self):
            return 1

    class Root:
        replay_buffer = Leaf()

    owner, attribute = resolve(Root(), "replay_buffer.sample")
    assert isinstance(owner, Leaf) and attribute == "sample"


def _cell(algorithm="bbf", status="ok", phases=None, total=100.0):
    spec = CellSpec(
        algorithm=algorithm,
        label=algorithm.upper(),
        task="Jamesbond",
        seed=1,
        experiment=f"{algorithm}/atari100k",
        total_frames=100_000,
        profiled=True,
    )
    profile = {
        "phases": phases if phases is not None else {},
        "calls": dict.fromkeys(phases or {}, 3),
        "total_wall_seconds": total,
        "setup_seconds": 1.0,
        "agent_steps": 100_000,
        "sync_cuda": True,
        "environment": {
            "gpu_index": 3,
            "gpu_name": "RTX 5090",
            "torch": "2.11.0",
            "torchrl": "0.12.0",
            "loadavg": 1.5,
            "cpu_count": 384,
        },
    }
    return {"spec": spec, "status": status, "error": "", "profile": profile}


def test_rows_collapse_paths_and_shares_sum_to_one():
    cell = _cell(
        phases={
            "run": 5.0,
            "run/env_step": 20.0,
            "run/gradient_step": 40.0,
            "run/gradient_step/replay_sample": 10.0,
            "run/eval_final": 1.0,
            "run/eval_final/env_step": 14.0,
            "run/eval_final/rollout_inference": 10.0,
        }
    )
    rows = rows_for(cell, None, CFG)
    by_phase = {r["phase"]: r for r in rows}
    assert by_phase["eval_final"]["self_seconds"] == pytest.approx(25.0)
    assert sum(r["share"] for r in rows) == pytest.approx(1.0)


def test_aggregate_schema_and_ordering():
    rows = rows_for(
        _cell(algorithm="dreamerv3", phases={"run": 10.0, "run/env_step": 90.0}), None, CFG
    )
    rows += rows_for(
        _cell(algorithm="bbf", phases={"run": 50.0, "run/gradient_step": 50.0}), None, CFG
    )
    summary = aggregate(rows)
    assert list(summary.columns) == COLUMNS
    assert set(summary.status) <= STATUSES


def test_reaggregation_reads_a_measured_cell_back(tmp_path):
    spec = CellSpec(
        algorithm="bbf",
        label="BBF",
        task="Jamesbond",
        seed=1,
        experiment="bbf/atari100k",
        total_frames=100_000,
        profiled=True,
    )
    cell_dir = tmp_path / spec.cell_id
    cell_dir.mkdir(parents=True)
    (cell_dir / "profile_phases.json").write_text(
        json.dumps({"agent_steps": 100_000, "total_wall_seconds": 10.0, "phases": {"run": 10.0}})
    )
    cell = load_cell(spec, tmp_path)
    assert cell["status"] == "ok"
