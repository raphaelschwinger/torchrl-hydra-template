"""Tests for the environment-throughput study."""

from __future__ import annotations

import sys

import pandas as pd
import pytest
from hydra import compose, initialize_config_dir

from scripts.envbench.task import COLUMNS, aggregate
from src.envbench import BUILDERS, STATUSES, CellSpec
from src.envbench.timing import EXPECTED_OBS_SHAPE, time_loop
from src.utils.paths import repo_root, results_dir, study_dir

CONFIG_DIR = repo_root() / "configs"


@pytest.fixture(scope="module")
def cfg():
    with initialize_config_dir(version_base="1.3", config_dir=str(CONFIG_DIR)):
        return compose(config_name="envbench")


def test_config_composes(cfg):
    assert cfg.study_name == "envthroughput"
    assert study_dir(cfg.study_name).is_dir()
    assert len(cfg.num_envs) > 0
    assert len(cfg.seeds) > 0


def test_every_provider_is_buildable_and_well_formed(cfg):
    for name, provider in cfg.providers.items():
        assert provider.suite in cfg.suites, f"{name}: unknown suite {provider.suite!r}"
        assert provider.label and provider.substrate, f"{name}: missing label/substrate"
        if provider.get("command", None):
            continue
        assert provider.kind in BUILDERS, f"{name}: unknown kind {provider.kind!r}"


def test_command_providers_point_at_a_script_that_exists(cfg):
    for name, provider in cfg.providers.items():
        command = provider.get("command", None)
        if not command:
            continue
        scripts = [part for part in command if str(part).endswith(".py")]
        assert scripts, f"{name}: command declares no script"
        for script in scripts:
            assert (repo_root() / str(script)).is_file(), f"{name}: missing {script}"


def test_every_suite_declares_a_known_contract(cfg):
    for name, suite in cfg.suites.items():
        assert suite.obs_contract in EXPECTED_OBS_SHAPE, f"{name}: unknown contract"
        assert suite.action_repeat >= 1


def test_registry_import_pulls_in_no_provider():
    before = set(sys.modules)
    import src.envbench.registry  # noqa: F401

    banned = {"gymnasium", "envpool", "dm_control", "jax", "stable_baselines3"}
    newly_loaded = set(sys.modules) - before
    pulled = banned & newly_loaded
    assert not pulled, f"registry import pulled in {sorted(pulled)}"


class _FakeProvider:
    device = "cpu"

    def __init__(self):
        self.steps = 0

    def step(self):
        self.steps += 1


def test_time_loop_excludes_warmup():
    provider = _FakeProvider()
    iters, wall = time_loop(provider, warmup_seconds=0.0, measure_seconds=0.0, min_iters=50)
    assert iters == 50
    assert provider.steps > iters
    assert wall > 0


def _row(provider="envpool_atari", *, seed=0, status="ok", sps=1000.0):
    spec = CellSpec(
        suite="atari",
        task="Pong",
        provider=provider,
        kind="envpool_atari",
        num_envs=16,
        seed=seed,
        action_repeat=4,
        obs_contract="atari-std",
        parallelism="thread",
        device="cpu",
    )
    row = spec.to_json()
    row.pop("options")
    row.pop("kind")
    row.update(
        label="EnvPool",
        substrate="C++ thread pool, async engine",
        status=status,
        sps=sps,
        fps=sps * 4,
        iters=100,
        wall_s=1.0,
        setup_s=0.5,
        loadavg=1.0,
        cpu_count=8,
        gpu_name="",
        provider_version="envpool 1.2.5",
        obs_signature="uint8(4,84,84)",
        sim_seconds_per_agent_step=0.0,
        error="" if status == "ok" else "boom",
    )
    return row


def test_aggregate_schema_and_seed_averaging():
    rows = [_row(seed=s, sps=1000.0 + 100 * s) for s in range(3)]
    summary = aggregate(rows)
    assert list(summary.columns) == COLUMNS
    assert len(summary) == 1
    assert summary.n_seeds.iloc[0] == 3
    assert summary.sps_mean.iloc[0] == pytest.approx(1100.0)


def test_a_partly_failed_cell_is_not_reported_as_ok():
    rows = [_row(seed=0), _row(seed=1), _row(seed=2, status="failed")]
    summary = aggregate(rows)
    assert summary.status.iloc[0] == "failed"


def test_committed_results_match_the_figure_and_config(cfg):
    path = results_dir(cfg.study_name) / f"{cfg.results_name}.parquet"
    if not path.is_file():
        pytest.skip("run the envbench sweep first")

    data = pd.read_parquet(path)
    assert list(data.columns) == COLUMNS
    assert set(data.status) <= STATUSES
    assert set(data.provider) <= set(cfg.providers)
