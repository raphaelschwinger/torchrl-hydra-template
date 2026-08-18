"""Tests for scripts/update_algo_results.py (no W&B network calls)."""
from __future__ import annotations

from pathlib import Path

import pytest

from scripts.update_algo_results import (
    ExperimentSpec,
    ResultRow,
    build_table,
    format_project_not_found_error,
    get_eval_return,
    infer_experiment_config,
    load_experiment_registry,
    replace_results_table,
    resolve_entity,
    row_to_markdown,
)


@pytest.fixture
def registry() -> list[ExperimentSpec]:
    return load_experiment_registry()


def test_load_experiment_registry_includes_known_experiments(registry):
    paths = {spec.path for spec in registry}
    assert "dqn/gym" in paths
    assert "dqn/ale" in paths
    assert "ddpg/gym" in paths
    assert "a2c/gym" in paths
    assert "rainbow/atari100k" in paths


def test_infer_experiment_config_cartpole(registry):
    config = {
        "algorithm": {"_target_": "src.algorithms.dqn.DQNAlgorithm"},
        "environment": {"name": "CartPole-v1"},
        "trainer": {"seed": 42, "total_frames": 500_100},
    }
    assert infer_experiment_config(config, registry) == "experiment=dqn/gym"


def test_infer_experiment_config_pong(registry):
    config = {
        "algorithm": {
            "_target_": "src.algorithms.dqn.DQNAlgorithm",
            "obs_key": "pixels",
        },
        "environment": {"name": "ALE/Pong-v5"},
        "trainer": {"seed": 42, "total_frames": 40_000_100},
    }
    assert infer_experiment_config(config, registry) == "experiment=dqn/ale"


def test_infer_experiment_config_der_jamesbond(registry):
    config = {
        "algorithm": {
            "_target_": "src.algorithms.rainbow.RainbowAlgorithm",
            "obs_key": "pixels",
            "encoder_type": "data_efficient",
        },
        "environment": {
            "name": "ALE/Jamesbond-v5",
            "gymnasium_wrappers": [{"_target_": "src.environments.atari_wrappers.MaxAndSkipEnv"}],
        },
        "trainer": {"seed": 1, "total_frames": 100_000},
    }
    assert infer_experiment_config(config, registry) == "experiment=rainbow/atari100k"


def test_infer_experiment_config_halfcheetah_ddpg(registry):
    config = {
        "algorithm": {"_target_": "src.algorithms.ddpg.DDPGAlgorithm"},
        "environment": {"name": "HalfCheetah-v4"},
        "trainer": {"seed": 42, "total_frames": 1_000_000},
    }
    assert infer_experiment_config(config, registry) == "experiment=ddpg/gym"


def test_infer_experiment_config_emits_task_override(registry):
    """A game the experiment does not default to still resolves, as an override."""
    config = {
        "algorithm": {
            "_target_": "src.algorithms.dqn.DQNAlgorithm",
            "obs_key": "pixels",
        },
        "environment": {"name": "ALE/Breakout-v5"},
        "trainer": {"seed": 42, "total_frames": 40_000_100},
    }
    assert infer_experiment_config(config, registry) == (
        "experiment=dqn/ale environment.task=Breakout"
    )


def test_infer_experiment_config_matches_legacy_dmc_shape(registry):
    """Historical runs logged dm_control as name: cheetah + task: run."""
    config = {
        "algorithm": {"_target_": "src.algorithms.tdmpc2.TDMPC2Algorithm"},
        "environment": {"backend": "dm_control", "name": "cheetah", "task": "run"},
        "trainer": {"seed": 1, "total_frames": 1_000_000},
    }
    assert infer_experiment_config(config, registry) == "experiment=tdmpc2/dmc"


def test_infer_experiment_config_matches_legacy_algorithm_target(registry):
    """Runs logged before the dreamer package re-export shortened `_target_`."""
    config = {
        "algorithm": {
            "_target_": "src.algorithms.dreamer.dreamer.DreamerAlgorithm",  # old path
            "dreamer_config": {"_target_": "src.algorithms.dreamer.model.DreamerV3"},
        },
        "environment": {"name": "ALE/Hero-v5"},
        "trainer": {"seed": 42, "total_frames": 110_000},
    }
    assert infer_experiment_config(config, registry) == "experiment=dreamer/atari100k"


def test_infer_experiment_config_dreamer_variant(registry):
    """R2Dreamer runs resolve to the dreamer experiment plus an algorithm override."""
    config = {
        "algorithm": {
            "_target_": "src.algorithms.dreamer.DreamerAlgorithm",
            "dreamer_config": {"_target_": "src.algorithms.dreamer.model.R2Dreamer"},
        },
        "environment": {
            "name": "ALE/Hero-v5",
            "gymnasium_wrappers": [{"_target_": "gymnasium.wrappers.AtariPreprocessing"}],
        },
        "trainer": {"seed": 42, "total_frames": 110_000},
    }
    assert infer_experiment_config(config, registry) == (
        "experiment=dreamer/atari100k algorithm=r2dreamer"
    )


def _bbf_run_config(replay_ratio: int) -> dict:
    return {
        "algorithm": {
            "_target_": "src.algorithms.bbf.BBFAlgorithm",
            "obs_key": "pixels",
            "replay_ratio": replay_ratio,
            "replay_capacity": 105_000,
        },
        "environment": {"name": "ALE/Jamesbond-v5"},
        "trainer": {"seed": 1, "total_frames": 100_000},
    }


def test_infer_experiment_config_bbf_rr2(registry):
    assert infer_experiment_config(_bbf_run_config(2), registry) == (
        "experiment=bbf/atari100k"
    )


def test_infer_experiment_config_bbf_rr8(registry):
    """RR2 and RR8 share an identity and a benchmark; only scalars separate them."""
    assert infer_experiment_config(_bbf_run_config(8), registry) == (
        "experiment=bbf/atari100k_rr8"
    )


def test_build_table_empty():
    table = build_table([])
    assert "No finished runs tagged ``template`` yet" in table
    assert "| Run | Environment |" in table


def test_row_to_markdown():
    row = ResultRow(
        run_name="dqn_cartpole_2025-01-01",
        run_url="https://wandb.ai/LatentLab/torchrl-hydra-template/runs/abc123",
        environment="CartPole-v1",
        config="experiment=dqn/gym",
        seed=42,
        frames=500_100,
        eval_return="500.0",
        notes="—",
    )
    md = row_to_markdown(row)
    assert "[dqn_cartpole_2025-01-01](" in md
    assert "`experiment=dqn/gym`" in md
    assert "500,100" in md


def test_replace_results_table():
    readme = Path("src/algorithms/dqn/README.md").read_text(encoding="utf-8")
    new_table = build_table([])
    updated = replace_results_table(readme, new_table)
    assert updated != readme
    assert "No finished runs tagged ``template`` yet" in updated
    assert "## Experimental results" in updated


class _FakeRun:
    """Minimal stand-in for a wandb Run (no network)."""

    def __init__(self, summary: dict, history: dict[str, list] | None = None) -> None:
        self.summary = summary
        self._history = history or {}

    def history(self, keys, pandas=False):
        key = keys[0]
        if key not in self._history:
            raise ValueError(f"no such key: {key}")
        return [{key: v} for v in self._history[key]]


def test_eval_return_prefers_trainer_summary():
    run = _FakeRun(
        {"eval/final_return_mean": 1125.0, "eval/return_mean": 999.0},
        {"charts/episodic_return": [1.0, 2.0]},
    )
    assert get_eval_return(run) == (1125.0, "")


def test_eval_return_averages_last_100_canonical_points():
    """openrlbenchmark's own rule, so the README column matches its tables."""
    run = _FakeRun({}, {"charts/episodic_return": [0.0] * 500 + [10.0] * 100})
    value, note = get_eval_return(run)
    assert value == pytest.approx(10.0)
    assert note == ""


def test_eval_return_ignores_nan_summary():
    run = _FakeRun(
        {"eval/final_return_mean": float("nan")},
        {"charts/episodic_return": [5.0, 7.0]},
    )
    assert get_eval_return(run) == (pytest.approx(6.0), "")


def test_eval_return_falls_back_to_legacy_keys_with_a_note():
    """Runs predating the unified protocol still populate the table, labelled."""
    run = _FakeRun({"eval/score_mean_last10pct": 42.0})
    assert get_eval_return(run) == (42.0, "legacy eval/score_mean_last10pct")

    run = _FakeRun({"eval/return_mean": 7.0})
    assert get_eval_return(run) == (7.0, "legacy eval/return_mean")

    run = _FakeRun({}, {"train/episode_reward": [1.0, 9.0, 3.0]})
    assert get_eval_return(run) == (9.0, "legacy best train/episode_reward")


def test_eval_return_missing_everywhere():
    assert get_eval_return(_FakeRun({})) == (None, "")


def test_resolve_entity_prefers_explicit():
    class FakeApi:
        default_entity = "login-default"

    assert resolve_entity(FakeApi(), "explicit") == "explicit"


def test_resolve_entity_uses_env(monkeypatch):
    class FakeApi:
        default_entity = "login-default"

    monkeypatch.setenv("WANDB_ENTITY", "from-env")
    assert resolve_entity(FakeApi(), None) == "from-env"


def test_format_project_not_found_error_lists_projects():
    msg = format_project_not_found_error(
        entity="rschwinger",
        project="torchrl-hydra-template",
        available_projects=["introdrl", "PretrainWM"],
    )
    assert "rschwinger/torchrl-hydra-template" in msg
    assert "introdrl" in msg
    assert "LatentLab" in msg
