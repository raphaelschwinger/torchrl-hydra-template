"""Unit tests for environment config resolution (no env is actually built)."""
from __future__ import annotations

import pytest

from src.environments import Environment
from src.environments.factory import _split_dmc_id


class TestSplitDmcId:
    """``task: "<domain>-<task>"`` is the canonical dm_control id form."""

    @pytest.mark.parametrize(
        ("task", "expected"),
        [
            ("cheetah-run", ("cheetah", "run")),
            ("walker-walk", ("walker", "walk")),
            # dm_control uses underscores inside domain and task names, so the
            # FIRST hyphen is always the separator.
            ("ball_in_cup-catch", ("ball_in_cup", "catch")),
            ("finger-turn_hard", ("finger", "turn_hard")),
            ("point_mass-easy", ("point_mass", "easy")),
        ],
    )
    def test_splits_on_first_hyphen(self, task, expected):
        assert _split_dmc_id(None, task) == expected

    def test_explicit_domain_and_task_still_supported(self):
        assert _split_dmc_id("cheetah", "run") == ("cheetah", "run")

    @pytest.mark.parametrize("task", [None, "", "cheetah"])
    def test_rejects_missing_or_unsplittable_task(self, task):
        with pytest.raises(ValueError, match="task: <domain>-<task>"):
            _split_dmc_id(None, task)

    def test_rejects_domain_without_task(self):
        with pytest.raises(ValueError, match="requires `task`"):
            _split_dmc_id("cheetah", None)


class TestEnvironmentValidation:
    """Missing keys fail at config time, not deep inside gym.make."""

    def test_gymnasium_requires_name(self):
        with pytest.raises(ValueError, match="environment.name is required"):
            Environment(backend="gymnasium", task="cheetah-run")

    def test_dm_control_requires_task(self):
        with pytest.raises(ValueError, match="environment.task is required"):
            Environment(backend="dm_control")

    def test_dm_control_accepts_task_only(self):
        env = Environment(backend="dm_control", task="cheetah-run")
        assert env._factory_kwargs["task"] == "cheetah-run"
        assert env._factory_kwargs["name"] is None

    def test_gymnasium_accepts_name_only(self):
        env = Environment(name="CartPole-v1")
        assert env._factory_kwargs["name"] == "CartPole-v1"
