"""Smoke tests: every shipped experiment composes and instantiates.

These don't run training (which would be slow/flaky) — they only verify
config composition and that ``hydra.utils.instantiate(cfg.trainer)``
returns a TorchRL ``Trainer``. Catches schema mismatches early.

Run with::

    pytest tests/test_smoke.py -v
"""

from __future__ import annotations

import pytest
import torchrl.trainers.algorithms.configs  # noqa: F401  (registers ConfigStore entries)
from torchrl.trainers import Trainer

import src.algorithms.reinforce.configs  # noqa: F401  (registers REINFORCETrainerConfig)
from tests.conftest import load_experiment_cfg


@pytest.mark.parametrize(
    "experiment",
    [
        "dqn/cartpole",
        "dqn/atari_pong",
        "ppo/pendulum",
        "ppo/halfcheetah",
        "reinforce/pendulum",
    ],
)
def test_experiment_composes(experiment: str) -> None:
    cfg = load_experiment_cfg(experiment)
    assert "trainer" in cfg
    assert cfg.trainer._target_


@pytest.mark.parametrize(
    "experiment",
    [
        "dqn/cartpole",
        "ppo/pendulum",
        "ppo/halfcheetah",
        "reinforce/pendulum",
    ],
)
def test_experiment_instantiates_trainer(experiment: str) -> None:
    """Build the trainer for each experiment that runs on CPU. Skips the
    Atari smoke since the env startup is heavyweight for unit tests."""
    import hydra

    cfg = load_experiment_cfg(
        experiment,
        extra_overrides=[
            "trainer.total_frames=10",
            "trainer.progress_bar=false",
            "collector.frames_per_batch=10",
            "logger.exp_name=test",  # avoid ${hydra:job.name} interpolation
        ],
    )
    trainer = hydra.utils.instantiate(cfg.trainer)
    assert isinstance(trainer, Trainer)
