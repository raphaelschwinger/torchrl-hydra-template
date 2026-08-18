"""Shared pytest fixtures and helpers for the test suite."""
from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import DictConfig


CONFIGS_DIR = str(Path(__file__).parent.parent / "configs")


@pytest.fixture(scope="session")
def config_dir() -> str:
    return CONFIGS_DIR


def load_experiment_cfg(
    experiment: str,
    extra_overrides: list[str] | None = None,
) -> DictConfig:
    """Load a fully composed Hydra config for a given experiment.

    Uses hydra's compose API so no Hydra app is initialized; safe to call
    from pytest without side effects on the working directory.

    Args:
        experiment: experiment path, e.g. "reinforce/cartpole"
        extra_overrides: additional CLI-style overrides, e.g. ["logger=[]"]

    Returns:
        fully composed DictConfig
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    # Reset any previous Hydra state
    GlobalHydra.instance().clear()

    overrides = [f"experiment={experiment}", *(extra_overrides or [])]
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base="1.3"):
        cfg = compose(config_name="train", overrides=overrides)

    return cfg
