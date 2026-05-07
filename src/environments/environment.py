"""Environment: thin config wrapper around the gymnasium env factory.

Holds construction kwargs and produces fresh ``TransformedEnv`` instances on
demand.  Never holds a live env itself — the trainer controls env lifecycle
by calling ``make_env()`` when it needs one.
"""
from __future__ import annotations

from torchrl.envs import EnvBase

from src.environments.factory import make_env


class Environment:
    """Wraps environment parameters and produces TorchRL envs.

    Args:
        name: gymnasium env name (e.g. ``"CartPole-v1"``).
        transforms: list of ``_target_``-keyed dicts; each is instantiated as
            a ``torchrl.envs.transforms`` object and composed on top of the
            base env. ``None`` or empty leaves the env un-transformed.
    """

    def __init__(
        self,
        name: str,
        transforms: list | None = None,
        **_: object,
    ) -> None:
        self._factory_kwargs: dict = {
            "name": name,
            "transforms": transforms,
        }

    def make_env(self, num_envs: int = 1, device: str = "cpu") -> EnvBase:
        return make_env(**self._factory_kwargs, num_envs=num_envs, device=device)
