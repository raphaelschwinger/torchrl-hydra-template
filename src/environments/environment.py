"""Environment: thin config wrapper around the env factory.

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
        name: gymnasium env id (e.g. ``"CartPole-v1"``); required for
            ``backend="gymnasium"``. Omitted for ``backend="dm_control"``,
            where the domain is read from ``task``.
        transforms: list of ``_target_``-keyed dicts; each is instantiated as
            a ``torchrl.envs.transforms`` object and composed on top of the
            base env. ``None`` or empty leaves the env un-transformed.
        gym_kwargs: optional extra kwargs for the base env. When
            ``gymnasium_wrappers`` is also given, TorchRL-specific keys
            (``from_pixels``, ``pixels_only``) are split off for
            ``GymWrapper``; the rest go to ``gymnasium.make``.
        gymnasium_wrappers: list of ``_target_``-keyed dicts for gymnasium
            wrappers applied between ``gymnasium.make`` and TorchRL's
            ``GymWrapper`` (e.g. ``gymnasium.wrappers.AtariPreprocessing``).
        gym_backend: optional gym backend name (e.g. ``"gymnasium"``).
        backend: ``"gymnasium"`` (default) or ``"dm_control"``.
        task: for ``backend="dm_control"``, the ``"<domain>-<task>"`` id
            (e.g. ``"cheetah-run"``); required for that backend.
        env_id: benchmark id reported to loggers, e.g. ``"Pong-v5"``. Kept
            separate from ``name`` because openrlbenchmark matches
            ``config.env_id`` exactly and its human-normalised-score table is
            keyed without the ``"ALE/"`` namespace prefix.
        action_repeat: environment frames consumed per agent step (frame skip
            or action repeat). Reporting metadata only — the actual repeat is
            implemented by the env stack. Used to derive the ``frames`` metric
            from ``global_step``.

    Raises:
        ValueError: if the keys required by ``backend`` are missing. Checked
            eagerly here — at config time — rather than deep inside
            ``gym.make`` when the trainer first builds an env.
    """

    def __init__(
        self,
        name: str | None = None,
        transforms: list | None = None,
        gym_kwargs: dict | None = None,
        gymnasium_wrappers: list | None = None,
        gym_backend: str | None = None,
        backend: str = "gymnasium",
        task: str | None = None,
        env_id: str | None = None,
        action_repeat: int = 1,
        **_: object,
    ) -> None:
        if backend == "gymnasium" and not name:
            raise ValueError(
                "environment.name is required for backend='gymnasium' "
                "(e.g. name: CartPole-v1)."
            )
        if backend == "dm_control" and not (name or task):
            raise ValueError(
                "environment.task is required for backend='dm_control' "
                "(e.g. task: cheetah-run)."
            )
        # Reporting metadata; deliberately kept out of _factory_kwargs so it
        # never reaches the env constructors.
        self.env_id = env_id or task or name
        self.action_repeat = int(action_repeat)
        self._factory_kwargs: dict = {
            "name": name,
            "transforms": transforms,
            "gym_kwargs": gym_kwargs,
            "gymnasium_wrappers": gymnasium_wrappers,
            "gym_backend": gym_backend,
            "backend": backend,
            "task": task,
        }

    def make_env(self, num_envs: int = 1, device: str = "cpu") -> EnvBase:
        return make_env(**self._factory_kwargs, num_envs=num_envs, device=device)
