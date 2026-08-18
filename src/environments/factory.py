"""Environment factory for gymnasium- and dm_control-backed TorchRL envs.

Builds a (possibly vectorised) ``TransformedEnv`` from a small parameter
set and an explicit list of transform descriptors.

Each transform descriptor is a dict with a ``_target_`` key (a dotted path
to a ``torchrl.envs.transforms`` class) plus its constructor kwargs.
Transforms are instantiated fresh per ``make_env()`` call so each env has
independent transform state.
"""
from __future__ import annotations

import importlib
import os
from contextlib import nullcontext
from functools import partial
from typing import Sequence

# kwargs that belong on GymWrapper/GymEnv, not on gymnasium.make
_TORCHRL_ONLY = {"from_pixels", "pixels_only", "categorical_action_encoding"}
# TorchRL's name → gymnasium's name for gym.make kwargs
_GYM_RENAME = {"frame_skip": "frameskip"}


def make_env(
    name: str | None = None,
    num_envs: int = 1,
    device: str = "cpu",
    transforms: list | None = None,
    gym_kwargs: dict | None = None,
    gymnasium_wrappers: list | None = None,
    gym_backend: str | None = None,
    backend: str = "gymnasium",
    task: str | None = None,
    **_: object,
):
    """Build a (possibly vectorised) ``TransformedEnv``.

    Args:
        name: gymnasium env id (e.g. ``"CartPole-v1"``). Not needed for
            ``backend="dm_control"``, where the domain comes from ``task``.
        num_envs: number of parallel envs (>1 -> ``ParallelEnv``; workers
            always run on CPU because CUDA contexts cannot survive ``fork``).
        device: target device string.
        transforms: list of ``_target_``-keyed dicts to apply on top of the
            base env. ``None`` or empty -> bare base env.
        gym_kwargs: extra kwargs for the base env. When ``gymnasium_wrappers``
            is provided, TorchRL-specific keys (``from_pixels``,
            ``pixels_only``) are separated out and passed to ``GymWrapper``
            while the rest go to ``gymnasium.make`` (``frame_skip`` is
            translated to ``frameskip``). Without ``gymnasium_wrappers``,
            all kwargs are forwarded to ``GymEnv`` unchanged.
        gymnasium_wrappers: list of ``_target_``-keyed dicts for gymnasium
            wrappers applied between ``gymnasium.make`` and ``GymWrapper``.
            Use this for wrappers that must see the raw gymnasium env (e.g.
            ``gymnasium.wrappers.AtariPreprocessing``).
        gym_backend: optional gym backend name for ``set_gym_backend``
            (e.g. ``"gymnasium"``); if ``None`` torchrl picks the default.
        backend: ``"gymnasium"`` (default) or ``"dm_control"``.
        task: for ``backend="dm_control"``, the ``"<domain>-<task>"`` id
            (e.g. ``"cheetah-run"``, ``"finger-turn_hard"``); required unless
            ``name`` carries the domain and ``task`` the bare task name.
            Ignored for ``backend="gymnasium"``.
    """
    worker_device = "cpu" if num_envs > 1 else device
    env_fn = _select_env_fn(
        backend,
        name,
        task,
        transforms,
        worker_device,
        gym_kwargs,
        gymnasium_wrappers,
        gym_backend,
    )

    if num_envs > 1:
        from torchrl.envs import ParallelEnv

        return ParallelEnv(num_envs, env_fn, mp_start_method="spawn")
    return env_fn()


def _select_env_fn(
    backend, name, task, transforms, device, gym_kwargs, gymnasium_wrappers, gym_backend
):
    """Return a no-arg env constructor for the requested backend."""
    if backend == "dm_control":
        return partial(
            _make_dmc_env, name=name, task=task, transforms=transforms, device=device
        )
    if backend == "gymnasium":
        return partial(
            _make_gymnasium_env,
            name=name,
            transforms=transforms,
            device=device,
            gym_kwargs=gym_kwargs,
            gymnasium_wrappers=gymnasium_wrappers,
            gym_backend=gym_backend,
        )
    raise ValueError(f"Unknown environment backend: {backend!r}")


def _instantiate_transform(cfg: dict):
    """Instantiate a transform from a ``_target_``-keyed dict (no Hydra runtime)."""
    cfg = dict(cfg)  # copy — don't mutate the caller
    target = cfg.pop("_target_")
    module_path, class_name = target.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(**cfg)


def _apply_transforms(base_env, transforms: list | None):
    from torchrl.envs import TransformedEnv
    from torchrl.envs.transforms import Compose

    if not transforms:
        return base_env

    transform_objects = [_instantiate_transform(t) for t in transforms]
    return TransformedEnv(base_env, Compose(*transform_objects))


def _split_dmc_id(name: str | None, task: str | None) -> tuple[str, str]:
    """Resolve a dm_control ``(domain, task)`` pair from the env config.

    The canonical form is a single ``task: "<domain>-<task>"`` id, split on
    the *first* hyphen — dm_control uses underscores inside its own names
    (``ball_in_cup-catch``, ``finger-turn_hard``, ``point_mass-easy``), so
    the first hyphen is always the separator. An explicit ``name`` (domain)
    plus a bare ``task`` is also accepted.
    """
    if name:
        if not task:
            raise ValueError(
                "backend='dm_control' with an explicit `name` (domain) also "
                "requires `task` (e.g. name: cheetah, task: run)."
            )
        return name, task
    if not task or "-" not in task:
        raise ValueError(
            "backend='dm_control' requires `task: <domain>-<task>` "
            f"(e.g. task: cheetah-run). Got name={name!r}, task={task!r}."
        )
    domain, subtask = task.split("-", 1)
    return domain, subtask


def _make_dmc_env(
    name: str | None,
    task: str | None,
    transforms: list | None,
    device: str,
):
    # dm_control initialises a renderer at import time; default to headless
    # (no rendering) unless the user configured a GL backend themselves.
    os.environ.setdefault("MUJOCO_GL", "disabled")
    from torchrl.envs import DMControlEnv

    domain, subtask = _split_dmc_id(name, task)
    base_env = DMControlEnv(domain, subtask, device=device)
    return _apply_transforms(base_env, transforms)


def _instantiate_gymnasium_wrapper(env, cfg: dict):
    """Instantiate a gymnasium wrapper from a ``_target_``-keyed dict."""
    cfg = dict(cfg)
    target = cfg.pop("_target_")
    module_path, class_name = target.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_path), class_name)
    return cls(env, **cfg)


def _make_gymnasium_env(
    name: str,
    transforms: list | None,
    device: str,
    gym_kwargs: dict | None = None,
    gymnasium_wrappers: list | None = None,
    gym_backend: str | None = None,
):
    from torchrl.envs import GymEnv, GymWrapper

    backend_ctx = nullcontext()
    if gym_backend is not None:
        from torchrl.envs import set_gym_backend
        backend_ctx = set_gym_backend(gym_backend)

    with backend_ctx:
        if gymnasium_wrappers:
            import gymnasium as gym
            try:
                import ale_py
                gym.register_envs(ale_py)
            except ImportError:
                pass

            torchrl_kwargs = {
                k: v for k, v in (gym_kwargs or {}).items() if k in _TORCHRL_ONLY
            }
            make_kwargs = {
                _GYM_RENAME.get(k, k): v
                for k, v in (gym_kwargs or {}).items()
                if k not in _TORCHRL_ONLY
            }
            gym_env = gym.make(name, **make_kwargs)
            for wrapper_cfg in gymnasium_wrappers:
                gym_env = _instantiate_gymnasium_wrapper(gym_env, wrapper_cfg)
            base_env = GymWrapper(gym_env, device=device, **torchrl_kwargs)
        else:
            base_env = GymEnv(name, device=device, **(gym_kwargs or {}))

    return _apply_transforms(base_env, transforms)
