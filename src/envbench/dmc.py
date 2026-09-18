"""DeepMind Control providers, all configured to one observation contract.

`dmc-state` is `cheetah-run` with state observations (position ++ velocity, 17
dims) and action repeat 2. State rather than pixels because neither EnvPool's
DMC nor MJX has a matching pixel path, and a pixel contract would shrink the DMC
comparison to two providers.

The fairness invariant that replaces "same observation" here is **simulated time
per agent step**. dm_control's cheetah has a 0.01 s control timestep, so action
repeat 2 means every provider must advance 0.02 s of physics per agent step.
Providers spell that differently (`frame_skip`, an explicit repeat loop,
`ctrl_dt`), and getting it wrong makes one provider look twice as fast for doing
half the work — so each provider reports the value and `check_contract` verifies it.
"""

from __future__ import annotations

import os
from importlib.metadata import PackageNotFoundError, version

import numpy as np

from src.envbench.base import Provider, continuous_action_pool, describe, signature
from src.envbench.spec import CellSpec, ProviderUnavailable

CONTROL_TIMESTEP = 0.01
ACTION_REPEAT = 2
SIM_SECONDS_PER_STEP = CONTROL_TIMESTEP * ACTION_REPEAT  # 0.02

TASKS = {
    "cheetah_run": {
        "domain": "cheetah",
        "task": "run",
        "envpool": "CheetahRun-v1",
        "gym": "dm_control/cheetah-run-v0",
        "playground": "CheetahRun",
    }
}


def _versions(*names: str) -> str:
    parts = []
    for name in names:
        try:
            parts.append(f"{name} {version(name)}")
        except PackageNotFoundError:
            parts.append(f"{name} missing")
    return "; ".join(parts)


def _headless() -> None:
    # MuJoCo must not try to open a GL context: these providers never render,
    # and on a headless node the probe itself would fail.
    os.environ.setdefault("MUJOCO_GL", "disabled")


# --------------------------------------------------------------------------- #
# dm_control, stepped serially — the no-vectorisation baseline
# --------------------------------------------------------------------------- #


class DmControlSerialProvider(Provider):
    def __init__(self, envs, actions: np.ndarray, **kwargs) -> None:
        super().__init__(**kwargs)
        self._envs = envs
        self._actions = actions
        self._index = 0

    def step(self) -> None:
        batch = self._actions[self._index % len(self._actions)]
        self._index += 1
        for i, env in enumerate(self._envs):
            action = batch[i]
            for _ in range(ACTION_REPEAT):
                step = env.step(action)
                if step.last():
                    env.reset()
                    break

    def close(self) -> None:
        for env in self._envs:
            env.close()


def build_dmc_native(spec: CellSpec) -> Provider:
    _headless()
    from dm_control import suite

    meta = TASKS[spec.task]
    envs = [
        suite.load(meta["domain"], meta["task"], task_kwargs={"random": spec.seed + i})
        for i in range(spec.num_envs)
    ]
    first = envs[0]
    step = first.reset()
    for env in envs[1:]:
        env.reset()

    obs = np.concatenate([np.ravel(v) for v in step.observation.values()])
    spec_action = first.action_spec()
    rng = np.random.default_rng(spec.seed)
    actions = continuous_action_pool(
        rng,
        spec.num_envs,
        int(spec_action.shape[0]),
        float(spec_action.minimum[0]),
        float(spec_action.maximum[0]),
        spec_action.dtype,
    )

    provider = DmControlSerialProvider(
        envs,
        actions,
        num_envs=spec.num_envs,
        versions=_versions("dm-control", "mujoco"),
        obs_signature=signature(obs.dtype, (obs.size,)),
        obs_shape=(int(obs.size),),
        sim_seconds_per_agent_step=first.control_timestep() * ACTION_REPEAT,
    )
    provider.parallelism = "none"
    return provider


# --------------------------------------------------------------------------- #
# Gymnasium vector envs over Shimmy's dm_control adapter
# --------------------------------------------------------------------------- #


def _shimmy_thunk(env_id: str):
    def thunk():
        import gymnasium as gym
        import shimmy
        from gymnasium.wrappers import FlattenObservation

        gym.register_envs(shimmy)
        env = gym.make(env_id)
        env = FlattenObservation(env)
        return _ActionRepeat(env, ACTION_REPEAT)

    return thunk


def build_gym_shimmy(spec: CellSpec) -> Provider:
    _headless()
    import gymnasium as gym

    asynchronous = bool(spec.options.get("asynchronous", False))
    thunk = _shimmy_thunk(TASKS[spec.task]["gym"])
    fns = [thunk for _ in range(spec.num_envs)]

    if asynchronous:
        env = gym.vector.AsyncVectorEnv(fns, shared_memory=True, context="forkserver")
    else:
        env = gym.vector.SyncVectorEnv(fns)

    obs, _ = env.reset(seed=spec.seed)
    action_space = env.single_action_space
    rng = np.random.default_rng(spec.seed)
    actions = continuous_action_pool(
        rng,
        spec.num_envs,
        int(action_space.shape[0]),
        float(action_space.low[0]),
        float(action_space.high[0]),
        action_space.dtype,
    )

    provider = _GymVectorProvider(
        env,
        actions,
        num_envs=spec.num_envs,
        versions=_versions("shimmy", "dm-control", "gymnasium"),
        obs_signature=describe(np.asarray(obs)),
        obs_shape=tuple(np.asarray(obs).shape[1:]),
        sim_seconds_per_agent_step=SIM_SECONDS_PER_STEP,
    )
    provider.parallelism = "process" if asynchronous else "none"
    return provider


class _GymVectorProvider(Provider):
    def __init__(self, env, actions: np.ndarray, **kwargs) -> None:
        super().__init__(**kwargs)
        self._env = env
        self._actions = actions
        self._index = 0

    def step(self) -> None:
        action = self._actions[self._index % len(self._actions)]
        self._index += 1
        self._env.step(action)

    def close(self) -> None:
        self._env.close()


def _make_action_repeat():
    import gymnasium as gym

    class ActionRepeat(gym.Wrapper):
        """Apply one action for `repeat` control steps, summing reward."""

        def __init__(self, env, repeat: int) -> None:
            super().__init__(env)
            self._repeat = repeat

        def step(self, action):
            total = 0.0
            terminated = truncated = False
            obs = info = None
            for _ in range(self._repeat):
                obs, reward, terminated, truncated, info = self.env.step(action)
                total += float(reward)
                if terminated or truncated:
                    break
            return obs, total, terminated, truncated, info

    return ActionRepeat


class _ActionRepeatProxy:
    """Defers building the gym.Wrapper subclass until gymnasium is importable."""

    _cls = None

    def __call__(self, env, repeat):
        if _ActionRepeatProxy._cls is None:
            _ActionRepeatProxy._cls = _make_action_repeat()
        return _ActionRepeatProxy._cls(env, repeat)


_ActionRepeat = _ActionRepeatProxy()


# --------------------------------------------------------------------------- #
# EnvPool
# --------------------------------------------------------------------------- #


class _EnvpoolDmProvider(Provider):
    def __init__(self, env, actions: np.ndarray, **kwargs) -> None:
        super().__init__(**kwargs)
        self._env = env
        self._actions = actions
        self._index = 0

    def step(self) -> None:
        action = self._actions[self._index % len(self._actions)]
        self._index += 1
        self._env.step(action)

    def close(self) -> None:
        close = getattr(self._env, "close", None)
        if close is not None:
            close()


def build_envpool_dmc(spec: CellSpec) -> Provider:
    import envpool

    env = envpool.make_dm(
        TASKS[spec.task]["envpool"],
        num_envs=spec.num_envs,
        batch_size=spec.num_envs,
        num_threads=0,
        frame_skip=ACTION_REPEAT,  # EnvPool's spelling of action repeat
        frame_stack=1,
        seed=spec.seed,
    )
    timestep = env.reset()
    observation = timestep.observation
    obs = np.concatenate(
        [
            np.asarray(observation.position).reshape(spec.num_envs, -1),
            np.asarray(observation.velocity).reshape(spec.num_envs, -1),
        ],
        axis=1,
    )
    action_spec = env.action_spec()
    rng = np.random.default_rng(spec.seed)
    actions = continuous_action_pool(
        rng,
        spec.num_envs,
        int(action_spec.shape[0]),
        float(np.min(action_spec.minimum)),
        float(np.max(action_spec.maximum)),
        np.float32,
    )

    provider = _EnvpoolDmProvider(
        env,
        actions,
        num_envs=spec.num_envs,
        versions=_versions("envpool"),
        obs_signature=describe(obs),
        obs_shape=(obs.shape[1],),
        sim_seconds_per_agent_step=CONTROL_TIMESTEP * ACTION_REPEAT,
    )
    provider.parallelism = "thread"
    return provider


# --------------------------------------------------------------------------- #
# MuJoCo Playground / MJX — GPU-resident, JAX
# --------------------------------------------------------------------------- #


class _MjxProvider(Provider):
    parallelism = "gpu-batched"
    device = "cuda"

    def __init__(self, jax_mod, step_fn, state, actions, **kwargs) -> None:
        super().__init__(**kwargs)
        self._jax = jax_mod
        self._step = step_fn
        self._state = state
        self._actions = actions
        self._index = 0

    def step(self) -> None:
        action = self._actions[self._index % self._actions.shape[0]]
        self._index += 1
        self._state = self._step(self._state, action)

    def step_blocking(self) -> None:
        self.step()
        self._jax.block_until_ready(self._state)

    def synchronize(self) -> None:
        self._jax.block_until_ready(self._state)


def build_mjx_playground(spec: CellSpec) -> Provider:
    try:
        import jax
        import jax.numpy as jnp
        from mujoco_playground import registry
    except ImportError as exc:  # pragma: no cover - depends on the venv in use
        raise ProviderUnavailable(str(exc)) from exc

    if not any(d.platform == "gpu" for d in jax.devices()):
        raise ProviderUnavailable("no JAX GPU device visible")

    meta = TASKS[spec.task]
    config = registry.get_default_config(meta["playground"])
    # Playground's default ctrl_dt equals dm_control's control timestep; scaling
    # it by the action repeat is how this provider spells "action repeat 2".
    config.ctrl_dt = CONTROL_TIMESTEP * ACTION_REPEAT
    env = registry.load(meta["playground"], config=config)

    reset = jax.jit(jax.vmap(env.reset))
    step = jax.jit(jax.vmap(env.step))

    keys = jax.random.split(jax.random.PRNGKey(spec.seed), spec.num_envs)
    state = jax.block_until_ready(reset(keys))

    rng = np.random.default_rng(spec.seed)
    pool = continuous_action_pool(rng, spec.num_envs, int(env.action_size), -1.0, 1.0, np.float32)
    actions = jnp.asarray(pool)  # keep the pool device-resident

    obs_leaf = np.asarray(jax.tree.leaves(state.obs)[0])

    return _MjxProvider(
        jax,
        step,
        state,
        actions,
        num_envs=spec.num_envs,
        versions=_versions("jax", "mujoco-mjx", "playground"),
        obs_signature=describe(obs_leaf),
        obs_shape=tuple(obs_leaf.shape[1:]),
        sim_seconds_per_agent_step=float(env.dt),
    )
