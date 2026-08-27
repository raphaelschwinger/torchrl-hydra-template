"""Atari providers, all configured to one observation contract.

`atari-std` is the standard ALE v5 preprocessing pipeline: frame-skip 4 with
max-pooling over the last two frames, greyscale, 84x84, 4-frame stack, sticky
actions p=0.25, up to 30 no-ops at reset, no reward clipping, no episodic life.

Every knob below is set **explicitly**, including where the value equals the
provider's own default. That is not redundancy: the defaults disagree across
providers (ale-py ships `reward_clipping=True` and `use_fire_reset=True`,
everyone ships `repeat_action_probability=0.0`), so a builder that relies on
defaults produces numbers that look reasonable and are not comparable. This is
the failure mode that would make the paper wrong rather than merely incomplete.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

import numpy as np

from src.envbench.base import Provider, describe, discrete_action_pool
from src.envbench.spec import CellSpec

FRAME_SKIP = 4
SCREEN = 84
STACK = 4
STICKY = 0.25
NOOP_MAX = 30
MAX_EPISODE_STEPS = 27_000  # agent steps; 108_000 emulator frames

# Per-provider spelling of the same game.
GAMES = {
    "Pong": {"rom": "pong", "envpool": "Pong-v5", "gym": "ALE/Pong-v5"},
    "Breakout": {"rom": "breakout", "envpool": "Breakout-v5", "gym": "ALE/Breakout-v5"},
}


def _versions(*names: str) -> str:
    parts = []
    for name in names:
        try:
            parts.append(f"{name} {version(name)}")
        except PackageNotFoundError:
            parts.append(f"{name} missing")
    return "; ".join(parts)


class _ArrayStepProvider(Provider):
    """Providers whose `step` takes a batched action array and returns arrays."""

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


# --------------------------------------------------------------------------- #
# Gymnasium vector envs (the baseline every researcher starts from)
# --------------------------------------------------------------------------- #


def _gym_thunk(env_id: str):
    def thunk():
        import ale_py
        import gymnasium as gym
        from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation

        gym.register_envs(ale_py)
        env = gym.make(env_id, frameskip=1, repeat_action_probability=STICKY)
        env = AtariPreprocessing(
            env,
            noop_max=NOOP_MAX,
            frame_skip=FRAME_SKIP,
            screen_size=SCREEN,
            terminal_on_life_loss=False,
            grayscale_obs=True,
            scale_obs=False,
        )
        return FrameStackObservation(env, stack_size=STACK)

    return thunk


def build_gym_vector(spec: CellSpec) -> Provider:
    import gymnasium as gym

    asynchronous = bool(spec.options.get("asynchronous", False))
    thunk = _gym_thunk(GAMES[spec.task]["gym"])
    fns = [thunk for _ in range(spec.num_envs)]

    if asynchronous:
        # forkserver, not fork: forking a process that already owns a thread
        # pool (ALE, BLAS) is undefined behaviour and deadlocks intermittently.
        env = gym.vector.AsyncVectorEnv(fns, shared_memory=True, context="forkserver")
    else:
        env = gym.vector.SyncVectorEnv(fns)

    obs, _ = env.reset(seed=spec.seed)
    rng = np.random.default_rng(spec.seed)
    actions = discrete_action_pool(rng, spec.num_envs, int(env.single_action_space.n))

    provider = _ArrayStepProvider(
        env,
        actions,
        num_envs=spec.num_envs,
        versions=_versions("gymnasium", "ale-py"),
        obs_signature=describe(np.asarray(obs)),
        obs_shape=tuple(np.asarray(obs).shape[1:]),
    )
    provider.parallelism = "process" if asynchronous else "none"
    return provider


# --------------------------------------------------------------------------- #
# ale-py's native C++ vector env
# --------------------------------------------------------------------------- #


def build_ale_native(spec: CellSpec) -> Provider:
    import inspect

    from ale_py.vector_env import AtariVectorEnv

    wanted = dict(
        game=GAMES[spec.task]["rom"],
        num_envs=spec.num_envs,
        batch_size=spec.num_envs,  # == num_envs is synchronous mode
        num_threads=0,  # 0 -> one thread per env, ALE's own default policy
        frameskip=FRAME_SKIP,
        maxpool=True,
        grayscale=True,
        stack_num=STACK,
        img_height=SCREEN,
        img_width=SCREEN,
        noop_max=NOOP_MAX,
        repeat_action_probability=STICKY,
        episodic_life=False,
        reward_clipping=False,
        use_fire_reset=False,
        max_num_frames_per_episode=MAX_EPISODE_STEPS * FRAME_SKIP,
    )

    # The constructor's keyword names moved between ale-py 0.10, 0.11 and 0.12.
    # Fail naming the missing knob rather than silently accepting its default.
    from src.envbench.spec import ContractError

    accepted = set(inspect.signature(AtariVectorEnv.__init__).parameters)
    missing = sorted(set(wanted) - accepted)
    if missing:
        raise ContractError(f"ale-py {version('ale-py')} does not accept {missing}")

    env = AtariVectorEnv(**wanted)
    obs, _ = env.reset(seed=spec.seed)
    rng = np.random.default_rng(spec.seed)
    actions = discrete_action_pool(rng, spec.num_envs, int(env.single_action_space.n))

    provider = _ArrayStepProvider(
        env,
        actions,
        num_envs=spec.num_envs,
        versions=_versions("ale-py"),
        obs_signature=describe(np.asarray(obs)),
        obs_shape=tuple(np.asarray(obs).shape[1:]),
    )
    provider.parallelism = "thread"
    return provider


# --------------------------------------------------------------------------- #
# EnvPool
# --------------------------------------------------------------------------- #


def build_envpool_atari(spec: CellSpec) -> Provider:
    import envpool

    env = envpool.make_gymnasium(
        GAMES[spec.task]["envpool"],
        num_envs=spec.num_envs,
        batch_size=spec.num_envs,  # synchronous mode
        num_threads=0,
        stack_num=STACK,
        frame_skip=FRAME_SKIP,
        img_height=SCREEN,
        img_width=SCREEN,
        gray_scale=True,
        noop_max=NOOP_MAX,
        repeat_action_probability=STICKY,
        episodic_life=False,
        reward_clip=False,
        use_fire_reset=False,
        max_episode_steps=MAX_EPISODE_STEPS,
        seed=spec.seed,
    )
    obs, _ = env.reset()
    rng = np.random.default_rng(spec.seed)
    actions = discrete_action_pool(rng, spec.num_envs, int(env.action_space.n))

    provider = _ArrayStepProvider(
        env,
        actions,
        num_envs=spec.num_envs,
        versions=_versions("envpool"),
        obs_signature=describe(np.asarray(obs)),
        obs_shape=tuple(np.asarray(obs).shape[1:]),
    )
    provider.parallelism = "thread"
    return provider


# --------------------------------------------------------------------------- #
# Stable-Baselines3 vec envs
# --------------------------------------------------------------------------- #


def _sb3_thunk(env_id: str):
    def thunk():
        import ale_py
        import gymnasium as gym
        from stable_baselines3.common.atari_wrappers import AtariWrapper

        gym.register_envs(ale_py)
        # Stickiness comes from AtariWrapper, so the base env must not add it too.
        env = gym.make(env_id, frameskip=1, repeat_action_probability=0.0)
        return AtariWrapper(
            env,
            noop_max=NOOP_MAX,
            frame_skip=FRAME_SKIP,
            screen_size=SCREEN,
            terminal_on_life_loss=False,
            clip_reward=False,
            action_repeat_probability=STICKY,
        )

    return thunk


def build_sb3_vector(spec: CellSpec) -> Provider:
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

    subproc = bool(spec.options.get("subproc", False))
    thunk = _sb3_thunk(GAMES[spec.task]["gym"])
    fns = [thunk for _ in range(spec.num_envs)]

    # Deliberately not `make_atari_env`: its defaults re-introduce episodic life,
    # reward clipping and a different no-op count.
    venv = SubprocVecEnv(fns, start_method="forkserver") if subproc else DummyVecEnv(fns)
    venv = VecFrameStack(venv, n_stack=STACK)

    venv.seed(spec.seed)
    obs = venv.reset()
    rng = np.random.default_rng(spec.seed)
    actions = discrete_action_pool(rng, spec.num_envs, int(venv.action_space.n))

    obs = np.asarray(obs)
    # SB3's VecFrameStack stacks on the last axis (N,84,84,4); the contract is
    # channels-first, so report the transposed shape. The pixels are identical.
    signature = f"{obs.dtype}({obs.shape[3]},{obs.shape[1]},{obs.shape[2]})"

    provider = _ArrayStepProvider(
        venv,
        actions,
        num_envs=spec.num_envs,
        versions=_versions("stable-baselines3", "gymnasium", "ale-py"),
        obs_signature=signature,
        obs_shape=(obs.shape[3], obs.shape[1], obs.shape[2]),
    )
    provider.parallelism = "process" if subproc else "none"
    return provider
