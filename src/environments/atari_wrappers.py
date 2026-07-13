"""Atari preprocessing wrappers exposed as TorchRL transforms."""
from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from tensordict import TensorDictBase
from torchrl.envs.transforms import Transform


class MaxAndSkipEnv(gym.Wrapper):
    """Repeat actions and max-pool the final two raw Atari frames."""

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        self.skip = skip
        self._obs_buffer: deque[np.ndarray] = deque(maxlen=2)

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        self._obs_buffer.clear()
        return self.env.reset(**kwargs)

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        total_reward = 0.0
        observation = None
        terminated = truncated = False
        info: dict[str, Any] = {}
        for _ in range(self.skip):
            observation, reward, terminated, truncated, info = self.env.step(action)
            self._obs_buffer.append(observation)
            total_reward += float(reward)
            if terminated or truncated:
                break
        if len(self._obs_buffer) == 2:
            observation = np.maximum(self._obs_buffer[0], self._obs_buffer[1])
        return observation, total_reward, terminated, truncated, info


class EpisodicLifeEnv(gym.Wrapper):
    """Expose life loss as terminal while preserving the underlying game."""

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def _lives(self) -> int:
        ale = getattr(self.unwrapped, "ale", None)
        return int(ale.lives()) if ale is not None else 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        if self.was_real_done:
            observation, info = self.env.reset(seed=seed, options=options)
        else:
            # Match BBF-pytorch: advance from the life-loss screen with one
            # NOOP agent action (including the inner action repeat), without
            # resetting the actual ALE game.
            observation, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                observation, info = self.env.reset(seed=seed, options=options)
        self.lives = self._lives()
        return observation, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.was_real_done = bool(terminated or truncated)
        lives = self._lives()
        life_lost = lives < self.lives and lives > 0
        self.lives = lives
        if life_lost:
            terminated = True
        return observation, reward, terminated, truncated, info


def wrap_atari(
    env: gym.Env,
    *,
    frame_skip: int = 4,
    terminal_on_life_loss: bool,
) -> gym.Env:
    """Apply the BBF-pytorch Atari wrapper order."""
    env = MaxAndSkipEnv(env, skip=frame_skip)
    if terminal_on_life_loss:
        env = EpisodicLifeEnv(env)
    return env


class AtariPreprocessingTransform(Transform):
    """Install Atari Gym wrappers from the regular TorchRL transform list.

    Max-and-skip and classic episodic-life reset semantics need access to the
    raw Gymnasium/ALE env. This transform keeps that special handling out of the
    generic environment factory while still making the behavior explicit and
    composable in Hydra's ``transforms`` list.
    """

    def __init__(
        self,
        *,
        frame_skip: int = 4,
        terminal_on_life_loss: bool,
    ) -> None:
        super().__init__()
        self.frame_skip = frame_skip
        self.terminal_on_life_loss = terminal_on_life_loss
        self._is_wrapped = False

    def _ensure_wrapped(self) -> None:
        if self._is_wrapped:
            return
        if self.parent is None:
            raise RuntimeError(
                f"{type(self).__name__} must be attached to a TransformedEnv."
            )

        base_env = getattr(self.parent, "base_env", self.parent)
        gym_env = getattr(base_env, "_env", None)
        if gym_env is None:
            raise RuntimeError(
                f"{type(self).__name__} requires a TorchRL GymEnv/GymWrapper "
                "with a raw Gymnasium env stored on `_env`."
            )

        base_env._env = wrap_atari(
            gym_env,
            frame_skip=self.frame_skip,
            terminal_on_life_loss=self.terminal_on_life_loss,
        )
        if isinstance(base_env._env, EpisodicLifeEnv):
            base_env._env.lives = base_env._env._lives()
        self._is_wrapped = True

    def _reset(
        self,
        tensordict: TensorDictBase,
        tensordict_reset: TensorDictBase,
    ) -> TensorDictBase:
        self._ensure_wrapped()
        return tensordict_reset

    def _step(
        self,
        tensordict: TensorDictBase,
        next_tensordict: TensorDictBase,
    ) -> TensorDictBase:
        self._ensure_wrapped()
        return next_tensordict
