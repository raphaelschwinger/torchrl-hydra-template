# MaxAndSkipEnv / EpisodicLifeEnv adapted from https://github.com/DLR-RM/stable-baselines3
# (stable_baselines3/common/atari_wrappers.py), MIT license. Changes: life-loss
# reset advances with one NOOP step instead of FireReset.
"""Atari preprocessing: gym wrappers and TorchRL transforms."""
from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDictBase
from torchrl.envs.transforms import Transform


class NoopResetEnv(gym.Wrapper):
    """Start real games from a randomized state using 1..noop_max NOOPs."""

    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[Any, dict[str, Any]]:
        observation, info = self.env.reset(seed=seed, options=options)
        noops = (
            int(self.np_random.integers(1, self.noop_max + 1))
            if self.noop_max > 0
            else 0
        )
        for _ in range(noops):
            observation, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                observation, info = self.env.reset(options=options)
        return observation, info


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
    """Expose life loss as terminal while preserving the underlying game.

    This must stay a ``gym.Wrapper``; it cannot be ported to a TorchRL
    ``Transform`` without weakening its semantics.

    1. **Reset substitution.** On life loss (but not game over), ``reset()``
       below calls ``self.env.step(0)`` instead of ``self.env.reset()``, so
       the underlying ALE game keeps running under a "soft" episode boundary.
       A ``Transform`` cannot do this: ``TransformedEnv._reset()``
       unconditionally calls ``base_env._reset()`` *before* any transform's
       ``_reset`` hook runs, so the real reset has already happened by the
       time a transform could react. Only something that owns step/reset
       dispatch directly -- a ``gym.Wrapper``, not a ``Transform`` sitting on
       top of an already-built ``TransformedEnv`` -- can intercept and
       redirect that call before it fires.
    2. **The TorchRL-native alternative is deliberately weaker, and this repo
       already uses it elsewhere.** ``torchrl.envs.EndOfLifeTransform``
       (see ``configs/environment/ale.yaml``) sidesteps the reset-
       substitution problem entirely by not resetting on life loss at all:
       it tags the transition with an ``"end-of-life"`` key for the loss
       module's TD bootstrap and leaves episode boundaries at real game
       over. That's correct for the standard ALE protocol, but it is *not*
       equivalent to this class -- it never fragments a training episode, so
       it doesn't reset ``RewardSum``/``StepCounter`` or segment replay-buffer
       trajectories at life loss the way ``EpisodicLifeEnv`` does. Atari-100k
       (BBF/DER) needs that stronger, true-episode-boundary behaviour to
       match BBF-pytorch, which is why it keeps this gym wrapper instead of
       ``EndOfLifeTransform``.
    """

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


class MaxAndSkipTransform(Transform):
    """Repeat actions, sum rewards, and max-pool the last two pixel frames.

    TorchRL equivalent of :class:`MaxAndSkipEnv`. Use this when episodic-life
    handling is not required, or when no downstream gym wrapper must sit outside
    max-and-skip (see :class:`EpisodicLifeEnv`).

    Atari-100k training keeps max-and-skip at the gym level via
    ``gymnasium_wrappers`` so :class:`EpisodicLifeEnv` wraps it and life-loss
    is evaluated only after each aggregated agent step.
    """

    invertible = False

    def __init__(self, frame_skip: int = 4) -> None:
        super().__init__()
        if frame_skip < 1:
            raise ValueError("frame_skip must be >= 1.")
        self.frame_skip = frame_skip

    def _max_pool_pixels(self, obs_buffer: deque[torch.Tensor]) -> torch.Tensor:
        if len(obs_buffer) == 2:
            return torch.maximum(obs_buffer[0], obs_buffer[1])
        return obs_buffer[-1]

    def _step(
        self,
        tensordict: TensorDictBase,
        next_tensordict: TensorDictBase,
    ) -> TensorDictBase:
        parent = self.parent
        if parent is None:
            raise RuntimeError(f"{type(self).__name__} requires a parent env.")

        reward_key = parent.reward_key
        pixels_key = getattr(parent, "pixel_key", "pixels")
        reward = next_tensordict.get(reward_key)
        # maxlen=2: only the final two raw frames are ever max-pooled, matching
        # MaxAndSkipEnv. Without the bound, obs_buffer grows to frame_skip
        # frames and `_max_pool_pixels` silently falls back to "last frame
        # only" for any frame_skip != 2.
        obs_buffer: deque[torch.Tensor] = deque(maxlen=2)
        obs_buffer.append(next_tensordict.get(pixels_key))

        for _ in range(self.frame_skip - 1):
            terminated = next_tensordict.get("terminated")
            truncated = next_tensordict.get("truncated")
            if (terminated | truncated).any():
                break
            next_tensordict = parent._step(tensordict)
            reward = reward + next_tensordict.get(reward_key)
            obs_buffer.append(next_tensordict.get(pixels_key))

        return next_tensordict.set(
            pixels_key,
            self._max_pool_pixels(obs_buffer),
        ).set(reward_key, reward)
