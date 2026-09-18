"""The uniform stepping interface every provider is reduced to.

`step()` takes no arguments. Each provider owns a pre-generated pool of uniform
random actions and indexes into it, so action sampling never enters the timed
loop: it is not the thing being measured, and its cost differs by action-space
type, which would quietly bias the comparison toward discrete-action providers.
"""

from __future__ import annotations

import abc

import numpy as np

# Actions are drawn from a fixed pool that the loop cycles through. Large enough
# that the simulator sees varied input, small enough to stay in cache and to let
# a GPU provider keep the whole pool device-resident.
ACTION_POOL = 64


class Provider(abc.ABC):
    """One vectorised environment, reduced to what the timing loop needs."""

    parallelism: str = "none"
    device: str = "cpu"

    def __init__(
        self,
        num_envs: int,
        *,
        versions: str = "",
        obs_signature: str = "",
        obs_shape: tuple[int, ...] = (),
        sim_seconds_per_agent_step: float = 0.0,
    ) -> None:
        self.num_envs = num_envs
        self.versions = versions
        self.obs_signature = obs_signature
        self.obs_shape = obs_shape
        self.sim_seconds_per_agent_step = sim_seconds_per_agent_step

    @abc.abstractmethod
    def step(self) -> None:
        """Advance every environment by one agent step."""

    def step_blocking(self) -> None:
        """Advance and wait for completion.

        Identical to `step()` for synchronous providers. Accelerator providers
        override it: their `step()` only enqueues work.
        """
        self.step()

    def synchronize(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Block until all previously enqueued steps have actually run.

        Nothing to wait for unless a provider dispatches asynchronously.
        """

    def close(self) -> None:  # noqa: B027 - optional hook, no-op by default
        """Release workers, thread pools and device memory."""


def signature(dtype, per_env_shape) -> str:
    """A short signature like `uint8(4,84,84)` for one environment's observation.

    Recorded per cell and compared across providers: it is the cheap, mechanical
    evidence that everyone was actually asked to do the same work. One formatter,
    used everywhere, so that two providers agreeing cannot be hidden by two
    spellings of the same shape.
    """
    return f"{dtype}({','.join(str(int(d)) for d in per_env_shape)})"


def describe(array: np.ndarray) -> str:
    """`signature` for a batched array whose leading axis is the batch."""
    return signature(array.dtype, array.shape[1:])


def discrete_action_pool(rng: np.random.Generator, num_envs: int, n: int) -> np.ndarray:
    return rng.integers(0, n, size=(ACTION_POOL, num_envs), dtype=np.int32)


def continuous_action_pool(
    rng: np.random.Generator,
    num_envs: int,
    dim: int,
    low: float,
    high: float,
    dtype: np.dtype | str,
) -> np.ndarray:
    pool = rng.uniform(low, high, size=(ACTION_POOL, num_envs, dim))
    return pool.astype(dtype)
