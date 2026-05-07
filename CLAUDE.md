# Claude Code instructions for torchrl-hydra-template

See `AGENTS.md` for the full codebase guide. This file adds Claude-specific notes.

## Maintenance rule

**Always update `README.md` and `AGENTS.md`** when changing a public API, adding an
algorithm, renaming a class, or changing a convention. README targets human readers;
AGENTS.md targets AI agents.

## Design principles

The template enforces a hard split between three components:

| Component       | Owns                                                                    |
|-----------------|-------------------------------------------------------------------------|
| **Algorithm**   | Everything that affects learning: network, replay buffer, loss, optimiser, exploration, target-net schedule, collector config (`frames_per_batch`, `init_random_frames`, ...). Hyperparameters live as keyword arguments on `__init__`. |
| **Trainer**     | The loop. Device placement, data collection (creates `Collector` from algorithm config), logging, callbacks, checkpointing. **No knobs that affect learning live here.** |
| **Environment** | Fixed task definition. Env name + transform list. Independent of algorithm. |

Two derived rules:

1. **RL algorithm code should read like the paper's pseudocode.** `step()` should be
   short and obviously correspond to the algorithm's update equations.
2. **Anything that influences reward / sample efficiency lives in the algorithm file.**
   If a knob shifts the learning curve, it belongs on `__init__`.

Currently only DQN is implemented; other algorithms will follow.

## Key patterns (quick reference)

### Algorithm constructor

```python
class DQNAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        device: torch.device | None = None,
        *,
        replay_buffer: Callable[[], ReplayBuffer] = default_replay_buffer,
        network: Callable[[tuple[int, ...], int], nn.Module] = default_network,
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        # ... more HPs
    ) -> None:
        super().__init__(device)
        # ... store kwargs
```

- `*` makes every HP keyword-only.
- `BaseAlgorithm.__init__` only takes `device`. **No `cfg` parameter.**
- `replay_buffer` and `network` are **Callable factories** so design choices
  (storage type, MLP shape) are visible in code and overridable in code.
- Scalar HPs (`lr`, `gamma`, `batch_size`, `eps_*`, `frames_per_batch`,
  `init_random_frames`, `num_updates`, `hard_update_freq`, ...) are plain kwargs.
- `setup(make_env)` reads env specs from a short-lived proof env;
  the algorithm does not store a long-lived env reference.

### `step(batch)` shape

```python
def step(self, batch: TensorDict) -> dict[str, float]:
    # 1. Always: anneal exploration + store transitions
    # 2. Skip during warm-up
    # 3. Loop num_updates: sample -> loss -> backward -> optimiser -> target update
    return {"loss/td": ..., "epsilon": ...}
```

The trainer calls `step(batch)` with a TensorDict from `Collector`. The trainer never
touches the replay buffer, target net, or epsilon — those are algorithm internals.

### Instantiation in `train.py`

```python
alg_kwargs = {k: v for k, v in OmegaConf.to_container(cfg.algorithm, resolve=True).items()
              if k != "_target_"}
algorithm = AlgClass(device=None, **alg_kwargs)

env_kwargs = {k: v for k, v in OmegaConf.to_container(cfg.environment, resolve=True).items()
              if k != "_target_"}
environment = Environment(**env_kwargs)
```

### YAML convention

Algorithm YAML mirrors Python defaults so values are easy to override per
experiment / from CLI:

```yaml
# configs/algorithm/dqn.yaml
_target_: src.algorithms.dqn.DQNAlgorithm
# Default values and parameter descriptions: src/algorithms/dqn.py (DQNAlgorithm.__init__)

lr: 2.5e-4
gamma: 0.99
# ...
```

`replay_buffer` and `network` are **not** exposed in YAML — to swap a buffer storage
or change the network, edit the algorithm file.

## What not to do

- Do **not** put learning-affecting knobs on the trainer or env config (e.g. don't
  add `lr` or `gamma` to `trainer:` or `environment:`).
- Do **not** create config dataclasses (`DQNConfig`, etc.) for HPs.
- Do **not** add `cfg: DictConfig` to `BaseAlgorithm.__init__` or
  `AlgClass(cfg=cfg, ...)` in `train.py`.
- Do **not** add `OmegaConf` imports to `base.py` — it has no config logic.
- Do **not** add new algorithms or environment backends without first updating
  README.md and AGENTS.md to describe them.
