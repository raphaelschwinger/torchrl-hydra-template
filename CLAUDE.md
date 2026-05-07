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
3. **Hydra factories.** Callable design choices (`replay_buffer`, `network`, …) are
   configured with `_partial_` / nested `_target_` in `configs/algorithm/*.yaml` and
   built via **`instantiate(cfg.algorithm, device=None)`** in `train.py` / `eval.py`.

Currently only DQN is implemented; other algorithms will follow.

## Key patterns (quick reference)

### Algorithm constructor

```python
class DQNAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        device: torch.device | None = None,
        *,
        replay_buffer: Callable[[], ReplayBuffer] = lambda: TensorDictReplayBuffer(...),
        network: Callable[[int, int], nn.Module] = functools.partial(
            MLP, num_cells=[120, 84], activation_class=nn.ReLU
        ),
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        # ... more HPs
    ) -> None:
        super().__init__(device)
        # ... store kwargs
```

- `*` makes every HP keyword-only.
- `BaseAlgorithm.__init__` only takes `device`. **No `cfg` parameter.**
- `replay_buffer` is a **no-arg** factory; `network` is called as **`network(in_features,
  out_features)`** (flattened obs dim and `|A|`). In `setup()`, build the net with
  those two integers after reading specs from a short-lived proof env.
- For TorchRL `MLP`, use **`functools.partial(MLP, ...)`** in code and **`_partial_` +
  `_target_: torchrl.modules.MLP`** in YAML; leave `in_features` / `out_features`
  unbound so `setup()` fills them.
- **`activation_class` in YAML:** use **`hydra.utils.get_class`** with `path:
  torch.nn.ReLU` (or another layer class). Do **not** use `_target_: torch.nn.ReLU`
  as a nested kwarg to `MLP` — Hydra would instantiate a module instance, which
  breaks `MLP`'s API.
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

### Instantiation in `train.py` / `eval.py`

```python
from hydra.utils import instantiate, get_class
from omegaconf import OmegaConf

algorithm = instantiate(cfg.algorithm, device=None)

env_kwargs = {k: v for k, v in OmegaConf.to_container(cfg.environment, resolve=True).items()
              if k != "_target_"}
environment = Environment(**env_kwargs)

TrainerClass = get_class(cfg.trainer._target_)
trainer = TrainerClass(cfg=cfg, algorithm=algorithm, environment=environment)
```

**Algorithms** use `instantiate(cfg.algorithm, ...)` so nested `_partial_` /
`_target_` configs (factories) become real callables. **Environments** stay flat
`to_container` + `**kwargs`. **Trainers** use `get_class` + constructor (they take
`cfg` as a whole).

### YAML convention

Algorithm YAML mirrors Python defaults and exposes scalar overrides. **Design
choices implemented as `Callable`s** (`replay_buffer`, `network`, …) live in
YAML as **`_partial_: true`** blocks with nested `_target_` nodes, matching DQN in
`configs/algorithm/dqn.yaml`. That requires `instantiate(cfg.algorithm)` in the
entry points.

```yaml
# configs/algorithm/dqn.yaml (illustrative)
_target_: src.algorithms.dqn.DQNAlgorithm
replay_buffer:
  _partial_: true
  _target_: torchrl.data.TensorDictReplayBuffer
  storage:
    _target_: torchrl.data.LazyTensorStorage
    max_size: 10_000
    device: cpu
network:
  _partial_: true
  _target_: torchrl.modules.MLP
  num_cells: [120, 84]
  activation_class:
    _target_: hydra.utils.get_class
    path: torch.nn.ReLU
lr: 2.5e-4
gamma: 0.99
# ...
```

## What not to do

- Do **not** put learning-affecting knobs on the trainer or env config (e.g. don't
  add `lr` or `gamma` to `trainer:` or `environment:`).
- Do **not** create config dataclasses (`DQNConfig`, etc.) for HPs.
- Do **not** add `cfg: DictConfig` to `BaseAlgorithm.__init__` or
  `AlgClass(cfg=cfg, ...)` in `train.py`.
- Do **not** add `OmegaConf` imports to `base.py` — it has no config logic.
- Do **not** add new algorithms or environment backends without first updating
  README.md and AGENTS.md to describe them.
