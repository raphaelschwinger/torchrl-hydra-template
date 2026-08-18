# Claude Code instructions for torchrl-hydra-template

See `AGENTS.md` for the full codebase guide. This file adds Claude-specific notes.

## Maintenance rule

**Always update `README.md` and `AGENTS.md`** when changing a public API, adding an
algorithm, renaming a class, or changing a convention. README targets human readers;
AGENTS.md targets AI agents. Each algorithm package under `src/algorithms/<algo>/`
also has a `README.md` (theory, pseudocode, W&B results) — update it when adding
experiments or changing algorithm behaviour.

## Design principles

The template enforces a hard split between three components:

| Component       | Owns                                                                    |
|-----------------|-------------------------------------------------------------------------|
| **Algorithm**   | Everything that affects learning: network, replay buffer, loss, optimiser, exploration, target-net schedule, collector config (`frames_per_batch`, `init_random_frames`, ...). Hyperparameters live as keyword arguments on `__init__`. |
| **Trainer**     | The loop. Device placement, data collection (creates `Collector` from algorithm config), logging, callbacks, checkpointing. **No knobs that affect learning live here.** |
| **Environment** | One *benchmark*: backend, preprocessing stack, a `task` key naming the task within it, plus reporting metadata (`env_id`, `action_repeat`). Independent of algorithm. |
| **Evaluation**  | The *measurement* protocol: eval env stack, cadence, episode count, policy mode, and which stream is canonical. **No knobs that affect learning live here either.** |
| **Experiment**  | One algorithm × one benchmark. Owns everything that depends on the *task*: pixel-vs-state network choice, budgets, exploration schedules, replay capacity, episode length. |

Derived rules:

1. **RL algorithm code should read like the paper's pseudocode.** `step()` should be
   short and obviously correspond to the algorithm's update equations.
2. **Anything that influences reward / sample efficiency lives in the algorithm file.**
   If a knob shifts the learning curve, it belongs on `__init__`.
3. **`configs/algorithm/*.yaml` carries no task specifics.** The Python defaults and
   the algorithm YAML describe the algorithm; anything true only of a particular
   env or budget goes in `configs/experiment/<algo>/<benchmark>.yaml`. There is no
   `dqn_atari.yaml` / `ppo_atari.yaml` / `der.yaml` — those were per-task forks.
4. **Hydra factories.** Callable design choices (`replay_buffer`, `network`, …) are
   configured with `_partial_` / nested `_target_` in `configs/algorithm/*.yaml` and
   built via **`instantiate(cfg.algorithm, device=None)`** in `train.py` / `eval.py`.
5. **A factory with more than one variant becomes a config group**, never an inline
   dict patched from an experiment. Hydra *merges* dicts, so patching `_target_`
   alone leaves the previous option's kwargs behind — loudly for `NatureDQN`
   (`unexpected keyword argument 'num_cells'`), silently for PPO's heads. See
   `configs/algorithm/network/` and `configs/algorithm/policy/`.
6. **Environments are named per benchmark, tasks are overrides.** `gym`, `dmc`,
   `ale`, `atari100k` — each exposing `task`. Eval configs interpolate
   `${environment.task}` absolutely so one override moves both envs.
7. **Evaluation is its own config group.** `configs/evaluation/<benchmark>.yaml`
   selects the eval env stack *and* the protocol, so one override moves both:
   `- override /evaluation: atari100k`. All files inherit from
   `evaluation/none.yaml`, which is the schema of record — add new keys there
   first. Experiments never set `environment@eval_environment` directly.
8. **One x-axis for everything.** Metrics are logged against `global_step` in
   **agent steps** (what `batch.numel()` already counts), with
   `frames = global_step * environment.action_repeat` alongside. An algorithm
   must never define its own `log_step`.
9. **openrlbenchmark compatibility is a hard contract**, enforced by
   `tests/test_evaluation_contract.py`. `env_id` / `exp_name` / `seed` stay
   top-level in `configs/train.yaml`; `env_id` carries **no `ALE/` prefix** (the
   human-normalised-score table is keyed `Pong-v5`); episodes are logged one row
   each, never pre-aggregated; `wandb.log` is called **without** `step=` so
   `global_step` is a real data column.

Currently DQN (gym, ALE), DDPG (gym), A2C (gym), PPO (DMC, ALE),
TD-MPC2 (DMC), Rainbow/DER (Atari-100k), BBF (Atari-100k) and DreamerV3 +
variants (Atari-100k) are implemented; other algorithms will follow. Shared, algorithm-agnostic
building blocks (e.g. orthogonal-init actor-critic factories, and reusable code
adapted from external repos with source-attribution headers) live in
`src/components/`.

TD-MPC2 documents an accepted deviation from rule 3: architecturally coupled
subnetworks are built from scalar kwargs in `setup()` (no `_partial_` factories)
to stay state-dict compatible with official upstream checkpoints.

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
    return {"train/q_loss": ..., "train/epsilon": ...}
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
defaults:
  - network: mlp_q      # group: swapped whole, never patched in place
  - _self_

_target_: src.algorithms.dqn.DQNAlgorithm
replay_buffer:
  _partial_: true
  _target_: torchrl.data.TensorDictReplayBuffer
  storage:
    _target_: torchrl.data.LazyTensorStorage
    max_size: 10_000
    device: cpu
lr: 2.5e-4
gamma: 0.99
# ...
```

```yaml
# configs/algorithm/network/mlp_q.yaml — package inferred as algorithm.network
_partial_: true
_target_: src.components.networks.make_mlp_q_net
num_cells: [120, 84]
activation_class:
  _target_: hydra.utils.get_class
  path: torch.nn.ReLU
```

```yaml
# configs/experiment/dqn/ale.yaml — task specifics live here
defaults:
  - override /algorithm: dqn
  - override /algorithm/network: nature_dqn
  - override /environment: ale
  - override /environment@eval_environment: ale_eval
  - _self_

environment:
  task: Pong
algorithm:
  obs_key: pixels
  annealing_frames: 4_000_000
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
- Do **not** put task specifics (game name, pixel network, training budget,
  episode length) in `configs/algorithm/*.yaml` — they belong in the experiment.
- Do **not** create a per-task env config (`pong_train.yaml`, `jamesbond_train.yaml`).
  Add the *benchmark* once and select the task with `environment.task=`.
- Do **not** patch a network factory's `_target_` from an experiment body; override
  the config group instead (`override /algorithm/network: ...`).
- Do **not** put evaluation cadence or episode counts on `trainer:` — they
  belong in `configs/evaluation/`. Experiments select the group
  (`override /evaluation: atari100k`), never `environment@eval_environment`.
- Do **not** call `wandb.log(..., step=...)`, and do not log metrics from inside
  an algorithm. Route everything through `BaseTrainer.log_metrics` /
  `log_episodes`, which inject `global_step` and `frames`. The one accepted
  exception is Dreamer's `video/*`, which keeps its own `video/frame` axis.
- Do **not** pre-aggregate episode returns into one point per log boundary.
  openrlbenchmark averages the last 100 *logged points*, so a windowed mean
  silently changes what that window measures.
