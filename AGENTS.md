# Agent instructions for torchrl-hydra-template

## Project overview

A modular reinforcement learning research template built on [TorchRL](https://github.com/pytorch/rl) and [Hydra](https://github.com/facebookresearch/hydra). Three composable components — **Environment**, **Algorithm**, **Trainer** — are wired together by `src/train.py`.

## Key conventions

### Algorithm hyperparameter pattern

Hyperparameters live as **explicit keyword arguments on `__init__`**, not in a separate config dataclass. Every algorithm constructor follows this shape:

```python
class MyAlgorithm(BaseAlgorithm):
    """Short description.

    Args:
        cfg: Full Hydra config (trainer, logger, environment sections).
        device: Resolved torch.device; set by the Trainer.
        lr: Learning rate for the Adam optimizer.
        gamma: Discount factor.
        network: Dict with keys ``architecture``, ``hidden_sizes``, ``activation``,
            ``layer_norm``.
    """

    def __init__(
        self,
        cfg: DictConfig,
        device: torch.device | None = None,
        *,
        lr: float = 3e-4,
        gamma: float = 0.99,
        network: dict | None = None,
    ) -> None:
        super().__init__(cfg, device)
        self.lr = lr
        self.gamma = gamma
        self._network_cfg = network or {"architecture": "mlp", ...}
```

- All HPs are keyword-only (`*` separator after `device`).
- Nested dict groups (`network`, `replay_buffer`) use `dict | None = None` with body-level defaults.
- Every parameter has a type annotation, a sensible default, and a docstring line.
- `setup()` uses `self.xxx` directly — **no `self.acfg`**.
- Wrap nested dict configs before passing to `make_network`: `OmegaConf.create(self._network_cfg)`.

### How algorithms are instantiated (`src/train.py`)

```python
AlgClass = get_class(cfg.algorithm._target_)
alg_kwargs = {k: v for k, v in OmegaConf.to_container(cfg.algorithm, resolve=True).items()
              if k != "_target_"}
algorithm = AlgClass(cfg=cfg, device=None, **alg_kwargs)
```

YAML values override Python defaults; keys absent from the YAML fall back to constructor defaults.

### Environment pattern

`Environment.__init__` accepts env parameters explicitly:

```python
env_kwargs = {k: v for k, v in OmegaConf.to_container(cfg.environment, resolve=True).items()
              if k != "_target_"}
environment = Environment(**env_kwargs)
```

`make_env` in `src/environments/factory.py` builds TorchRL `TransformedEnv` instances. `Environment` stores params in `self._factory_kwargs` and calls `make_env(**self._factory_kwargs, num_envs=num_envs, device=device)`.

#### Gymnasium transforms list

For gymnasium envs, transforms are configured as an explicit list in the YAML. Each entry has a `_target_` key pointing to a `torchrl.envs.transforms` class; kwargs follow. They are instantiated fresh per `make_env()` call via `hydra.utils.instantiate` (important for stateful transforms like `CatFrames`).

```yaml
# configs/environment/my_env.yaml
from_pixels: true       # pass from_pixels=True to GymEnv for pixel obs
transforms:
  - _target_: torchrl.envs.transforms.ToTensorImage
    in_keys: [pixels]
    out_keys: [pixels]
  - _target_: torchrl.envs.transforms.GrayScale
    in_keys: [pixels]
    out_keys: [pixels]
  - _target_: torchrl.envs.transforms.Resize
    h: 84
    w: 84
    in_keys: [pixels]
    out_keys: [pixels]
  - _target_: torchrl.envs.transforms.CatFrames
    N: 4
    dim: -3
    in_keys: [pixels]
    out_keys: [observation]   # DQNAlgorithm expects "observation" key
  - _target_: torchrl.envs.transforms.StepCounter
```

Key rules:
- `CatFrames` must map `in_keys: [pixels]` → `out_keys: [observation]` so `DQNAlgorithm` finds the expected `"observation"` key.
- Always include `StepCounter` explicitly in the list — it is no longer auto-appended.
- `VecNorm` requires `init_stats()` — skip it (no hook in this architecture).
- dm_control and envpool use built-in transform pipelines; backend-specific kwargs (`normalize_obs`, `clip_rewards`) flow through `**kwargs`.

### YAML config files

- `configs/algorithm/<algo>.yaml` — sets non-default HP values; `_target_` points to the class.
- `configs/environment/<env>.yaml` — environment kwargs.
- `configs/experiment/<algo>/<env>.yaml` — composed overrides with `@package _global_`.
- Algorithm YAML comments reference `<AlgoClass>.__init__` as the canonical source of defaults/docs.

### Network config

`make_network(cfg, obs_shape, out_features)` in `src/networks/factory.py` accepts a `DictConfig` (not a plain dict). Always wrap the stored dict:

```python
net_cfg = OmegaConf.create(self._network_cfg)
q_net = make_network(net_cfg, obs_shape, num_actions)
```

## File map

```
src/
  train.py                  — entry point; unpacks cfg.algorithm and cfg.environment as **kwargs
  algorithms/
    base.py                 — BaseAlgorithm ABC; TrainingState and CollectorConfig dataclasses
    dqn.py                  — DQNAlgorithm  (off-policy, StepTrainer)
    ppo.py                  — PPOAlgorithm  (on-policy, StepTrainer)
    reinforce.py            — ReinforceAlgorithm (on-policy, EpisodeTrainer)
  environments/
    environment.py          — Environment wrapper (holds factory kwargs, exposes make_env)
    factory.py              — make_env: Gymnasium / dm_control / envpool + transforms
  networks/
    factory.py              — make_network: MLP, AtariCNN
  trainers/
    BaseTrainer.py          — BaseTrainer ABC, TrainerEvent, Callback protocol, fire_callbacks
    EpisodeTrainer.py       — EpisodeTrainer (episodic rollouts via env.rollout)
    StepTrainer.py          — StepTrainer (fixed-size batches via SyncDataCollector)
configs/
  algorithm/                — per-algo HP overrides + _target_
  environment/              — env kwargs (cartpole, atari_breakout, atari_pong, dmc_humanoid)
  experiment/               — composed experiment configs (dqn/cartpole, dqn/atari-pong, …)
```

## Adding a new algorithm

1. Create `src/algorithms/my_algo.py` following the explicit-kwargs pattern above.
2. Implement `setup(make_env)` (zero-arg env factory; call `env = make_env()` only if you need env specs), `step(batch) -> dict`, `get_policy()`, `get_explore_policy()`, `_get_training_state()`, `_load_training_state()`.
3. Add `get_collector_config()` if using `StepTrainer` (see `src/trainers/StepTrainer.py`).
4. Create `configs/algorithm/my_algo.yaml` with `_target_` and any non-default values.
5. Create `configs/experiment/my_algo/<env>.yaml` with defaults overrides.

## Maintenance

**Always update `README.md` and `AGENTS.md`** when:
- Changing a public API (class signature, method name, config structure)
- Adding or removing an algorithm or environment backend
- Changing a cross-cutting convention (e.g. how HPs are passed, how configs are unpacked)

`README.md` is the human-facing reference; `AGENTS.md` is the agent-facing reference. Both must stay in sync with the code.

## Running experiments

```shell
python src/train.py experiment=dqn/cartpole
python src/train.py experiment=dqn/atari_pong trainer.accelerator=gpu trainer.devices=[0]
python src/train.py experiment=ppo/dmc_humanoid trainer.accelerator=gpu trainer.devices=[0]
python src/train.py experiment=dqn/cartpole algorithm.lr=1e-3
pytest tests/test_smoke.py -v
```
