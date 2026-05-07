# Agent instructions for torchrl-hydra-template

## Project overview

A thin Hydra wrapper around TorchRL's built-in algorithm trainers. The whole training entry point is six lines:

```python
@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg):
    if cfg.get("seed") is not None:
        seed_everything(int(cfg.seed))
    trainer = hydra.utils.instantiate(cfg.trainer)
    trainer.train()
```

Everything else (collector, loss module, replay buffer, networks, optimizer, env transforms) is composed in YAML using the structured configs that TorchRL ships in `torchrl.trainers.algorithms.configs` — see [`sota-implementations/dqn_trainer`](https://github.com/pytorch/rl/tree/main/sota-implementations/dqn_trainer) and [`sota-implementations/ppo_trainer`](https://github.com/pytorch/rl/tree/main/sota-implementations/ppo_trainer) upstream for reference.

The template's value is:
1. A curated set of **experiment recipes** under `configs/experiment/<algo>/<env>.yaml` that compose algorithm + environment defaults.
2. An **extension example** (REINFORCE) showing how to add an algorithm not shipped by TorchRL — see `src/algorithms/reinforce/`.
3. A small **custom-network** example (`src/networks/atari.py`) wired into an experiment.

## Key conventions

### Composition strategy

- `configs/algorithm/<name>.yaml` defines the algorithm-side composition: which trainer, loss, replay buffer, optimizer, default networks. References `${env.*}` keys via OmegaConf interpolation.
- `configs/environment/<name>.yaml` defines the env factory + transforms and exposes `env.obs_dim`, `env.action_dim`, and (for continuous-action algos) `env.policy_out_dim`.
- `configs/experiment/<algo>/<env>.yaml` glues an algorithm + environment together with `# @package _global_` and `override` directives.
- `configs/logger/{csv,wandb,tensorboard}.yaml` sets the logger; default is `wandb`, switch from CLI with `logger=csv` or `logger=tensorboard`.

### Schema-typed configs

Every component config is a dataclass registered with Hydra's ConfigStore (TorchRL provides them via `from torchrl.trainers.algorithms.configs import *`). Pulling a component into your YAML attaches its schema:

```yaml
defaults:
  - /trainer@trainer: dqn          # attaches DQNTrainerConfig
  - /loss@loss: dqn                # attaches DQNLossConfig
  - /network@networks.qvalue_network: mlp  # attaches MLPConfig
```

Then top-level sections in the same YAML override field values:

```yaml
trainer:
  total_frames: 500_000
  log_interval: 10_000
optimizer:
  lr: 2.5e-4
```

To **drop** an attached schema (e.g. swap a default MLP for a custom CNN class), use `null` in defaults:

```yaml
defaults:
  - override /network@networks.qvalue_network: null
networks:
  qvalue_network:
    _target_: src.networks.atari.AtariCNN
    out_features: ${env.action_dim}
```

### Per-environment shape exposure

Every `configs/environment/<name>.yaml` MUST set:

- `env.obs_dim` — flat observation size (or `null` if image-based)
- `env.action_dim` — number of discrete actions, or continuous-action dim
- `env.policy_out_dim` — for continuous algos using `tanh_normal`: `2 * action_dim` (loc, scale). Optional otherwise.

Algorithm configs interpolate from these so that `python src/train.py algorithm=dqn environment=cartpole` works without an experiment file.

### Top-level `train.yaml` is NOT `# @package _global_`

Only group files (in `configs/algorithm/`, `configs/environment/`, `configs/experiment/`) need that directive. The top-level config sits at the root package by default.

### Don't shadow TorchRL ConfigStore group names

TorchRL registers groups like `/network`, `/loss`, `/trainer`, `/transform`. If you create `configs/network/<x>.yaml` in your search path, Hydra tries to merge our file's content with TorchRL's `/network` schema and recurses. Use the schema attachment pattern shown above (`defaults: - /network@networks.qvalue_network: mlp`) instead of creating a parallel `network/` group.

## File map

```
src/
  train.py                  — @hydra.main → instantiate cfg.trainer, train
  eval.py                   — load checkpoint, run greedy rollouts
  algorithms/
    dqn/                    — empty (uses TorchRL DQNTrainer directly)
    ppo/                    — empty (uses TorchRL PPOTrainer directly)
    reinforce/
      trainer.py            — make_reinforce_trainer factory
      configs.py            — REINFORCETrainerConfig + ReinforceLossConfig
                              (registered with Hydra ConfigStore on import)
  networks/
    atari.py                — AtariCNN (Mnih 2015 architecture)
  utils/
    seeding.py              — seed_everything

configs/
  train.yaml                — entry config, defaults list
  paths/default.yaml        — root_dir / log_dir / output_dir
  algorithm/                — dqn.yaml, ppo.yaml, reinforce.yaml
  environment/              — cartpole.yaml, atari_pong.yaml, pendulum.yaml, halfcheetah.yaml
  logger/                   — csv.yaml, wandb.yaml, tensorboard.yaml
  experiment/
    dqn/                    — cartpole.yaml, atari_pong.yaml
    ppo/                    — pendulum.yaml, halfcheetah.yaml
    reinforce/              — pendulum.yaml

tests/test_smoke.py         — composes every shipped experiment + builds Trainer
```

## Adding a new algorithm

### Case A: TorchRL ships it (DQN / PPO / SAC / TD3 / DDPG / CQL / IQL)

1. Create `configs/algorithm/<name>.yaml`. Copy `configs/algorithm/dqn.yaml` and adapt:
   - `defaults`: pull `/trainer@trainer: <name>`, `/loss@loss: <name>`, the right network/model schemas, etc.
   - Set hyperparameters in top-level sections (`optimizer.lr`, `trainer.total_frames`, etc.).
2. Create `configs/experiment/<name>/<env>.yaml`:
   ```yaml
   # @package _global_
   defaults:
     - override /algorithm: <name>
     - override /environment: <env>
     - _self_
   ```
3. Add a parametrize entry in `tests/test_smoke.py`.

### Case B: TorchRL does NOT ship it (custom)

See `src/algorithms/reinforce/` as the canonical example:

1. `src/algorithms/<name>/trainer.py` — `make_<name>_trainer(...)` factory. Build the loss, collector, optimizer; return a `torchrl.trainers.Trainer`.
2. `src/algorithms/<name>/configs.py` — define `<NAME>TrainerConfig(TrainerConfig)` and any custom loss config; register with `cs.store(group="trainer", name="<name>", node=...)`.
3. Add `from src.algorithms.<name>.configs import *  # noqa` to `src/train.py` so the registration runs at import time.
4. Create `configs/algorithm/<name>.yaml` and `configs/experiment/<name>/<env>.yaml` as in Case A.

## Adding a new environment

1. Create `configs/environment/<name>.yaml` modeled on existing files. Set `env.obs_dim`, `env.action_dim`, `env.policy_out_dim` (the last only matters for continuous-action algorithms).
2. Compose transforms via `defaults: - /transform@transform_<thing>: <kind>` and reference them in `training_env.create_env_fn.transform.transforms`.
3. For Atari/ALE envs: `_register_atari_envs()` in `src/train.py` already handles registration.
4. For envs with float64 observations (most MuJoCo): include the `double_to_float` transform.

## Maintenance

**Always update `README.md` and `AGENTS.md`** when:
- Changing how configs compose (e.g. introducing a new top-level group)
- Adding or removing an algorithm/environment/experiment
- Changing the entry point or eval script

`README.md` is human-facing; `AGENTS.md` is agent-facing. Both must stay in sync with the code.

## Running experiments

```shell
# Curated experiment recipes
python -m src.train experiment=dqn/cartpole
python -m src.train experiment=dqn/atari_pong trainer.progress_bar=true
python -m src.train experiment=ppo/pendulum
python -m src.train experiment=ppo/halfcheetah
python -m src.train experiment=reinforce/pendulum

# Compose without an experiment file
python -m src.train algorithm=dqn environment=cartpole logger=wandb

# Override individual hyperparams
python -m src.train experiment=dqn/cartpole optimizer.lr=1e-3 trainer.total_frames=200_000

# Multi-run with hydra
python -m src.train -m experiment=dqn/cartpole seed=0,1,2

# Eval a checkpoint
python -m src.eval experiment=dqn/cartpole checkpoint=outputs/<run>/trainer_<step>.pt num_episodes=10

# Tests
pytest tests/test_smoke.py -v
```
