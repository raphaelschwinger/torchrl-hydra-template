# torchrl-hydra-template

A small Hydra-driven template for training reinforcement learning agents with [TorchRL](https://github.com/pytorch/rl). The goal is to make it easy to swap algorithms and environments from a single CLI:

```bash
python -m src.train experiment=dqn/atari_pong
python -m src.train algorithm=ppo environment=halfcheetah trainer.total_frames=2_000_000
python -m src.train -m experiment=dqn/cartpole seed=0,1,2
```

Built on top of TorchRL's first-class algorithm trainers (`DQNTrainer`, `PPOTrainer`, `SACTrainer`, `TD3Trainer`, `DDPGTrainer`, `CQLTrainer`, `IQLTrainer`) shipped in [`torchrl.trainers.algorithms`](https://github.com/pytorch/rl/tree/main/torchrl/trainers/algorithms). The template adds the Hydra config layer and a curated set of experiment recipes; it does **not** introduce its own algorithm abstractions.

## Philosophy

The whole training entry point fits in a screen:

```python
@hydra.main(config_path="../configs", config_name="train", version_base="1.3")
def main(cfg):
    if cfg.get("seed") is not None:
        seed_everything(int(cfg.seed))
    trainer = hydra.utils.instantiate(cfg.trainer)
    trainer.train()
```

Every component (collector, loss, replay buffer, networks, optimizer, env transforms) is composed in YAML using the structured configs that TorchRL exports from `torchrl.trainers.algorithms.configs`. Each component has a `_target_` so Hydra instantiates it directly — there's no wrapper layer between YAML and TorchRL.

## Quick start

```bash
# Install dependencies (uv recommended)
uv sync

# Smoke check on CartPole
python -m src.train experiment=dqn/cartpole \
    trainer.total_frames=20_000 trainer.progress_bar=true

# PPO on Pendulum
python -m src.train experiment=ppo/pendulum \
    trainer.total_frames=100_000

# Run all compose tests
pytest tests/test_smoke.py -v
```

## Repo layout

```
configs/
  train.yaml                — top-level entry config
  paths/default.yaml        — root_dir / log_dir / output_dir
  algorithm/                — dqn.yaml, ppo.yaml, reinforce.yaml
  environment/              — cartpole.yaml, atari_pong.yaml, pendulum.yaml, halfcheetah.yaml
  logger/                   — csv.yaml, wandb.yaml, tensorboard.yaml
  experiment/               — composed recipes per <algo>/<env>

src/
  train.py                  — @hydra.main entry point
  eval.py                   — load checkpoint, run greedy rollouts
  algorithms/
    dqn/                    — uses TorchRL DQNTrainer directly (empty package)
    ppo/                    — uses TorchRL PPOTrainer directly (empty package)
    reinforce/              — custom trainer example (TorchRL doesn't ship REINFORCE)
      trainer.py
      configs.py
  networks/atari.py         — AtariCNN encoder for image-based DQN/PPO
  utils/seeding.py          — set torch / numpy / random seeds

tests/test_smoke.py         — composes every shipped experiment and builds the Trainer
```

## Shipped experiments

| Recipe | Trainer | Notes |
| --- | --- | --- |
| `dqn/cartpole` | `DQNTrainer` | Mirrors TorchRL's `sota-implementations/dqn_trainer/config/config.yaml`. |
| `dqn/atari_pong` | `DQNTrainer` | Mnih 2015 preprocessing + `AtariCNN`. Composes; needs GPU for full training. |
| `ppo/pendulum` | `PPOTrainer` | Mirrors TorchRL's `sota-implementations/ppo_trainer/config/config.yaml`. |
| `ppo/halfcheetah` | `PPOTrainer` | Standard MuJoCo benchmark with `DoubleToFloat` transform. |
| `reinforce/pendulum` | custom | Demonstrates the "add a new algorithm" extension pattern. |

## How configs compose

```
train.yaml
  defaults:
    - paths: default
    - algorithm: ???      # required: dqn / ppo / reinforce / ...
    - environment: ???    # required: cartpole / pendulum / halfcheetah / atari_pong / ...
    - logger: wandb       # default; switch with `logger=csv` or `logger=tensorboard`
    - experiment: null    # optional bundle (overrides algorithm + environment)
```

Each `algorithm/<name>.yaml` references TorchRL's structured configs:

```yaml
# configs/algorithm/dqn.yaml
defaults:
  - /network@networks.qvalue_network: mlp     # → MLPConfig
  - /model@models.qvalue_model: qvalue        # → QValueModelConfig
  - /collector@collector: sync                # → SyncDataCollectorConfig
  - /replay_buffer@replay_buffer: base
  - /trainer@trainer: dqn                     # → DQNTrainerConfig
  - /optimizer@optimizer: adam
  - /loss@loss: dqn                           # → DQNLossConfig
  - /target_net_updater@target_net_updater: hard

networks:
  qvalue_network:
    in_features: ${env.obs_dim}      # interpolated from environment config
    out_features: ${env.action_dim}
    num_cells: [120, 84]
# ... rest of the file overrides hyperparameters
```

The environment file exposes the keys that algorithm files interpolate:

```yaml
# configs/environment/cartpole.yaml
env:
  obs_dim: 4
  action_dim: 2
  action_space: one-hot
```

An experiment file picks both:

```yaml
# configs/experiment/dqn/cartpole.yaml
# @package _global_
defaults:
  - override /algorithm: dqn
  - override /environment: cartpole
  - _self_
```

## Adding a new algorithm

If TorchRL ships it (DQN/PPO/SAC/TD3/DDPG/CQL/IQL), just copy `configs/algorithm/dqn.yaml`, swap the `_target_` schemas in the `defaults` list, and write an experiment file. No Python code needed.

If TorchRL doesn't ship it, see `src/algorithms/reinforce/` as a worked example. You write a builder function returning a `torchrl.trainers.Trainer`, plus a dataclass config registered with Hydra's `ConfigStore` so YAML can pull it in via `defaults: - /trainer@trainer: <yourname>`.

Full instructions: see [`AGENTS.md`](./AGENTS.md).

## Adding a new environment

Create `configs/environment/<name>.yaml`. Compose the env via TorchRL's shipped `gym` / `dm_control` env configs and a list of `/transform@transform_*` defaults. Define `env.obs_dim`, `env.action_dim`, and `env.policy_out_dim` so algorithm configs can interpolate network shapes.

For Atari (or anything ALE-based): the entry point already calls `gym.register_envs(ale_py)` so `PongNoFrameskip-v4`-style names work.

For envs with float64 observations (most MuJoCo tasks): include the `double_to_float` transform.

## CLI overrides

Anything in the composed config can be overridden from the command line:

```bash
# Tune individual hyperparameters
python -m src.train experiment=dqn/cartpole \
    optimizer.lr=1e-3 trainer.total_frames=200_000

# Switch loggers
python -m src.train experiment=ppo/pendulum logger=wandb

# Sweep with Hydra multirun
python -m src.train -m experiment=ppo/halfcheetah seed=0,1,2 optimizer.lr=1e-4,3e-4,1e-3
```

## Eval

`python -m src.eval` builds the same trainer, restores weights from a checkpoint, and runs greedy rollouts:

```bash
python -m src.eval experiment=dqn/cartpole \
    checkpoint=outputs/<run>/trainer_<step>.pt num_episodes=10
```

This is intentionally minimal — the policy is pulled from the trainer's loss module's `actor_network` (PPO/REINFORCE) or `value_network` (DQN). For richer eval (multi-seed, video recording), extend `src/eval.py`.

## Comparison to BenchMARL

[BenchMARL](https://github.com/facebookresearch/BenchMARL) is the closest existing TorchRL-on-Hydra template; we've intentionally landed in a thinner spot. BenchMARL has a custom `Algorithm` ABC (with `_get_loss`/`_get_parameters`/etc.), a custom `Experiment` orchestrator, and `Task`/`Benchmark` abstractions. With TorchRL now shipping per-algorithm trainers + structured configs, that abstraction is no longer needed for single-agent RL — `trainer.train()` is the orchestrator and Hydra's multirun replaces the `Benchmark` class. The result is a smaller surface area and tighter alignment with upstream TorchRL.

## Acknowledgements

- [TorchRL](https://github.com/pytorch/rl) — the trainers, losses, and configs.
- [Hydra](https://github.com/facebookresearch/hydra) — config composition.
- [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) — directory conventions (`configs/<group>/`, `# @package _global_` experiments, `paths/`).
- [BenchMARL](https://github.com/facebookresearch/BenchMARL) — comparison point that helped clarify the right level of abstraction.
