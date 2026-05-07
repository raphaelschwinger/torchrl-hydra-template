<div align="center">

# TorchRL Hydra Template

A clean, modular template for deep reinforcement learning research.<br>
Click on [<kbd>Use this template</kbd>](https://github.com/raphaelschwinger/torchrl-hydra-template/generate) to initialize a new repository.

_Suggestions are always welcome!_

</div>

## Philosophy

Reinforcement learning code tends to become monolithic — training loops, environment setup, network construction, and algorithm logic all tangled together. This template separates those concerns into three composable components, inspired by how [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) structures deep learning code:

| Component       | Owns                                              | Lightning analogy        |
|-----------------|---------------------------------------------------|--------------------------|
| **Environment** | Task definition, simulator, observation transforms | `LightningDataModule`    |
| **Algorithm**   | Networks, loss, optimizer, update rule, replay buffer | `LightningModule`        |
| **Trainer**     | Training loop, device management, data collection, callbacks | `Trainer`                |

Each component has a single responsibility. You can swap any one without touching the others — change the environment from CartPole to Atari without modifying your algorithm, or switch from DQN to PPO without rewriting your training loop.

## Main technologies

**[TorchRL](https://github.com/pytorch/rl)** — A PyTorch-native library for reinforcement learning that provides modular, composable primitives for environments, replay buffers, data collectors, and loss modules. It leverages [`TensorDict`](https://github.com/pytorch/tensordict) as a universal data carrier, making it easy to swap components without rewriting glue code, and supports GPU-accelerated batched simulation out of the box.

**[Hydra](https://github.com/facebookresearch/hydra)** — A configuration framework by Meta Research that lets you compose hierarchical configs from multiple YAML files and override any parameter from the command line. This makes it trivial to launch hyperparameter sweeps, compare algorithm variants, and keep every experiment setting version-controlled without touching Python code.

## Quick start

```shell
# clone template
git clone https://github.com/raphaelschwinger/torchrl-hydra-template
cd torchrl-hydra-template

# install requirements
uv sync

# activate virtual environment
source .venv/bin/activate

# run an experiment configured in configs/experiment e.g.:
python src/train.py experiment=reinforce/cartpole
```

## Architecture

### The three components

```
train.py  →  Trainer(algorithm, environment)
               ├── owns: device, env lifecycle, data collector, eval loop, callbacks
               └── calls: algorithm.step(batch) → metrics

Algorithm    →  owns: networks, loss, optimizer, replay buffer (if any)
               ├── setup(env)             — reads specs from a live env, builds networks
               ├── step(batch)            — processes collected data, returns metrics
               ├── get_policy()           — greedy policy for evaluation
               └── get_explore_policy()   — exploration policy for data collection

Environment  →  config wrapper + env factory
               ├── obs_shape, num_actions — metadata for network sizing
               └── make_env(num_envs, device) → TransformedEnv
```

The data flows in one direction: the **Trainer** creates environments from the **Environment** config, collects data using the **Algorithm**'s exploration policy, and passes batches to the **Algorithm** for learning. The Algorithm never creates environments or manages the training loop — it only defines *what* to optimize.

### Environment

The `Environment` class is a thin wrapper around TorchRL's environment creation. It holds the configuration (name, backend, observation transforms) and produces `TransformedEnv` instances on demand:

```python
from src.environments.environment import Environment

env = Environment(name="CartPole-v1", backend="gymnasium", obs_shape=[4], num_actions=2)
print(env.obs_shape)    # (4,) for CartPole
print(env.num_actions)  # 2 for CartPole

# Create a vectorized env on GPU
train_env = env.make_env(num_envs=8, device="cuda:0")
```

The Environment is a *factory*, not a live env holder. This lets the Trainer create separate train and eval environments and control their lifecycle. The underlying `make_env()` function in `src/environments/factory.py` handles backend-specific setup and vectorization.

**Supported backends:**
- **Gymnasium** — classic control (CartPole), Atari (Breakout, Pong), and any Gymnasium-compatible env. Transforms are specified as a list in the environment YAML, each with a `_target_` key and instantiated via `hydra.utils.instantiate`.
- **DeepMind Control** — continuous control tasks (Humanoid, Walker, etc.)
- **Envpool** — vectorized Atari/Gym environments via multi-threaded simulation

### Algorithm

The `BaseAlgorithm` defines what to optimize — networks, loss function, optimizer, and update rule. It exposes a hybrid interface: `step()` is the primary method the Trainer calls, with optional hooks for finer control.

**Required methods:**

| Method | Purpose |
|--------|---------|
| `setup(env)` | Build networks, loss module, and optimizer. Receives a live env to read `action_spec`, `observation_spec`, etc. |
| `step(batch)` | Process one batch of collected data and return a metrics dict. This is where the learning happens. |
| `get_policy()` | Return the greedy/deterministic policy (used for evaluation). |
| `get_explore_policy()` | Return the exploration policy (used for data collection). |

**Optional hooks:**

| Hook | Default | Used by |
|------|---------|---------|
| `on_batch_collected(batch)` | Return batch unchanged | DQN (stores transitions in replay buffer) |
| `should_skip_update(frames)` | `False` | DQN (skip gradient updates during warmup) |
| `on_step_complete(frames)` | No-op | DQN (epsilon decay, target network update) |
| `get_collector_config()` | Raises `NotImplementedError` | DQN, PPO (tells StepTrainer how to configure the data collector) |

The `step()` method is intentionally unconstrained — the algorithm decides what to do with the batch internally. For DQN, `step()` samples from the replay buffer and does a single gradient update. For PPO, `step()` computes GAE advantages and runs multiple epochs of minibatch updates. This keeps algorithm-specific logic where it belongs: in the algorithm.

**Built-in algorithms:**

| Algorithm | Type | Key features |
|-----------|------|--------------|
| **REINFORCE** | On-policy, episodic | Monte-Carlo returns, policy gradient |
| **DQN** | Off-policy, step-based | Replay buffer, target network, epsilon-greedy |
| **PPO** | On-policy, batch-based | GAE, clipped surrogate, multi-epoch updates |

### Trainer

The Trainer manages *how* training runs — the loop structure, device placement, data collection, evaluation, and callbacks. There are two trainer types for different RL paradigms:

#### EpisodeTrainer

For algorithms that learn from complete episodes (e.g., REINFORCE with Monte-Carlo returns).

```
while step < total_frames:
    episode = env.rollout(policy=algorithm.get_explore_policy())
    episode = algorithm.on_batch_collected(episode)
    metrics = algorithm.step(episode)
    step += len(episode)
    fire_callbacks(ON_STEP_END, metrics, step)
```

The EpisodeTrainer rolls out full episodes using `env.rollout()` and passes them directly to the algorithm. No `SyncDataCollector` is used.

#### StepTrainer

For algorithms that learn from fixed-size batches of transitions (e.g., DQN, PPO).

```
collector = SyncDataCollector(env, algorithm.get_explore_policy(), ...)

for batch in collector:
    batch = algorithm.on_batch_collected(batch)      # DQN: store in replay buffer
    step += batch.numel()
    if not algorithm.should_skip_update(step):        # DQN: skip during warmup
        metrics = algorithm.step(batch)               # PPO: multi-epoch updates inside
    algorithm.on_step_complete(step)                  # DQN: epsilon decay + target update
    fire_callbacks(ON_STEP_END, metrics, step)
```

The StepTrainer creates a `SyncDataCollector` using parameters from `algorithm.get_collector_config()`. The collector handles env stepping, policy inference, and batching. The algorithm's `step()` handles everything else.

#### What the Trainer owns

- **Device management** — resolves `accelerator` + `devices` config to a `torch.device`, sets the algorithm's device
- **Environment lifecycle** — creates train/eval environments via `Environment.make_env()`
- **Data collection** — creates and runs the `SyncDataCollector` (StepTrainer) or `env.rollout()` (EpisodeTrainer)
- **Evaluation** — `trainer.evaluate(num_episodes)` runs the greedy policy and returns reward statistics
- **Callbacks** — fires `ON_TRAIN_START`, `ON_STEP_END`, `ON_TRAIN_END` events
- **Checkpointing** — orchestrates save/load of algorithm state + step counter

## How algorithms map to trainers

| Algorithm | Trainer | Why |
|-----------|---------|-----|
| REINFORCE | `EpisodeTrainer` | Updates after full episodes using Monte-Carlo returns |
| DQN | `StepTrainer` | Collects fixed-size batches, stores in replay buffer, samples for updates |
| PPO | `StepTrainer` | Collects fixed-size batches, computes GAE, does multi-epoch minibatch updates |

The trainer type is specified in the experiment config:

```yaml
# configs/experiment/reinforce/cartpole.yaml
trainer:
  _target_: src.trainers.EpisodeTrainer

# configs/experiment/dqn/cartpole.yaml (uses default StepTrainer)
# no override needed
```

## Adding a new algorithm

### Step 1: Define the algorithm class

Create `src/algorithms/my_algo.py`. Hyperparameters are explicit keyword arguments on `__init__` — no separate config dataclass needed:

```python
import torch
from omegaconf import DictConfig
from src.algorithms.base import BaseAlgorithm, CollectorConfig, TrainingState

class MyAlgorithm(BaseAlgorithm):
    """One-line description.

    Args:
        cfg: Full Hydra config (trainer, logger, environment sections).
        device: Resolved torch.device; set by the Trainer.
        lr: Adam learning rate.
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
        self._network_cfg = network or {
            "architecture": "mlp", "hidden_sizes": [256, 256],
            "activation": "tanh", "layer_norm": False,
        }

    def setup(self, make_env):
        from omegaconf import OmegaConf
        from src.networks.factory import make_network
        env = make_env()  # call only if you need env specs
        obs_shape = tuple(self.cfg.environment.obs_shape)
        num_actions = int(self.cfg.environment.num_actions)
        net = make_network(OmegaConf.create(self._network_cfg), obs_shape, num_actions)
        # Build loss module, optimizer ...

    def get_policy(self):
        return self.actor

    def get_explore_policy(self):
        return self.actor

    def get_collector_config(self):
        return CollectorConfig(
            frames_per_batch=2048,
            total_frames=int(self.cfg.trainer.total_frames),
        )

    def step(self, batch):
        # Your update logic here
        return {"loss/policy": loss.item()}

    def _get_training_state(self) -> TrainingState: ...
    def _load_training_state(self, state: TrainingState) -> None: ...
```

### Step 2: Add a config file

Create `configs/algorithm/my_algo.yaml`. Only list values that differ from the Python defaults:

```yaml
_target_: src.algorithms.my_algo.MyAlgorithm
# Default values and parameter descriptions: src/algorithms/my_algo.py (MyAlgorithm.__init__)

lr: 3e-4
gamma: 0.99
network:
  architecture: mlp
  hidden_sizes: [256, 256]
  activation: tanh
  layer_norm: false
```

### Step 3: Create an experiment config

Create `configs/experiment/my_algo/cartpole.yaml`:

```yaml
# @package _global_
defaults:
  - override /algorithm: my_algo
  - override /environment: cartpole
  - override /logger: []
  - _self_

trainer:
  _target_: src.trainers.StepTrainer  # or src.trainers.EpisodeTrainer
  total_frames: 500_000
```

### Step 4: Run it

```shell
python src/train.py experiment=my_algo/cartpole
```

## Adding a model-based algorithm (e.g. Dreamer)

The framework handles model-based RL naturally. A Dreamer-style algorithm would:

1. **Use `StepTrainer`** — collects real environment transitions via `SyncDataCollector`
2. **Own a replay buffer** — stores real transitions (like DQN)
3. **Own a world model** — encoder, dynamics model, reward predictor, decoder
4. **Do everything in `step()`**:
   - Store real transitions in the replay buffer
   - Sample a batch from the buffer
   - Update the world model on real data
   - Imagine trajectories in latent space using the world model
   - Update actor-critic on imagined trajectories

```python
class DreamerAlgorithm(BaseAlgorithm):
    def setup(self, make_env):
        # Build world model (encoder, RSSM, reward head, decoder)
        # Build actor-critic
        # Build replay buffer
    
    def on_batch_collected(self, batch):
        self.replay_buffer.extend(batch)  # store real transitions
        return batch
    
    def step(self, batch):
        sample = self.replay_buffer.sample()
        
        # 1. Update world model on real data
        world_model_loss = self.update_world_model(sample)
        
        # 2. Imagine trajectories in latent space
        imagined = self.imagine(sample, horizon=15)
        
        # 3. Update actor-critic on imagined data
        actor_loss, critic_loss = self.update_actor_critic(imagined)
        
        return {
            "loss/world_model": world_model_loss,
            "loss/actor": actor_loss,
            "loss/critic": critic_loss,
        }
    
    def get_explore_policy(self):
        return self.actor  # with exploration noise
```

The key insight is that `step()` is unconstrained — the algorithm can do arbitrarily complex processing internally. The Trainer doesn't need to know about world models, imagination, or multi-phase updates.

## Configuration

### Hydra config structure

```
configs/
├── train.yaml              <- Top-level defaults (trainer, checkpoint)
├── eval.yaml               <- Evaluation defaults
├── algorithm/              <- Algorithm hyperparameters + network architecture
│   ├── reinforce.yaml
│   ├── dqn.yaml
│   └── ppo.yaml
├── environment/            <- Environment kwargs (name, backend, transforms list)
│   ├── cartpole.yaml
│   ├── atari_breakout.yaml
│   ├── atari_pong.yaml
│   └── dmc_humanoid.yaml
├── logger/                 <- Logger backends
│   ├── wandb.yaml
│   └── tensorboard.yaml
└── experiment/             <- Composed algorithm x environment configs
    ├── reinforce/cartpole.yaml
    ├── dqn/cartpole.yaml
    ├── dqn/atari_breakout.yaml
    ├── dqn/atari-pong.yaml
    └── ppo/dmc_humanoid.yaml
```

### Config override hierarchy

```
Python constructor defaults  ←  configs/algorithm/*.yaml  ←  experiment config  ←  CLI overrides
```

Each algorithm's `__init__` carries typed defaults for every hyperparameter. YAML files override those defaults. Experiment configs override the YAML. CLI args override everything:

```shell
python src/train.py experiment=dqn/cartpole algorithm.lr=1e-3 trainer.total_frames=2_000_000
```

### Algorithm hyperparameters

Hyperparameters live as **explicit keyword arguments on `__init__`**, not in a separate config dataclass. This gives IDE hover docs and completion while keeping Hydra override capability. The constructor is the single source of truth for:

1. **Typed defaults** — every hyperparameter has an explicit Python default so the algorithm is runnable without any YAML.
2. **Inline documentation** — the docstring `Args:` block explains what each parameter controls.
3. **Discoverability** — a reader opening `reinforce.py` immediately sees all knobs without cross-referencing YAML files.

`train.py` unpacks `cfg.algorithm` as `**kwargs` so YAML values flow through automatically:

```python
alg_kwargs = {k: v for k, v in OmegaConf.to_container(cfg.algorithm, resolve=True).items()
              if k != "_target_"}
algorithm = AlgClass(cfg=cfg, device=None, **alg_kwargs)
```

### Network architecture

Network architecture is embedded in each algorithm config — no separate config group. This keeps algorithm and network tightly coupled:

```yaml
# configs/algorithm/dqn.yaml (excerpt)
network:
  architecture: mlp        # "mlp" | "cnn_atari"
  hidden_sizes: [128, 128]
  activation: relu
```

Override in an experiment config when needed (e.g. CNN for Atari):

```yaml
# configs/experiment/dqn/atari_breakout.yaml (excerpt)
algorithm:
  network:
    architecture: cnn_atari
    conv_channels: [32, 64, 64]
    conv_kernels: [8, 4, 3]
    conv_strides: [4, 2, 1]
    fc_hidden: [512]
```

## Device selection

Device configuration follows PyTorch Lightning conventions:

```shell
# CPU (default)
python src/train.py experiment=reinforce/cartpole

# Single GPU
python src/train.py experiment=dqn/atari_breakout trainer.accelerator=gpu trainer.devices=[0]

# Second GPU
python src/train.py experiment=ppo/dmc_humanoid trainer.accelerator=gpu trainer.devices=[1]

# Apple Silicon
python src/train.py experiment=reinforce/cartpole trainer.accelerator=mps
```

## Logging

Pass a list of loggers — any combination of `wandb` and `tensorboard`:

```shell
# Both simultaneously
python src/train.py experiment=dqn/cartpole 'logger=[wandb,tensorboard]'

# TensorBoard only
python src/train.py experiment=reinforce/cartpole 'logger=[tensorboard]'

# No logging
python src/train.py experiment=reinforce/cartpole logger=[]
```

## Callbacks

The Trainer fires events at key points in training. Callbacks subscribe to these events:

| Event | When | Receives |
|-------|------|----------|
| `ON_TRAIN_START` | Before training loop | `state: {"cfg": cfg}` |
| `ON_STEP_END` | After each logged step | `metrics: dict, step: int` |
| `ON_TRAIN_END` | After training loop | `state: {"cfg": cfg}` |

**Built-in callbacks:**

- **ProgressCallback** — tqdm progress bar in the CLI
- **CheckpointCallback** — saves full training state (policy + optimizer + replay buffer metadata) every N steps and at the end
- **WandBLogger** — Weights & Biases integration
- **TensorBoardLogger** — TensorBoard integration

## Project structure

```
├── configs/                    <- Hydra configuration (compose-based)
│   ├── algorithm/              <- Per-algorithm hyperparameters + network architecture
│   │   ├── reinforce.yaml
│   │   ├── dqn.yaml
│   │   └── ppo.yaml
│   ├── environment/            <- Environment construction kwargs
│   │   ├── cartpole.yaml
│   │   ├── atari_breakout.yaml
│   │   └── dmc_humanoid.yaml
│   ├── logger/                 <- Logger backend configs
│   ├── experiment/             <- Composed algorithm x environment overrides
│   │   └── [algo]/[env].yaml
│   ├── train.yaml              <- Top-level train defaults
│   └── eval.yaml               <- Top-level eval defaults
│
├── src/                        <- Source code
│   ├── algorithms/             <- RL algorithm implementations
│   │   ├── base.py             <- BaseAlgorithm ABC (setup / step / get_policy / checkpoint)
│   │   ├── reinforce.py        <- REINFORCE (Monte-Carlo policy gradient)
│   │   ├── dqn.py              <- DQN (Q-learning + replay buffer + target network)
│   │   ├── ppo.py              <- PPO (actor-critic + GAE + clipped surrogate)
│   │   └── utils.py            <- Shared utilities (e.g. last_episode_return)
│   ├── environments/
│   │   ├── environment.py      <- Environment config wrapper + factory
│   │   └── factory.py          <- make_env: Gymnasium + dm_control + transforms
│   ├── networks/
│   │   └── factory.py          <- MLP, AtariCNN (dispatched from algorithm config)
│   ├── trainers/               <- Trainer implementations
│   │   ├── BaseTrainer.py      <- BaseTrainer ABC, TrainerEvent, Callback, fire_callbacks
│   │   ├── EpisodeTrainer.py   <- EpisodeTrainer (episodic rollouts via env.rollout)
│   │   └── StepTrainer.py      <- StepTrainer (fixed-size batches via SyncDataCollector)
│   ├── callbacks/
│   │   ├── logger.py           <- WandBLogger, TensorBoardLogger
│   │   ├── checkpoint.py       <- CheckpointCallback
│   │   └── progress.py         <- ProgressCallback (tqdm bar)
│   ├── utils/                  <- Device resolution, seeding, callback builders
│   ├── train.py                <- Training entry point
│   └── eval.py                 <- Evaluation entry point
│
├── tests/
│   └── test_smoke.py           <- One training cycle per experiment config
│
├── pyproject.toml              <- Dependencies + build config (uv / hatchling)
└── uv.lock                     <- Dependency lock file
```

## Smoke tests

```shell
pytest tests/test_smoke.py -v
```

Each test loads the full experiment config, applies minimal-frame overrides (CPU, no logging, small buffer), and asserts that one complete training cycle runs without error.

## Acknowledgements

This project builds on the ideas pioneered by [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) by @ashleve and further refined in [yet-another-lightning-hydra-template](https://github.com/gorodnitskiy/yet-another-lightning-hydra-template) by @gorodnitskiy. Their work on combining structured Hydra configs with clean training pipelines served as the foundation; this template adapts that philosophy to the reinforcement learning setting with TorchRL.
