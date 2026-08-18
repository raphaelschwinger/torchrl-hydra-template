# Architecture

```
train.py  ->  Trainer(algorithm, environment)
                ├── owns: device, env lifecycle, Collector, eval, callbacks, checkpoints
                │         (defaults: checkpoints/last.pt at train end, then a final
                │          eval of evaluation.final_num_episodes episodes)
                ├── logs: every metric row against global_step (agent steps)
                └── calls: algorithm.step(batch) -> metrics

Evaluation   ->  the measurement protocol (configs/evaluation/<benchmark>.yaml)
               ├── eval_environment    — env stack to measure on (null = reuse train)
               ├── every_n_steps       — periodic eval cadence (0 = final only)
               ├── num_episodes / final_num_episodes
               ├── policy              — eval | explore
               └── canonical_source    — train | eval, feeds charts/episodic_return

Algorithm    ->  owns: network, replay buffer, loss, optimiser, exploration,
                       collector config (frames_per_batch, init_random_frames, ...)
               ├── setup(make_env)        — read env specs, build everything
               ├── step(batch)            — anneal eps, store, sample, update
               ├── get_policy()           — greedy policy (eval)
               ├── get_explore_policy()   — eps-greedy policy (collection)
               └── get_collector_config() — frames_per_batch + init_random_frames

Environment  ->  factory: env name + transforms list
               └── make_env(num_envs, device) -> TransformedEnv
```

See [Evaluation & benchmarks](evaluation.md) for the evaluation protocol in detail.

## Algorithm

The `BaseAlgorithm` API is small:

| Method                    | Purpose                                                           |
|---------------------------|-------------------------------------------------------------------|
| `setup(make_env)`         | Build network, replay buffer, loss, optimiser. Read env specs by calling `make_env()`. |
| `step(batch)`             | Process one batch and return metrics. Where the learning happens. |
| `get_policy()`            | Greedy policy used by `trainer.evaluate()`.                       |
| `get_explore_policy()`    | Exploration policy used by the data collector.                    |
| `get_collector_config()`  | Tells the trainer how to size the `Collector`.                    |

`step()` is intentionally unconstrained — the algorithm decides what to do with the
batch. For DQN that means: anneal epsilon, store, skip during warm-up, otherwise
loop `num_updates` of (sample → loss → backward → optimiser → target update).

## Algorithm hyperparameters

Hyperparameters live as **explicit keyword arguments on `__init__`**, not in a
config dataclass:

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
        batch_size: int = 128,
        max_grad_norm: float = 10.0,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        annealing_frames: int = 250_000,
        frames_per_batch: int = 1_000,
        init_random_frames: int = 10_000,
        num_updates: int = 100,
        hard_update_freq: int = 50,
        ...
    ): ...
```

This buys three things:

1. **Typed defaults** — every hyperparameter has an explicit Python default so the
   algorithm is runnable without any YAML.
2. **Inline documentation** — IDE hover shows you the parameter and its default.
3. **Discoverability** — opening `src/algorithms/dqn/dqn.py` shows every knob without YAML lookups.

`replay_buffer` and `network` are `Callable` factories rather than scalars because
they encode design decisions (which storage backend, what MLP shape). Their defaults
live in `src/algorithms/dqn/dqn.py` as constructor kwargs; in YAML they are
`_partial_` blocks that `train.py` turns into real callables:

```python
algorithm = instantiate(cfg.algorithm, device=None)   # recursive: _partial_ -> callable
```

Networks that have more than one variant live in their own config group, so a
swap replaces the whole node. Hydra *merges* dicts, so patching only
`network._target_` would leave the previous option's kwargs behind — silently,
when the new factory happens to accept them:

```shell
python src/train.py experiment=dqn/gym algorithm/network=nature_dqn
python src/train.py experiment=ppo/dmc algorithm/policy=nature_cnn_categorical
```

## Environment

Just an env name plus an explicit transforms list:

```yaml
# configs/environment/gym.yaml
task: CartPole-v1        # the single override axis
name: ${environment.task}
transforms:
  - _target_: torchrl.envs.transforms.StepCounter
```

Every environment config exposes `task`, so switching task never means editing
YAML: `environment.task=Acrobot-v1`.

For envs that need extra `GymEnv` constructor arguments (e.g. `frame_skip`,
`from_pixels` for pixel-based Atari), pass them via `gym_kwargs`, and pin the
gym backend with `gym_backend`:

```yaml
# configs/environment/ale.yaml
task: Pong
name: ALE/${environment.task}-v5
gym_backend: gymnasium
gym_kwargs:
  frame_skip: 4
  from_pixels: true
  pixels_only: false
  categorical_action_encoding: true
transforms:
  - _target_: torchrl.envs.NoopResetEnv
    noops: 30
    random: true
  # ...
```

`make_env` in `src/environments/factory.py` instantiates each transform fresh per
call (so stateful transforms like `CatFrames` get independent state), composes
them on top of `GymEnv(name, **gym_kwargs)`, and wraps in `ParallelEnv` when
`num_envs > 1`.

Backends supported: **gymnasium** (default) and **dm_control**. For DeepMind
Control Suite tasks set `backend: dm_control` and give a `<domain>-<task>` id as
`task`; the factory splits it on the first hyphen (dm_control uses underscores
inside its own names, so this is unambiguous), builds a
`torchrl.envs.DMControlEnv`, and defaults `MUJOCO_GL=disabled` (headless, no
rendering) unless you set it yourself:

```yaml
# configs/environment/dmc.yaml
backend: dm_control
task: cheetah-run
transforms:
  - _target_: torchrl.envs.transforms.FrameSkipTransform   # action repeat 2
    frame_skip: 2
  - _target_: torchrl.envs.transforms.CatTensors            # flatten obs dict
    in_keys: [position, velocity]
    out_key: observation
  # ...
```

For the eval-side env configs, the periodic/final evaluation cadence, and the
`train:` / `eval:` stage flags, see [Evaluation & benchmarks](evaluation.md).

## Trainer

`StepTrainer` creates a `torchrl.collectors.Collector` from the algorithm's
collector config and the trainer-level `total_frames`, then iterates:

```python
for batch in self.collector:
    self._step += batch.numel()
    metrics = self.algorithm.step(batch)
    ep_rewards, ep_lengths, _ = _batch_metrics(batch)
    self.log_episodes(ep_rewards, ep_lengths, self._step, source="train")
    if self._should_log(...):
        self.log_metrics(row, self._step)
        fire_callbacks(ON_STEP_END, self.callbacks, metrics=row, step=self._step)
    if self._should_eval(...):
        self.run_evaluation(evaluation.num_episodes, step=self._step)
```

`BaseTrainer` owns:
- **Device** — resolves `accelerator` + `devices` to `torch.device`.
- **Env lifecycle** — creates train/eval envs via `Environment.make_env()`.
- **Eval** — `run_evaluation()` / `run_final_evaluation()` follow the
  `configs/evaluation/` protocol; the eval env is built once and reused, and
  module `.training` flags are restored around every rollout.
- **Metrics** — `log_metrics()` / `log_episodes()` inject `global_step` and
  `frames` into every row.
- **Callbacks** — fires `ON_TRAIN_START`, `ON_METRICS`, `ON_STEP_END`, `ON_TRAIN_END`.
- **Checkpoints** — orchestrates save/load of algorithm state.

Trainer config knobs (`total_frames`, `seed`, `accelerator`, `devices`,
`num_envs`, `log_every_n_steps`) only control how training runs, never what is
learned.

## Configuration

```
configs/
├── train.yaml              <- top-level defaults (train/eval flags, env_id,
│                              exp_name, seed, run_name, checkpoint)
├── eval.yaml               <- evaluation entry point defaults
├── trainer/
│   ├── default.yaml        <- the loop: seed, total_frames, num_envs, logging
│   ├── cpu.yaml
│   ├── gpu.yaml            <- accelerator: gpu (set devices=[N] on the CLI)
│   └── eval.yaml           <- total_frames: 0 (used by eval.yaml)
├── algorithm/              <- one config per algorithm class; no env specifics
│   ├── dqn.yaml            <- DQN HPs
│   ├── ddpg.yaml           <- DDPG HPs
│   ├── a2c.yaml            <- A2C HPs
│   ├── ppo.yaml            <- PPO HPs (cleanRL continuous-action defaults)
│   ├── rainbow.yaml        <- Rainbow HPs
│   ├── tdmpc2.yaml         <- TD-MPC2 HPs (model_size=5)
│   ├── dreamer.yaml        <- DreamerV3 (+ dreamerpro.yaml, r2dreamer.yaml)
│   ├── network/            <- swappable Q-network (DQN)
│   │   ├── mlp_q.yaml      <- state obs
│   │   └── nature_dqn.yaml <- pixel obs
│   ├── policy/             <- swappable actor/critic/trunk (PPO)
│   │   ├── mlp_normal.yaml <- continuous control, state obs
│   │   └── nature_cnn_categorical.yaml  <- discrete control, pixel obs
│   └── dreamer/            <- model-size presets (12m ... 400m)
├── environment/            <- one config per benchmark; pick task with `task`
│   ├── gym.yaml            <- gymnasium state obs (classic control + MuJoCo)
│   ├── dmc.yaml            <- dm_control, task: <domain>-<task>
│   ├── ale.yaml            <- Atari, standard protocol (train)
│   ├── ale_eval.yaml       <- same without EndOfLife / Sign / VecNorm
│   ├── atari100k.yaml      <- Atari-100k protocol (train)
│   └── atari100k_eval.yaml <- same without EpisodicLife / Sign
├── evaluation/             <- measurement protocol; also selects the eval env
│   ├── none.yaml           <- base schema; training stream only, no rollouts
│   ├── gym.yaml            <- final eval only, canonical_source: train
│   ├── ale.yaml            <- ale_eval stack, canonical_source: train
│   ├── atari100k.yaml      <- every 10k + 100 final episodes, canonical_source: eval
│   ├── atari100k_native.yaml <- same protocol on the experiment's own stack
│   └── dmc.yaml            <- periodic every 10k (TD-MPC2 upstream cadence)
├── logger/
│   ├── wandb.yaml
│   └── tensorboard.yaml
├── paths/default.yaml
└── experiment/              <- algorithm x benchmark, plus task/budget overrides
    ├── dqn/{gym,ale}.yaml
    ├── ddpg/gym.yaml
    ├── a2c/gym.yaml
    ├── ppo/{dmc,ale}.yaml
    ├── rainbow/atari100k.yaml   <- the Data-Efficient Rainbow preset
    ├── tdmpc2/dmc.yaml
    ├── dreamer/{atari100k,dmc}.yaml
    └── bbf/{atari100k,atari100k_rr8}.yaml
```

Anything that depends on the *task* — pixel networks, replay capacity,
exploration schedules, episode length, training budget — lives in the
experiment. `configs/algorithm/*.yaml` describes the algorithm only.

### Override hierarchy

```
Python __init__ defaults  <-  configs/algorithm/dqn.yaml  <-  experiment config  <-  CLI overrides
```

```shell
python src/train.py experiment=dqn/gym algorithm.lr=1e-3 trainer.total_frames=200_000
```

## Logging

Defaults: plain CLI runs log to **tensorboard**; runs launched via
`experiment=...` log to **wandb**. Override with any combination of `wandb` and
`tensorboard`:

```shell
python src/train.py experiment=dqn/gym 'logger=[wandb,tensorboard]'
python src/train.py experiment=dqn/gym 'logger=[tensorboard]'
python src/train.py experiment=dqn/gym logger=[]
```

## Callbacks

The trainer fires events at key points:

| Event             | When                              | Receives                                  |
|-------------------|-----------------------------------|-------------------------------------------|
| `ON_TRAIN_START`  | Before the loop                   | `state: {"cfg": cfg}`                     |
| `ON_METRICS`      | Per metric row (incl. per episode)| `metrics: dict, step: int`                |
| `ON_STEP_END`     | After each logged step            | `metrics: dict, step: int`                |
| `ON_TRAIN_END`    | After the loop                    | `state: {"cfg": cfg, "summary": dict}`    |

`ON_METRICS` and `ON_STEP_END` are separate on purpose. `ON_METRICS` means "one
row of metrics at this x position" and fires once per completed episode —
loggers implement it. `ON_STEP_END` means "the loop crossed a log boundary" and
fires only there — the progress bar and checkpointer implement it, and would
misbehave if driven per episode.

Built-in callbacks: `ProgressCallback` (tqdm bar), `CheckpointCallback`,
`WandBLogger`, `TensorBoardLogger`.
