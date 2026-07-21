<div align="center">

# TorchRL Hydra Template

A clean, modular template for deep reinforcement learning research.<br>
Click on [<kbd>Use this template</kbd>](https://github.com/raphaelschwinger/torchrl-hydra-template/generate) to initialize a new repository.

_Suggestions are always welcome!_

</div>

## Philosophy

Reinforcement learning code tends to become monolithic — training loop, environment
setup, network construction, replay buffer, and update rule all tangled together.
This template enforces a hard split into three components, inspired by how
[PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) structures
deep learning code:

| Component       | Owns                                                                  | Lightning analogy        |
|-----------------|-----------------------------------------------------------------------|--------------------------|
| **Algorithm**   | Everything that affects learning: network, replay buffer, loss, optimiser, exploration, target-net schedule, collector config. **All hyperparameters live here.** | `LightningModule`        |
| **Trainer**     | The loop. Device placement, data collection, logging, callbacks, checkpointing. **No knobs that affect reward.** | `Trainer`                |
| **Environment** | Fixed task definition: env name + transform list. Independent of algorithm. | `LightningDataModule`    |

Two derived rules:

1. **RL algorithm code reads like the paper.** `step()` is short and corresponds to
   the update equations. The DQN file looks like Mnih et al. (2015)'s pseudocode,
   not framework glue.
2. **Anything that influences reward or sample efficiency lives in the algorithm.**
   If a knob shifts the learning curve, it goes on `__init__`. The trainer cannot
   silently change behaviour.

**Implemented experiments:**

| Algorithm | Environment    | Config                          |
|-----------|----------------|---------------------------------|
| DQN       | CartPole-v1    | `experiment=dqn/cartpole`       |
| DQN       | ALE/Pong-v5    | `experiment=dqn/pong`           |
| DDPG      | HalfCheetah-v4 | `experiment=ddpg/halfcheetah`   |
| A2C       | HalfCheetah-v4 | `experiment=a2c/halfcheetah`    |
| DreamerV3       | ALE/Jamesbond-v5<br>(Atari100k) | `experiment=dreamer/atari100k`  |

Other algorithms will follow.

### Algorithm documentation

Each algorithm lives in its own package with theory, pseudocode, and benchmark
results. Experimental metrics are tracked on
[W&B (LatentLab/torchrl-hydra-template)](https://wandb.ai/LatentLab/torchrl-hydra-template/table).

| Algorithm | Docs |
|-----------|------|
| DQN | [`src/algorithms/dqn/README.md`](src/algorithms/dqn/README.md) |
| DDPG | [`src/algorithms/ddpg/README.md`](src/algorithms/ddpg/README.md) |
| A2C | [`src/algorithms/a2c/README.md`](src/algorithms/a2c/README.md) |
| DreamerV3       | [`src/algorithms/dreamer/README.md`](src/algorithms/dreamer/README.md) |

After new benchmark training runs, tag them with `template` on W&B and refresh
the markdown tables in each algorithm README:

```shell
python scripts/update_algo_results.py              # rewrite tables from W&B
python scripts/update_algo_results.py --dry-run    # preview without writing
```

Requires `wandb login` (or `WANDB_API_KEY`). By default the script reads finished
runs tagged `template` from `LatentLab/torchrl-hydra-template`. Use
`--entity`, `--project`, `--tag`, or `--algo dqn` to override scope.

## Main technologies

**[TorchRL](https://github.com/pytorch/rl)** — A PyTorch-native library for
reinforcement learning that provides modular primitives for environments, replay
buffers, data collectors, and loss modules. It uses
[`TensorDict`](https://github.com/pytorch/tensordict) as a universal data carrier,
making it easy to swap components without rewriting glue code.

**[Hydra](https://github.com/facebookresearch/hydra)** — A configuration framework
that lets you compose hierarchical configs from multiple YAML files and override
any parameter from the command line. Trivial to launch hyperparameter sweeps and
keep every experiment setting version-controlled.

## Quick start

```shell
git clone https://github.com/raphaelschwinger/torchrl-hydra-template
cd torchrl-hydra-template

uv sync
source .venv/bin/activate

python src/train.py experiment=dqn/cartpole
```

A full training run (500k frames, ~7 minutes on CPU) reproduces the torchrl SOTA
reference for DQN-CartPole.

For Atari Pong (mirrors the torchrl SOTA `dqn_atari.py` reference, 40M frames on
GPU):

```shell
python src/train.py experiment=dqn/pong
```

## Architecture

```
train.py  ->  Trainer(algorithm, environment)
                ├── owns: device, env lifecycle, Collector, eval, callbacks, checkpoints
                └── calls: algorithm.step(batch) -> metrics

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

### Algorithm

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

### Algorithm hyperparameters

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
live in `src/algorithms/dqn/dqn.py` as constructor kwargs and inline lambdas. To
swap them, edit those defaults or pass a different factory in code.

`train.py` unpacks `cfg.algorithm` as `**kwargs`, so YAML values override defaults
and CLI overrides override YAML:

```python
alg_kwargs = {k: v for k, v in OmegaConf.to_container(cfg.algorithm, resolve=True).items()
              if k != "_target_"}
algorithm = AlgClass(device=None, **alg_kwargs)
```

### Environment

Just an env name plus an explicit transforms list:

```yaml
# configs/environment/cartpole.yaml
name: CartPole-v1
transforms:
  - _target_: torchrl.envs.transforms.StepCounter
```

For envs that need extra `GymEnv` constructor arguments (e.g. `frame_skip`,
`from_pixels` for pixel-based Atari), pass them via `gym_kwargs`, and pin the
gym backend with `gym_backend`:

```yaml
# configs/environment/pong_train.yaml
name: ALE/Pong-v5
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

Backends supported: **gymnasium**.

#### Separate evaluation environment

For tasks where training-time observations differ from what evaluation should
see (e.g. Atari, where the SOTA reference clips rewards and ends episodes on
life loss during training but not during eval), declare a second env via the
Hydra package override:

```yaml
# configs/experiment/dqn/pong.yaml
defaults:
  - override /environment: pong_train
  - override /environment@eval_environment: pong_eval
```

When `eval_environment` is set, `BaseTrainer.evaluate()` uses it; otherwise it
falls back to `environment`.

### Trainer

`StepTrainer` creates a `torchrl.collectors.Collector` from the algorithm's
collector config and the trainer-level `total_frames`, then iterates:

```python
for batch in self.collector:
    self._step += batch.numel()
    metrics = self.algorithm.step(batch)
    if self._should_log(...):
        fire_callbacks(ON_STEP_END, self.callbacks, metrics=metrics, step=self._step)
```

`BaseTrainer` owns:
- **Device** — resolves `accelerator` + `devices` to `torch.device`.
- **Env lifecycle** — creates train/eval envs via `Environment.make_env()`.
- **Eval** — `evaluate(num_episodes)` runs the greedy policy.
- **Callbacks** — fires `ON_TRAIN_START`, `ON_STEP_END`, `ON_TRAIN_END` events.
- **Checkpoints** — orchestrates save/load of algorithm state.

Trainer config knobs (`total_frames`, `seed`, `accelerator`, `devices`,
`num_envs`, `log_every_n_steps`) only control how training runs, never what is
learned.

## Configuration

```
configs/
├── train.yaml              <- top-level defaults (trainer, checkpoint)
├── eval.yaml               <- evaluation defaults
├── algorithm/
│   ├── dqn.yaml            <- DQN HPs (CartPole defaults)
│   ├── dqn_atari.yaml      <- DQN HPs (Atari/NatureDQN defaults)
│   ├── ddpg.yaml           <- DDPG HPs (HalfCheetah defaults)
│   └── a2c.yaml            <- A2C HPs (HalfCheetah/MuJoCo defaults)
├── environment/
│   ├── cartpole.yaml       <- env name + transforms
│   ├── pong_train.yaml     <- Pong with EndOfLife + Sign + VecNorm (training)
│   ├── pong_eval.yaml      <- Pong without those transforms (evaluation)
│   └── halfcheetah.yaml    <- HalfCheetah-v4 (DoubleToFloat + InitTracker)
├── logger/
│   ├── wandb.yaml
│   └── tensorboard.yaml
├── paths/default.yaml
└── experiment/
    ├── dqn/
    │   ├── cartpole.yaml   <- composed: algorithm + environment + trainer overrides
    │   └── pong.yaml       <- composed Atari Pong experiment
    ├── ddpg/
    │   └── halfcheetah.yaml <- composed DDPG HalfCheetah experiment
    └── a2c/
        └── halfcheetah.yaml <- composed A2C HalfCheetah experiment
```

### Override hierarchy

```
Python __init__ defaults  <-  configs/algorithm/dqn.yaml  <-  experiment config  <-  CLI overrides
```

```shell
python src/train.py experiment=dqn/cartpole algorithm.lr=1e-3 trainer.total_frames=200_000
```

## Logging

Defaults: plain CLI runs log to **tensorboard**; runs launched via
`experiment=...` log to **wandb**. Override with any combination of `wandb` and
`tensorboard`:

```shell
python src/train.py experiment=dqn/cartpole 'logger=[wandb,tensorboard]'
python src/train.py experiment=dqn/cartpole 'logger=[tensorboard]'
python src/train.py experiment=dqn/cartpole logger=[]
```

## Callbacks

The trainer fires events at key points:

| Event             | When                  | Receives                            |
|-------------------|-----------------------|-------------------------------------|
| `ON_TRAIN_START`  | Before the loop       | `state: {"cfg": cfg}`               |
| `ON_STEP_END`     | After each logged step| `metrics: dict, step: int`          |
| `ON_TRAIN_END`    | After the loop        | `state: {"cfg": cfg}`               |

Built-in callbacks: `ProgressCallback` (tqdm bar), `CheckpointCallback`,
`WandBLogger`, `TensorBoardLogger`.

## Adding a new algorithm

1. Create `src/algorithms/my_algo/my_algo.py` with an `__init__.py` re-export and
   `README.md` (theory, pseudocode, W&B results). Follow the kwargs pattern above.
   Use `Callable` factories for design choices (inline lambdas, `functools.partial`,
   or small helpers).
2. Implement `setup(make_env)`, `step(batch)`, `get_policy()`,
   `get_explore_policy()`, `get_collector_config()`,
   `_get_training_state()`, `_load_training_state()`.
3. Add `configs/algorithm/my_algo.yaml` mirroring scalar defaults from `__init__`.
4. Add `configs/experiment/my_algo/<env>.yaml` composing your algorithm + env.
5. Add a smoke test in `tests/test_smoke.py`.
6. Update `README.md` and `AGENTS.md`.

## Smoke test

```shell
pytest tests/test_smoke.py -v
```

Loads the experiment config, applies minimal-frame overrides, and asserts that
one full training cycle runs without error.


## Contribution

Template improvements are welcome. If you started from
[**Use this template**](https://github.com/raphaelschwinger/torchrl-hydra-template/generate),
your repository has no git link to the upstream template by default — add a remote
manually (see below).

Not every change in a derived project belongs upstream. Contribute back when the
change is **template-worthy**: a generic algorithm, trainer or callback fix,
reusable environment config, documentation, or smoke test. Keep project-specific
work (pretraining experiments, paper configs, custom paths) in your own repo.

### Pulling template updates into your repo

One-time setup:

```shell
git remote add upstream https://github.com/raphaelschwinger/torchrl-hydra-template.git
git fetch upstream
```

How you sync depends on how you created your repo.

#### Fork or clone of the template

Histories are already linked — merge or rebase works out of the box:

```shell
git checkout main
git merge upstream/main          # or: git rebase upstream/main
# resolve conflicts in shared files (src/, configs/, tests/)
pytest tests/test_smoke.py -v
```

#### Created via [Use this template](https://github.com/raphaelschwinger/torchrl-hydra-template/generate)

GitHub starts a fresh repository with a new initial commit. The files match the
template, but git sees **no shared history**, so `git merge upstream/main` fails
with *refusing to merge unrelated histories*.

**Recommended — one-time history reconnect.** Rebase your project-specific
commits onto `upstream/main` so regular merges work from then on:

```shell
git remote add upstream https://github.com/raphaelschwinger/torchrl-hydra-template.git
git fetch upstream

# <initial-commit> = your repo's first commit (see git log --oneline --reverse)
git rebase --onto upstream/main <initial-commit> main
git push --force-with-lease origin main
```

Example: if `git log --oneline --reverse | head -1` shows `3a43db2 Initial
commit`, run `git rebase --onto upstream/main 3a43db2 main`.

After reconnecting, sync the same way as a fork:

```shell
git checkout main
git merge upstream/main
pytest tests/test_smoke.py -v
```

This rewrites history on `main`. Only run it once, early in the project, or
coordinate with collaborators before force-pushing.

**Without reconnecting** — pull in upstream changes selectively:

```shell
git cherry-pick <commit-sha>            # one upstream commit at a time

# — or — copy changed files manually
git diff upstream/main -- src/algorithms/dqn/dqn.py
pytest tests/test_smoke.py -v
```

Once you diverge, conflicts are likely — resolve them only in shared template
files.

### Feeding improvements back to the template

No separate fork clone is required. What matters is a **clean branch**: one
branched from `upstream/main` that contains only template-relevant changes, not
your full research history. You can create that branch in your existing derived
repo using the same `upstream` remote as above.

| Option | When to use |
|--------|-------------|
| **Pull request** | You have a focused, template-ready change |
| **GitHub issue** | Idea, bug report, or discussion before coding |
| **Cherry-pick / extract** | The improvement is buried in mixed commits on `main` |

**Pull request workflow** (works in your existing repo):

```shell
# one-time (if not already done for sync)
git remote add upstream https://github.com/raphaelschwinger/torchrl-hydra-template.git
git fetch upstream

# branch from upstream, not from your diverged main
git checkout -b contribute/my-fix upstream/main

# bring in your change (pick one):
git cherry-pick <commit-sha>            # if the commit is already template-only
# — or — copy changed files manually and commit

pytest tests/test_smoke.py -v
git push -u origin contribute/my-fix
# open PR: your-repo/contribute/my-fix → torchrl-hydra-template/main
```

Where to push and open the PR:

- **Maintainers / collaborators with write access:** push the branch directly to
  `upstream` and open an in-repo PR — no GitHub fork needed.
- **Everyone else:** push the branch to your repo (`origin`) and open a
  **cross-repo PR** from `your-repo:contribute/my-fix` →
  `torchrl-hydra-template:main`. GitHub supports this without cloning a
  separate fork.
- **Optional fork:** only if you prefer a dedicated template checkout; functionally
  equivalent to the branch-from-upstream flow above.

"Clean" here means **isolated diffs**, not a second repository.

**PR checklist:**

- Smoke test passes (`pytest tests/test_smoke.py -v`).
- Update `README.md` and `AGENTS.md` if you add or rename algorithms or change
  conventions (see [Adding a new algorithm](#adding-a-new-algorithm)).
- Keep PRs scoped — one algorithm, one bug fix, or one trainer improvement is
  easier to review than a large research dump.

**Issue workflow:** open an issue on
[torchrl-hydra-template](https://github.com/raphaelschwinger/torchrl-hydra-template/issues),
link to a minimal repro or branch in your repo, and explain why the change
belongs in the template rather than staying project-specific.

## Acknowledgements

This project builds on the ideas pioneered by
[lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) by
@ashleve and further refined in
[yet-another-lightning-hydra-template](https://github.com/gorodnitskiy/yet-another-lightning-hydra-template)
by @gorodnitskiy. Their work on combining structured Hydra configs with clean
training pipelines served as the foundation; this template adapts that philosophy
to the reinforcement learning setting with TorchRL.

The DQN reference implementation in `src/algorithms/dqn/dqn.py` is modelled on the
torchrl SOTA reference at
[`pytorch/rl/sota-implementations/dqn/dqn_cartpole.py`](https://github.com/pytorch/rl/blob/main/sota-implementations/dqn/dqn_cartpole.py).
The DDPG reference implementation in `src/algorithms/ddpg/ddpg.py` is modelled on the
torchrl SOTA reference at
[`pytorch/rl/sota-implementations/ddpg/ddpg.py`](https://github.com/pytorch/rl/blob/main/sota-implementations/ddpg/ddpg.py).
The A2C reference implementation in `src/algorithms/a2c/a2c.py` is modelled on the
torchrl SOTA reference at
[`pytorch/rl/sota-implementations/a2c/a2c_mujoco.py`](https://github.com/pytorch/rl/blob/main/sota-implementations/a2c/a2c_mujoco.py).
