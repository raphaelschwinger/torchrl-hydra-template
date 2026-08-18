<div align="center">

# TorchRL Hydra Template

A clean, modular template for deep reinforcement learning research.<br>
Click on [<kbd>Use this template</kbd>](https://github.com/raphaelschwinger/torchrl-hydra-template/generate) to initialize a new repository.

_Suggestions are always welcome!_

</div>

## Philosophy

Reinforcement learning code tends to become monolithic — training loop, environment
setup, network construction, replay buffer, and update rule all tangled together.
This template enforces a hard split into five components, inspired by how
[PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) structures
deep learning code:

| Component       | Owns                                                                  | Lightning analogy        |
|-----------------|-----------------------------------------------------------------------|--------------------------|
| **Algorithm**   | Everything that affects learning: network, replay buffer, loss, optimiser, exploration, target-net schedule, collector config. **All hyperparameters live here.** | `LightningModule`        |
| **Trainer**     | The loop. Device placement, data collection, logging, callbacks, checkpointing. **No knobs that affect reward.** | `Trainer`                |
| **Environment** | One benchmark: backend, preprocessing stack, and a `task` key naming the task within it. Independent of algorithm. | `LightningDataModule`    |
| **Evaluation**  | The measurement protocol: eval env stack, cadence, episode count, policy mode, and which stream is canonical. **Also no knobs that affect reward.** | —                        |
| **Experiment**  | One algorithm × one benchmark, plus everything that depends on the *task*: which network, what budget, which schedules. | —                        |

Four derived rules:

1. **RL algorithm code reads like the paper.** `step()` is short and corresponds to
   the update equations. The DQN file looks like Mnih et al. (2015)'s pseudocode,
   not framework glue.
2. **Anything that influences reward or sample efficiency lives in the algorithm.**
   If a knob shifts the learning curve, it goes on `__init__`. The trainer cannot
   silently change behaviour.
3. **The algorithm config knows nothing about the task.** Pixel networks, training
   budgets and exploration schedules are properties of a benchmark, not of DQN, so
   they live in the experiment. That is what keeps one `configs/algorithm/dqn.yaml`
   serving both CartPole and Atari.
4. **How you measure is separate from what you train.** Evaluation cadence, episode
   counts and the eval env stack are one override (`evaluation=atari100k`), and
   every run logs the same metric names on the same axis — so results are
   comparable across algorithms, and directly consumable by
   [openrlbenchmark](https://github.com/openrlbenchmark/openrlbenchmark).

## Implemented algorithms


| Algorithm | Reference | Docs |
|-----------|-----------|------|
| DQN | Mnih et al. (2015), [*Human-level control through deep reinforcement learning*](https://www.nature.com/articles/nature14236) | [`src/algorithms/dqn/README.md`](src/algorithms/dqn/README.md) |
| DDPG | Lillicrap et al. (2016), [*Continuous control with deep reinforcement learning*](https://arxiv.org/abs/1509.02971) | [`src/algorithms/ddpg/README.md`](src/algorithms/ddpg/README.md) |
| A2C | Mnih et al. (2016), [*Asynchronous Methods for Deep Reinforcement Learning*](https://arxiv.org/abs/1602.01783) | [`src/algorithms/a2c/README.md`](src/algorithms/a2c/README.md) |
| PPO | Schulman et al. (2017), [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347) | [`src/algorithms/ppo/README.md`](src/algorithms/ppo/README.md) |
| TD-MPC2 | Hansen, Su & Wang (2024), [*TD-MPC2: Scalable, Robust World Models for Continuous Control*](https://arxiv.org/abs/2310.16828) | [`src/algorithms/tdmpc2/README.md`](src/algorithms/tdmpc2/README.md) |
| DreamerV3 | Hafner et al. (2025), [*Mastering Diverse Control Tasks through World Models*](https://www.nature.com/articles/s41586-025-08744-2) | [`src/algorithms/dreamer/README.md`](src/algorithms/dreamer/README.md) |
| Rainbow / DER | Hessel et al. (2018), [*Rainbow: Combining Improvements in Deep Reinforcement Learning*](https://arxiv.org/abs/1710.02298); van Hasselt et al. (2019), [*When to Use Parametric Models in Reinforcement Learning?*](https://arxiv.org/abs/1906.05243) (DER) | [`src/algorithms/rainbow/README.md`](src/algorithms/rainbow/README.md) |
| BBF | Schwarzer et al. (2023), [*Bigger, Better, Faster: Human-level Atari with human-level efficiency*](https://arxiv.org/abs/2305.19452) | [`src/algorithms/bbf/README.md`](src/algorithms/bbf/README.md) |

Other algorithms will follow. Experimental metrics are tracked on
[W&B (LatentLab/torchrl-hydra-template)](https://wandb.ai/LatentLab/torchrl-hydra-template/table).

After new benchmark training runs, tag them with `template` on W&B and refresh
the markdown tables in each algorithm README:

```shell
python scripts/update_algo_results.py              # rewrite tables from W&B
python scripts/update_algo_results.py --dry-run    # preview without writing
```

Requires `wandb login` (or `WANDB_API_KEY`). By default the script reads finished
runs tagged `template` from `LatentLab/torchrl-hydra-template`. Use
`--entity`, `--project`, `--tag`, or `--algo dqn` to override scope.

### Adding a new algorithm

1. Create `src/algorithms/my_algo/my_algo.py` with an `__init__.py` re-export and
   `README.md` (theory, pseudocode, W&B results). Follow the kwargs pattern
   described in [Architecture → Algorithm hyperparameters](docs/architecture.md#algorithm-hyperparameters).
   Use `Callable` factories for design choices (inline lambdas, `functools.partial`,
   or small helpers).
2. Implement `setup(make_env)`, `step(batch)`, `get_policy()`,
   `get_explore_policy()`, `get_collector_config()`,
   `_get_training_state()`, `_load_training_state()`.
3. Add `configs/algorithm/my_algo.yaml` mirroring scalar defaults from `__init__`.
   Keep it free of task specifics — no pixel networks, no per-benchmark budgets.
   If a network has more than one variant, give it a config group under
   `configs/algorithm/network/` or `configs/algorithm/policy/`.
4. Add `configs/experiment/my_algo/<benchmark>.yaml` composing algorithm +
   environment + trainer, and put the task-dependent overrides there. Name it
   after the benchmark (`gym`, `dmc`, `ale`, `atari100k`), not the task — the
   task is an override.
5. Add a smoke test in `tests/test_smoke.py`.
6. Update `README.md` and `AGENTS.md`.

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

python src/train.py experiment=dqn/gym
```

A full training run (500k frames, ~7 minutes on CPU) reproduces the torchrl SOTA
reference for DQN-CartPole.

For Atari Pong (mirrors the torchrl SOTA `dqn_atari.py` reference, 40M frames on
GPU):

```shell
python src/train.py experiment=dqn/ale
```

## Architecture

```
train.py  ->  Trainer(algorithm, environment)
                ├── owns: device, env lifecycle, Collector, eval, callbacks, checkpoints
                ├── logs: every metric row against global_step (agent steps)
                └── calls: algorithm.step(batch) -> metrics

Algorithm    ->  owns: network, replay buffer, loss, optimiser, exploration,
                       collector config (frames_per_batch, init_random_frames, ...)
Environment  ->  factory: env name + transforms list -> make_env(num_envs, device)
Evaluation   ->  the measurement protocol: eval env stack, cadence, episode count
```

Full component API, environment/evaluation configs, trainer internals, and the
Hydra config layout are in [docs/architecture.md](docs/architecture.md).

## Evaluation

Evaluation is its own config group, decoupled from the algorithm: eval env
stack, cadence, episode count, policy mode, and which stream is canonical are
all protocol, never learning knobs. Runs log in a layout
[openrlbenchmark](https://github.com/openrlbenchmark/openrlbenchmark) can
consume directly, so algorithms are comparable on one axis. See
[docs/evaluation.md](docs/evaluation.md) for the full protocol, the logging
contract, and benchmark results.

![DMC cheetah-run return](docs/figures/dmc.png)

## Documentation

| Doc | Covers |
|-----|--------|
| [docs/architecture.md](docs/architecture.md) | `BaseAlgorithm` / `Trainer` API, environment configs, Hydra config layout, logging, callbacks |
| [docs/evaluation.md](docs/evaluation.md) | Evaluation protocol, openrlbenchmark logging contract, multi-GPU sweeps, figures, benchmark results |
| [docs/contributing.md](docs/contributing.md) | Syncing with upstream, feeding changes back to the template |
| [docs/acknowledgements.md](docs/acknowledgements.md) | Prior art and attribution |

## Smoke test

```shell
pytest tests/test_smoke.py -v
```

Loads the experiment config, applies minimal-frame overrides, and asserts that
one full training cycle runs without error.
