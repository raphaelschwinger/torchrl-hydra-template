# Claude Code instructions for torchrl-hydra-template

See `AGENTS.md` for the full codebase guide. This file adds Claude-specific notes.

## Maintenance rule

**Always update `README.md` and `AGENTS.md`** when changing a public API, adding an algorithm, renaming a class, or changing a convention. README targets human readers; AGENTS.md targets AI agents.

## Architecture in one paragraph

This template is a thin Hydra wrapper around TorchRL's built-in trainers. `src/train.py` does `trainer = hydra.utils.instantiate(cfg.trainer); trainer.train()` — that's the whole entry point. Everything else (collector, loss, replay buffer, networks, optimizer, env transforms) is composed via Hydra's defaults list using the structured configs that TorchRL ships in `torchrl.trainers.algorithms.configs`. Adding a new algorithm that TorchRL ships natively (DQN/PPO/SAC/TD3/DDPG/CQL/IQL) is a YAML-only change. Adding a custom algorithm = write a factory + register a `TrainerConfig` dataclass with Hydra's `ConfigStore` (see `src/algorithms/reinforce/`).

## Key patterns (quick reference)

### Adding a TorchRL-shipped algorithm

Create `configs/algorithm/<name>.yaml` modeled on `configs/algorithm/dqn.yaml`. Pull the trainer/loss/network/optimizer schemas from ConfigStore via `defaults: - /trainer@trainer: <name>` etc. Wire env-specific dims via `${env.obs_dim}` / `${env.action_dim}` interpolation.

### Adding a custom algorithm

`src/algorithms/<name>/trainer.py` provides a `make_<name>_trainer(...)` factory that returns a configured `torchrl.trainers.Trainer`. `src/algorithms/<name>/configs.py` defines a `<NAME>TrainerConfig` dataclass and registers it: `cs.store(group="trainer", name="<name>", node=<NAME>TrainerConfig)`. Import the configs module from `src/train.py` so the registration runs. See `src/algorithms/reinforce/` for the canonical example.

### Adding a network architecture

Custom Python modules go under `src/networks/`. Reference them in YAML via `_target_: src.networks.<file>.<Class>`. To swap a default schema-typed network for a custom one in an experiment, drop the schema first: `defaults: - override /network@networks.qvalue_network: null` then redefine the entry. See `configs/experiment/dqn/atari_pong.yaml`.

## What not to do

- Do not write a custom `Algorithm` / `BaseTrainer` / `Callback` framework — TorchRL's `Trainer` is the orchestrator and its hook system replaces our old callback protocol.
- Do not name a search-path config group identically to a ConfigStore group name registered by TorchRL (e.g. our group named `network` collides with `/network` from ConfigStore and causes Hydra to recurse). Pick a distinct name (the experiment-level Atari override uses `qvalue_network` keys directly inside the experiment file rather than a separate `network/` group).
- Do not use `# @package _global_` on the top-level `configs/train.yaml` — only on group files (`configs/algorithm/*`, `configs/environment/*`, `configs/experiment/**`) that need to set keys at the root.
- Do not put env-specific values in `configs/algorithm/*` — algorithms reference `${env.*}` interpolation and the env config exposes those keys (`env.obs_dim`, `env.action_dim`, and for continuous-action algos `env.policy_out_dim`).
