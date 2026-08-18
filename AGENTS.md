# Agent instructions for torchrl-hydra-template

## Project overview

A modular reinforcement learning research template built on
[TorchRL](https://github.com/pytorch/rl) and
[Hydra](https://github.com/facebookresearch/hydra). Four composable components —
**Environment**, **Algorithm**, **Trainer**, **Evaluation** — are wired together by
`src/train.py` (and `src/eval.py`) via `src/utils/instantiate.py::build_trainer`.

Implemented experiments:

| Algorithm | Environment    | Experiment config             |
|-----------|----------------|-------------------------------|
| DQN       | CartPole-v1    | `experiment=dqn/gym`          |
| DQN       | ALE/Pong-v5    | `experiment=dqn/ale`          |
| DDPG      | HalfCheetah-v4 | `experiment=ddpg/gym`         |
| A2C       | HalfCheetah-v4 | `experiment=a2c/gym`          |
| PPO       | DMC cheetah-run | `experiment=ppo/dmc`         |
| PPO       | ALE/Jamesbond-v5 (Atari-100k) | `experiment=ppo/ale` |
| TD-MPC2   | dmc cheetah-run | `experiment=tdmpc2/dmc`      |
| DreamerV3 | ALE/Jamesbond-v5<br>(Atari100k) | `experiment=dreamer/atari100k` |
| DreamerV3 | DMC cheetah-run<br>(proprio) | `experiment=dreamer/dmc` |
| DER (Rainbow) | ALE/Jamesbond-v5<br>(Atari100k) | `experiment=rainbow/atari100k` |
| BBF       | ALE/Jamesbond-v5<br>(Atari100k) | `experiment=bbf/atari100k` |

Other algorithms will follow.

Cross-algorithm results for the two comparison groups (Atari-100k Jamesbond and
DMC cheetah-run, 3 seeds each) live in **docs/evaluation.md → Benchmark
results**, with the figures in `docs/figures/`. Regenerate them from W&B with
`./scripts/make_figures.sh --tag <sweep tag> --full --publish`; never hand-edit
the tables there, they are `rlops` output. `--full` raises rliable's
Stratified Bootstrap CI rep counts from openrlbenchmark's 10-rep quick-test
default to its own recommended values — omit it only for a fast layout
preview, never for a figure that ships.

**Experiments are named after the benchmark, not the task.** The table shows
each experiment's *default* task; every environment config exposes a single
`task` key, so any other task is one override:

```shell
python src/train.py experiment=dqn/ale environment.task=Breakout
python src/train.py experiment=tdmpc2/dmc environment.task=walker-walk
```

Eval env configs interpolate `name: ALE/${environment.task}-v5` — an *absolute*
reference, so under the `eval_environment` package they resolve against the
train env. One `environment.task=` moves both. Every env config also carries
`env_id` (the benchmark id logged to W&B, **without** the `ALE/` prefix) and
`action_repeat` (env frames per agent step), which are reporting metadata only.
Max-and-skip and episodic-life use `gymnasium_wrappers` (SB3-style gym wrappers
in [`src/environments/atari_wrappers.py`](src/environments/atari_wrappers.py))
so life loss is evaluated after each aggregated agent step; the rest of the
stack is TorchRL transforms (`NoopResetEnv`, `GrayScale`, `Resize`, `CatFrames`,
…). A TorchRL `MaxAndSkipTransform` is also available when episodic-life is not
required.
Rainbow with standard (Dopamine-style) hyperparameters is available as
`algorithm=rainbow`; the official data-efficient preset (DER) is applied by
`experiment=rainbow/atari100k`, since it is a property of the 100k budget.
BBF (`algorithm=bbf`) builds on the same Atari-100k stack but subclasses
`BaseAlgorithm` directly: it hand-writes the C51 target and reuses a torchrl
`PrioritizedSliceSampler` for contiguous-window sampling (n-step computed at
sample time), adding SPR self-prediction, an Impala-CNN ×4 encoder, periodic
shrink-and-perturb resets, annealed n-step/discount, DrQ augmentation and an EMA
target. It requires `trainer.num_envs=1` (a single contiguous stream).

## Design principles

1. **Readable algorithm code.** Each algorithm file should read close to the
   pseudocode from the paper. `step()` is short and corresponds to the update
   equations. Long config-shuffling and framework glue belong elsewhere.
2. **Hard separation of responsibilities.**
   - **Algorithm** owns everything that affects the learning curve: network, replay
     buffer, loss, optimiser, exploration, target-net schedule, and the collector
     config (`frames_per_batch`, `init_random_frames`, ...). All hyperparameters live
     as keyword arguments on `__init__`.
   - **Trainer** owns the loop. It creates the collector from
     `algorithm.get_collector_config()`, calls `algorithm.step(batch)`, manages the
     device, fires callbacks, and checkpoints.  Nothing on the trainer config affects
     reward or sample efficiency.
   - **Environment** is a fixed task definition: env name + transform list, plus
     `env_id` / `action_repeat` reporting metadata. It does not know about the
     algorithm.
   - **Evaluation** owns the measurement protocol: eval env stack, cadence,
     episode count, policy mode, and which stream is canonical. Like the trainer,
     nothing on it may affect reward or sample efficiency.
3. **One source of truth per concern.** HP defaults live in the algorithm's
   `__init__` (with type hints + docstrings). YAML mirrors them for overrides.
4. **Callable factories via Hydra.** Design choices that are `Callable`s (replay
   buffer, network) are configured in `configs/algorithm/*.yaml` with `_partial_`
   and nested `_target_` nodes. **`src/train.py` and `src/eval.py` build the
   algorithm with `hydra.utils.instantiate(cfg.algorithm, device=None)`** so those
   nested configs become real callables. Plain `OmegaConf.to_container` + `**kwargs`
   would pass dicts instead of partials.

   *Exception (TD-MPC2 precedent):* when subnetworks are architecturally coupled
   (shared latent dims, checkpoint-compatibility pins parameter names), factories
   add failure modes without research payoff. Then architecture knobs are plain
   scalar kwargs and the model/buffer are built inside `setup()` — document the
   deviation in the algorithm README.
5. **Reusable building blocks live in `src/components/`.** Anything not specific
   to one algorithm (e.g. two-hot discrete-regression math, SimNorm/NormedLinear/
   vmapped `Ensemble` layers, `RunningScale`) goes there so other algorithms can
   import it. Code adapted from external repos carries a source-attribution
   header comment (upstream URL + file path + license).

## Algorithm constructor pattern

```python
class DQNAlgorithm(BaseAlgorithm):
    def __init__(
        self,
        device: torch.device | None = None,
        *,
        # Design choices: factories injected as Callables
        replay_buffer: Callable[[], ReplayBuffer] = lambda: TensorDictReplayBuffer(...),
        # Q-net factory; setup() passes (obs_shape, num_actions) — see below.
        network: Callable[[tuple[int, ...], int], nn.Module] = functools.partial(
            make_mlp_q_net, num_cells=[120, 84], activation_class=nn.ReLU
        ),
        # Observation tensordict key (e.g. "observation" for vector obs, "pixels" for image obs).
        obs_key: str = "observation",
        # Scalar HPs
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        batch_size: int = 128,
        max_grad_norm: float = 10.0,
        eps_start: float = 1.0,
        eps_end: float = 0.05,
        annealing_frames: int = 250_000,
        frames_per_batch: int = 1_000,
        init_random_frames: int = 10_000,
        max_frames_per_traj: int = -1,
        num_updates: int = 100,
        hard_update_freq: int = 50,
    ) -> None:
        super().__init__(device)
        # ... store kwargs verbatim ...
```

Rules:
- `*` makes every HP keyword-only.
- `BaseAlgorithm.__init__(device)` — **no `cfg` parameter**. Algorithms read env
  specs from `make_env()` inside `setup()`.
- `replay_buffer` is a **no-arg** factory returning a `ReplayBuffer`.
- `network` (DQN) is a factory called as **`network(obs_shape, num_actions)`** —
  positional `obs_shape` is the raw observation shape tuple (e.g. `(4,)` for
  CartPole, `(4, 84, 84)` for stacked Atari frames) and `num_actions` is the
  discrete action count. For DDPG, `actor_network` and `value_network` use the
  same call signature with the continuous action vector size. Use the helpers
  in `src/components/networks.py`:
    - `make_mlp_q_net(obs_shape, num_actions, *, num_cells, activation_class)` —
      flattens `obs_shape` into a torchrl `MLP`. Default for vector observations.
    - `NatureDQN(obs_shape, num_actions, *, ...)` — Mnih et al. 2015 ConvNet+MLP
      head. Default for image observations.
    - `make_mlp_ddpg_actor(obs_shape, action_dim, *, num_cells, activation_class)` —
      MLP body for a deterministic actor (DDPG); no final tanh, the algorithm
      wraps it in `TanhModule` to rescale to the action spec.
    - `make_mlp_ddpg_critic(obs_shape, action_dim, *, num_cells, activation_class)` —
      state-action critic; takes `[obs, action]` concatenated by `ValueOperator`.
    - `make_mlp_a2c_actor(obs_shape, action_dim, *, num_cells, activation_class)` —
      MLP body for an A2C stochastic actor; outputs `2 * action_dim` features
      that `NormalParamExtractor` splits into `loc` and `scale` for TanhNormal.
    - `make_mlp_a2c_value(obs_shape, action_dim, *, num_cells, activation_class)` —
      state-value (V(s)) critic; `action_dim` is unused but kept for signature
      parity with the actor factory.
  All keep everything after the two positional args **kwarg-only**, so a Hydra
  `_partial_` config can pre-bind kwargs without colliding with `setup()`'s call.
- **Shared, algorithm-agnostic building blocks live in `src/components/`.**
  `src/components/networks.py` holds actor-critic factories with cleanRL-style
  orthogonal initialisation (same `(obs_shape, action_dim)` + kwarg-only
  convention):
    - `orthogonal_init_(module, *, hidden_gain, final_gain, bias_const)` —
      orthogonal weights (√2 hidden; 0.01 policy head / 1.0 value head), zero biases.
    - `make_normal_mlp_actor(...)` — MLP mean + `AddStateIndependentNormalScale`
      (state-independent learned log-std); outputs `(loc, scale)`.
    - `make_mlp_value(...)` — MLP V(s) critic.
    - `make_nature_cnn_trunk(...)` — Nature-DQN ConvNet+Linear trunk shared by
      actor/critic heads (pass as PPO's `common_network`).
    - `make_categorical_head(...)` / `make_value_head(...)` — logits / V(s)
      heads on trunk features.
- `obs_key` selects which tensordict key the observation comes from. Vector
  envs (CartPole) use `"observation"`; pixel envs (Atari with `from_pixels=True`)
  use `"pixels"`. The key is forwarded to `QValueActor.in_keys` and used to read
  the spec for the network factory.
- **Activation class in YAML:** `torchrl.modules.MLP` expects `activation_class`
  to be a **type** (it instantiates internally). In Hydra YAML, **`_target_:
  torch.nn.ReLU` nests an instantiation** and produces a module instance, which
  breaks `MLP`. Use **`hydra.utils.get_class`** instead:

  ```yaml
  activation_class:
    _target_: hydra.utils.get_class
    path: torch.nn.ReLU
  ```

- Scalar HPs are plain kwargs and **do** appear in YAML.

## `step(batch)` shape

```python
def step(self, batch: TensorDict) -> dict[str, float]:
    # 1. Always — anneal exploration, store transitions
    batch = batch.reshape(-1)
    self.greedy_module.step(batch.numel())
    self.replay_buffer.extend(batch)
    self._collected_frames += batch.numel()

    # 2. Warm-up gate
    if self._collected_frames < self.init_random_frames:
        return {"train/epsilon": float(self.greedy_module.eps)}

    # 3. Optimisation loop — sample, loss, backward, optimiser, target update
    for j in range(self.num_updates):
        sample = self.replay_buffer.sample(self.batch_size).to(self.device)
        loss = self.loss_module(sample)["loss"]
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_actor.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.target_updater.step()

    return {"train/q_loss": ..., "train/epsilon": ...}
```

The trainer never touches the replay buffer, target network or epsilon — those are
algorithm internals. `StepTrainer` merges timing (`time/collect`, `time/step`,
`time/speed`) and batch metrics into the algorithm's metrics dict at logging
boundaries. Episode returns (`train/episode_reward`, `train/episode_length`)
are accumulated across collector batches between logs (so small
`frames_per_batch`, e.g. DER's 4, still reports every completed episode);
instantaneous metrics like `train/q_values` come from the current batch.
This mirrors the torchrl SOTA DQN reference and keeps batch-level bookkeeping
out of the algorithm.

### On-policy variant (A2C, PPO)

A2C and PPO drop three of those internals entirely: no long-term replay
buffer, no target networks, no warm-up. Each `step(batch)` runs `GAE` on the
rollout under `no_grad`, refills a one-shot buffer with
`SamplerWithoutReplacement`, and does mini-batch updates (`A2CLoss` /
`ClipPPOLoss`). The buffer in `a2c.py` / `ppo.py` is built directly in
`setup()` (not exposed as a `_partial_` factory) because its size is locked
to `frames_per_batch / mini_batch_size` — it's an implementation detail of
the on-policy schedule, not a research choice. `get_collector_config()`
returns `init_random_frames=0` since the stochastic actor explores from
frame zero.

PPO extends the A2C shape with: `num_epochs` passes over the rollout
(re-iterating the buffer reshuffles), the clipped ratio objective + clipped
value loss (`ClipPPOLoss`), per-minibatch advantage normalization, linear lr
annealing over `anneal_frames` (an algorithm HP — keep it equal to
`trainer.total_frames` in experiment configs), and Adam ε=1e-5. **Run GAE on
the unflattened batch (before `reshape(-1)`)** — with `num_envs > 1` the
rollout is `[num_envs, T]` and GAE needs the trailing time dim. One
`PPOAlgorithm` class covers vector and pixel inputs: pass `common_network`
(e.g. `make_nature_cnn_trunk`) to share a trunk between actor and critic
heads via `ActorValueOperator` (the `algorithm/policy=nature_cnn_categorical`
option); leave it `None` for separate MLPs (`algorithm/policy=mlp_normal`). The
actor's distribution is picked from the action spec: `Categorical` over
`logits` for discrete specs, `IndependentNormal` over `(loc, scale)` for
continuous ones.

## Instantiation in `src/train.py` / `src/eval.py`

```python
from hydra.utils import instantiate, get_class
from omegaconf import OmegaConf

algorithm = instantiate(cfg.algorithm, device=None)  # recursive; resolves _partial_ factories

env_kwargs = {k: v for k, v in OmegaConf.to_container(cfg.environment, resolve=True).items()
              if k != "_target_"}
environment = Environment(**env_kwargs)

TrainerClass = get_class(cfg.trainer._target_)
trainer = TrainerClass(cfg=cfg, algorithm=algorithm, environment=environment)
```

The **environment** is still unpacked from a flat dict. The **algorithm** must use
`instantiate` whenever its YAML contains nested `_target_` / `_partial_` nodes
(e.g. `replay_buffer`, `network`).

YAML values override Python defaults where present; absent keys fall back to
constructor defaults.

## Environment

`Environment.__init__` accepts:
- `name`: gymnasium env id (e.g. `"CartPole-v1"`, `"ALE/Pong-v5"`), or the
  dm_control domain name (e.g. `"cheetah"`) when `backend: dm_control`.
- `transforms`: list of `_target_`-keyed dicts, each instantiated as a
  `torchrl.envs.transforms` object and composed on top of the base env.
  Always include `StepCounter` explicitly. Add `RewardSum` if you want
  `train/episode_reward` in the trainer metrics — it populates the
  `("next", "episode_reward")` key the trainer reads.
- `gym_kwargs`: optional dict forwarded straight to `GymEnv` (e.g.
  `{"frame_skip": 4, "from_pixels": true, "pixels_only": false,
  "categorical_action_encoding": true}` for Atari).
- `gym_backend`: optional backend name (`"gymnasium"`); if set, the GymEnv
  construction is wrapped in `set_gym_backend(...)`.
- `backend`: `"gymnasium"` (default) or `"dm_control"`. For dm_control,
  `gym_kwargs`/`gym_backend` do not apply; the env is built via
  `torchrl.envs.DMControlEnv` and the factory sets `MUJOCO_GL=disabled`
  (headless) unless already set.
- `task`: **the single override axis of every environment config.** For
  `backend: dm_control` it is the `"<domain>-<task>"` id (e.g. `"cheetah-run"`),
  split on the first hyphen by the factory — dm_control uses underscores inside
  its own names, so this is unambiguous. For gymnasium backends `name` is
  derived from it by interpolation.

```yaml
# configs/environment/gym.yaml — gymnasium state observations
task: CartPole-v1
name: ${environment.task}
transforms:
  - _target_: torchrl.envs.transforms.DoubleToFloat
  - _target_: torchrl.envs.transforms.InitTracker
  - _target_: torchrl.envs.transforms.StepCounter
  - _target_: torchrl.envs.transforms.RewardSum
```

```yaml
# configs/environment/ale.yaml — pixel-based Atari env
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
  # ... (see configs/environment/ale.yaml for the full SOTA stack)
```

```yaml
# configs/environment/dmc.yaml — dm_control env
backend: dm_control
task: cheetah-run
transforms:
  - _target_: torchrl.envs.transforms.FrameSkipTransform  # action repeat 2
    frame_skip: 2
  - _target_: torchrl.envs.transforms.CatTensors          # flatten dict obs -> "observation"
    in_keys: [position, velocity]
    out_key: observation
  # ... (DoubleToFloat, InitTracker, StepCounter, RewardSum)
```

The factory in `src/environments/factory.py` supports **gymnasium** (`GymEnv`)
and **dm_control** (`DMControlEnv`). dm_control observations are dicts; use
`CatTensors` to flatten them into a single `observation` key (alphabetical
`in_keys` order matches the upstream TD-MPC2 concatenation). For >1 `num_envs`,
workers run on CPU (`ParallelEnv` with `mp_start_method="spawn"`).

## Evaluation

Evaluation is a config group of its own, `configs/evaluation/`, holding the
*measurement* protocol. Like the trainer, it must carry nothing that shifts the
learning curve.

```yaml
# configs/train.yaml (and eval.yaml)
defaults:
  - environment: ???
  - evaluation: gym        # protocol + eval env stack

# configs/experiment/dqn/ale.yaml
defaults:
  - override /environment: ale
  - override /evaluation: ale
```

Every file inherits from `evaluation/none.yaml`, which is the **schema of
record** — add new keys there first, then override in the benchmark files:

| key | meaning |
|---|---|
| `eval_environment` | env stack to measure on (a nested defaults entry; `null` reuses the train env config) |
| `every_n_steps` | periodic eval cadence in agent steps; `0` = final only |
| `num_episodes` | episodes per periodic eval point |
| `final_num_episodes` | episodes in the post-training stage; `0` disables |
| `policy` | `eval` -> `get_policy()`, `explore` -> `get_explore_policy()` |
| `canonical_source` | `train` or `eval`; which stream feeds `charts/episodic_return` |
| `seed` | eval env seed base; the *n*th eval point seeds with `seed + n` |
| `summary_window` | episodes averaged for `eval/final_return_mean` (default 100) |
| `summary_max_step` | ignore canonical episodes past this agent step (`null` = whole run); `src/eval.py` forces it to `null` when not training, since a standalone eval logs every episode at the checkpoint's step and the cutoff would discard all of them |

Shipped: `none`, `gym`, `ale`, `atari100k`, `atari100k_native`, `dmc`.

`atari100k_native` is `atari100k` with `eval_environment` left `null`, so the
eval env is a fresh instance of the *experiment's own* training stack. It exists
for DreamerV3, whose 64x64 RGB `image` stack cannot be served by the shared
grayscale, frame-stacked `atari100k_eval` — and does not need to be, since that
stack is already eval-clean (`terminal_on_life_loss: false`, no reward clipping).

`canonical_source` is not cosmetic. On `atari100k`, `EpisodicLifeEnv` sets
`terminated=True` at life loss, so `RewardSum` resets and a training episode is
a life-long fragment — those benchmarks must measure from eval rollouts. On
`ale`, `EndOfLifeTransform` deliberately does *not* set `done` and
`SignTransform` sits after `RewardSum`, so training episodes are already
unclipped game scores and `canonical_source: train` is correct.

`BaseTrainer.evaluate()` builds the eval env **once** and reuses it, seeds it,
selects the policy from `evaluation.policy`, and snapshots/restores every
algorithm module's `.training` flag — Rainbow's `get_policy()` toggles noisy
layers, which periodic evaluation would otherwise leak into training.

It advances the rollout with **`env.step_mdp(td)`**, never `td["next"]`.
`td["next"]` keeps only env-written keys and discards whatever the policy left
at the root — for a recurrent policy that is its entire state (DreamerPolicy's
`stoch` / `deter` / `prev_action`), so it would silently re-initialise every
step and score near zero while looking like it ran fine. `step_mdp` is
`_StepMDP(keep_other=True)`, the same transition the collector uses, so
evaluation and collection advance identically.
`tests/test_evaluation_contract.py::test_evaluation_preserves_recurrent_policy_state`
guards this.

`src/train.py` and `src/eval.py` both go through
`src/utils/instantiate.py::build_trainer`, which reads the eval env from
`cfg.evaluation.eval_environment`. They differ only in the top-level `train:` /
`eval:` stage flags they pass to `trainer.run()`.

## Metrics and openrlbenchmark

Everything is logged against **`global_step` in agent steps** (what
`batch.numel()` counts), with `frames = global_step * action_repeat` alongside.
No algorithm may define its own axis.

| key | rows |
|---|---|
| `charts/episodic_return` / `_length` | one per episode, from `canonical_source` |
| `charts/train_episodic_return` / `_length` | one per completed training episode |
| `charts/eval_episodic_return` / `_length` | one per eval episode |
| `eval/return_{mean,std,min,max}`, `eval/episodes` | one per eval point |
| `eval/final_return_mean` / `_std` | run summary (logger summary, not history) |
| `train/*`, `time/*` | log boundaries |

Emit metrics only through `BaseTrainer.log_metrics(metrics, step)` and
`log_episodes(returns, lengths, step, source)` — they inject `global_step` and
`frames`. The callback protocol separates `on_metrics` (a row of metrics; fires
per episode, loggers want it) from `on_step_end` (the loop crossed a boundary;
progress bar and checkpointer want it).

**One `train/` family per algorithm.** Everything an algorithm returns from
`step()` (or from the optional `pop_train_metrics()` window-mean hook, which
`StepTrainer` *merges* into the same row) is prefixed `train/`. Dreamer's model
reports `loss/…` and `opt/…` internally; those are flattened to
`train/loss_…` / `train/opt_…` on the way out. No algorithm keeps its own
episode-return statistic either — `train/episode_reward` and the per-episode
`charts/*` rows come from the trainer, from the same tensordict keys, and a
private rolling mean next to them is a second definition of the same number.
The one accepted exception is Dreamer's `video/*`, which keeps its own
`video/frame` axis (see below).

Compatibility rules, enforced by `tests/test_evaluation_contract.py`:

- `env_id`, `exp_name`, `seed` stay **top-level** in `configs/train.yaml`;
  openrlbenchmark filters on `config.<key>` and nesting forces brittle
  `ceik=environment.value.task` selectors.
- `env_id` carries **no `ALE/` prefix** — the HNS table is keyed `Pong-v5`.
- `(exp_name, env_id)` must be unique across experiments, or two variants merge
  into one curve. `bbf/atari100k_rr8` sets `exp_name: bbf_rr8`;
  `rainbow/atari100k` sets `exp_name: der`.
- `WandBLogger` calls `wandb.log` **without** `step=`, plus
  `define_metric("*", step_metric="global_step")`. `rlops` joins with
  `history(keys=[xaxis, "_runtime", metric]).dropna()`, so a metric logged
  without `global_step` in the same row silently contributes nothing.
- Episodes are logged one row each, never pre-aggregated: openrlbenchmark's
  tables average the last 100 *logged points*.

## Trainer

`StepTrainer` is the only trainer.  It:
- creates a `torchrl.collectors.Collector` from `algorithm.get_collector_config()`
  and `cfg.trainer.total_frames`;
- iterates the collector, calls `algorithm.step(batch)`, and fires
  `ON_STEP_END` callbacks at logging boundaries;
- delegates device resolution to `src/utils/device.py`.

`BaseTrainer` owns env lifecycle, metric emission, the evaluation protocol and
checkpoint orchestration. `run(train, evaluate)` is the full lifecycle:
`ON_TRAIN_START` -> optional `_training_loop()` -> optional
`run_final_evaluation()` -> `ON_TRAIN_END` in a `finally`. By default
(`configs/train.yaml`) a final `checkpoints/last.pt` is written at train end
(`checkpoint.enabled: true`, `save_last: true`); periodic saves are off
(`save_every_n_steps: 0`) — set a positive value to enable them. The
post-training evaluation stage runs `evaluation.final_num_episodes` episodes
(the BBF and DER experiments set `100` for the official Atari-100k protocol) and
is skipped with `eval=false`. `checkpoint.resume_from` works regardless of
`checkpoint.enabled`.

## File map

```
src/
  train.py                  — entry point; instantiate(cfg.algorithm); environment **kwargs
  eval.py                   — evaluation entry point; same algorithm instantiation
  networks.py               — network factories: make_mlp_q_net, NatureDQN,
                              make_mlp_ddpg_actor, make_mlp_ddpg_critic,
                              make_mlp_a2c_actor, make_mlp_a2c_value
  components/
    networks.py             — shared actor-critic factories with orthogonal init:
                              orthogonal_init_, make_normal_mlp_actor, make_mlp_value,
                              make_nature_cnn_trunk, make_categorical_head, make_value_head
  algorithms/
    base.py                 — BaseAlgorithm ABC; TrainingState and CollectorConfig dataclasses
    dqn/
      dqn.py                — DQNAlgorithm; replay/network factories (defaults + setup contract)
      README.md             — theory, pseudocode, W&B benchmark table
    ddpg/
      ddpg.py               — DDPGAlgorithm; actor/critic/replay/noise factories
      README.md             — theory, pseudocode, W&B benchmark table
    a2c/
      a2c.py                — A2CAlgorithm; on-policy actor/critic with GAE + A2CLoss
      README.md             — theory, pseudocode, W&B benchmark table
    ppo/
      ppo.py                — PPOAlgorithm; on-policy clipped-ratio updates with GAE + ClipPPOLoss
      README.md             — theory, pseudocode, trick mapping, W&B benchmark table
    tdmpc2/
      tdmpc2.py             — TDMPC2Algorithm; world-model learning + slice replay buffer
      world_model.py        — WorldModel (upstream-checkpoint compatible) + api_model_conversion
      planner.py            — MPPIPlanner (latent-space planning, warm-started)
      policy.py             — TensorDictModule wrapper (reads obs + is_init)
      README.md             — theory, pseudocode, W&B benchmark table
    rainbow/
      rainbow.py            — RainbowAlgorithm(DQNAlgorithm): dueling/noisy/distributional network
                              + prioritized/multi-step replay, built from TorchRL's own classes
      README.md             — theory, pseudocode, Rainbow-vs-DER presets, W&B benchmark table
    bbf/
      bbf.py                — BBFAlgorithm(BaseAlgorithm): hand-written C51 + SPR + shrink-and-perturb
                              resets + annealed n-step/discount over a torchrl PrioritizedSliceSampler buffer
      networks.py           — BBFNetwork: Impala-CNN ×4 encoder, transition model, SPR projection/predictor, dueling C51 heads
      README.md             — theory, update-rule math, pseudocode→code, deviations, W&B benchmark table
  components/               — reusable building blocks (per-file attribution headers)
    math.py                 — symlog/symexp (canonical), two-hot discrete regression, squashed-Gaussian helpers (from nicklashansen/tdmpc2, MIT)
    layers.py               — SimNorm, NormedLinear, vmapped Ensemble, LayerNorm-Mish mlp (from nicklashansen/tdmpc2, MIT)
    scale.py                — RunningScale (trimmed-percentile value normalizer) (from nicklashansen/tdmpc2, MIT)
    distributions.py        — Dreamer distribution factories: OneHotDist, TwoHot, SymlogDist, ... (from NM512/r2dreamer)
    ema.py                  — polyak_update (in-place EMA of parameters)
    optim/                  — LaProp optimizer (Z-T-WANG/LaProp-Optimizer, MIT), adaptive gradient clipping
  environments/
    environment.py          — Environment wrapper (holds factory kwargs, exposes make_env)
    factory.py              — make_env: gymnasium/dm_control + transforms list + gym_kwargs/gym_backend
    atari_wrappers.py       — MaxAndSkip/EpisodicLife gym wrappers + MaxAndSkipTransform
  trainers/
    base.py                 — BaseTrainer ABC, TrainerEvent, Callback protocol, fire_callbacks
    step_trainer.py         — StepTrainer (Collector-driven loop)
  callbacks/                — ProgressCallback, CheckpointCallback, WandBLogger, TensorBoardLogger
  utils/                    — device resolution, seeding, callback builders
configs/
  trainer/{default,cpu,gpu,eval}.yaml — the loop (seed, total_frames, num_envs, logging, accelerator)
  algorithm/dqn.yaml        — DQN HPs; _partial_ replay_buffer + `network` group
  algorithm/ddpg.yaml       — DDPG HPs; _partial_ actor/critic/noise
  algorithm/a2c.yaml        — A2C HPs; _partial_ actor/value
  algorithm/ppo.yaml        — PPO HPs (cleanRL continuous-action defaults); `policy` group
  algorithm/tdmpc2.yaml     — TD-MPC2 HPs (model_size=5 preset; scalar knobs, no _partial_)
  algorithm/rainbow.yaml    — Rainbow HPs (standard Dopamine-style values; scalar knobs, no _partial_)
  algorithm/dreamer.yaml    — DreamerV3 (+ dreamerpro.yaml, r2dreamer.yaml variants)
  algorithm/bbf.yaml        — BBF HPs (official BBF.gin RR2 preset; scalar knobs, no _partial_)
  algorithm/network/{mlp_q,nature_dqn}.yaml — swappable DQN Q-network (state / pixels)
  algorithm/policy/{mlp_normal,nature_cnn_categorical}.yaml — swappable PPO actor+critic(+trunk)
  algorithm/dreamer/{12m..400m}.yaml — Dreamer model-size presets
  environment/gym.yaml      — gymnasium state obs (classic control + MuJoCo)
  environment/dmc.yaml      — dm_control, task: <domain>-<task> (FrameSkip 2 + CatTensors)
  environment/ale.yaml      — Atari, standard protocol (EndOfLife + Sign + VecNorm)
  environment/ale_eval.yaml — same without those three (true game scores)
  environment/atari100k.yaml      — Atari-100k protocol (gymnasium_wrappers: max-and-skip + episodic-life; clipped rewards)
  environment/atari100k_eval.yaml — same without episodic-life / Sign (true game-over, unclipped)
  evaluation/none.yaml      — base schema; training stream only, no eval rollouts
  evaluation/gym.yaml       — final eval only, canonical_source: train
  evaluation/ale.yaml       — ale_eval stack, canonical_source: train
  evaluation/atari100k.yaml — atari100k_eval, every 10k + 100 final episodes, canonical_source: eval
  evaluation/atari100k_native.yaml — same protocol on the experiment's own stack (DreamerV3)
  evaluation/dmc.yaml       — periodic every 10k agent steps (TD-MPC2 upstream)
  experiment/dqn/{gym,ale}.yaml — DQN on CartPole / Atari Pong (40M frames)
  experiment/ddpg/gym.yaml      — DDPG HalfCheetah (1M frames)
  experiment/a2c/gym.yaml       — A2C HalfCheetah (1M frames)
  experiment/ppo/{dmc,ale}.yaml — PPO DMC cheetah-run (1M) / Atari-100k JamesBond (100k)
  experiment/tdmpc2/dmc.yaml    — TD-MPC2 DMC cheetah-run
  experiment/rainbow/atari100k.yaml — Data-Efficient Rainbow on Atari-100k
  experiment/dreamer/atari100k.yaml — DreamerV3 on Atari-100k (image obs)
  experiment/dreamer/dmc.yaml       — DreamerV3 on DMC cheetah-run, proprio (mlp_keys=observation)
  experiment/bbf/atari100k.yaml     — BBF on Atari-100k (RR2 default; num_envs=1)
  experiment/bbf/atari100k_rr8.yaml — BBF flagship RR8 variant (same 40k-grad-step reset cadence)
  logger/{wandb,tensorboard}.yaml
  paths/default.yaml
  train.yaml, eval.yaml
tests/
  test_smoke.py             — smoke tests: DQN (CartPole, Pong), DDPG, A2C, PPO (DMC cheetah,
                              JamesBond), TD-MPC2, DER, BBF, DreamerV3
  test_bbf_buffer.py        — BBF buffer/sampling unit tests (n-step masking, C51 projection,
                              PrioritizedSliceSampler windows); no env, no ale_py
```

## Documentation

Algorithm and model READMEs use **GitHub-flavoured markdown math**: inline formulas
with `$...$`, display formulas with `$$...$$`. Do **not** use `\(...\)` or
`\[...\]` — those delimiters are not rendered on GitHub.

Example: `$Q(s, a; \theta)$`, `$\theta_{\text{target}}$`.

## Adding a new algorithm

1. Create `src/algorithms/my_algo/my_algo.py` with an `__init__.py` re-export
   following the kwargs pattern above. Use `Callable` factories for design choices
   (inline lambdas, `functools.partial`, or small helpers). Document the **call
   signature** each factory must satisfy (e.g. `network(obs_shape, num_actions)`).
2. Implement `setup(make_env)`, `step(batch) -> dict`, `get_policy()`,
   `get_explore_policy()`, `get_collector_config()`,
   `_get_training_state()`, `_load_training_state()`.
3. Create `configs/algorithm/my_algo.yaml` with `_target_`, scalar HPs, and any
   `_partial_` / nested `_target_` blocks for factories. Use `instantiate`-
   compatible patterns (see DQN: replay buffer + partial `MLP`).
4. Create `configs/experiment/my_algo/<env>.yaml` composing your algo + env.
5. Add `src/algorithms/my_algo/README.md` with theory, pseudocode, implementation
   mapping, and an experimental-results table (link to
   [W&B project table](https://wandb.ai/LatentLab/torchrl-hydra-template/table)).
   Tag benchmark W&B runs with `template`; register the algorithm's target prefix
   in `ALGO_TARGET_PREFIXES` in `scripts/update_algo_results.py`, then refresh the
   table via `python scripts/update_algo_results.py`. Use `$...$` for inline math (see
   [Documentation](#documentation)).
6. **Update `README.md` and `AGENTS.md`.**
7. Add a smoke test in `tests/test_smoke.py`.

## What not to do

- Do not place learning-affecting knobs on `trainer:` or `environment:` configs.
- Do not create `XxxConfig` dataclasses.
- Do not add `cfg: DictConfig` to `BaseAlgorithm` or pass `cfg=cfg` to algorithms.
- Do not pass `cfg.environment` directly to `Environment()` — unpack as `**kwargs`.
- Do not add `OmegaConf` imports to `base.py`.

## Running

```shell
python src/train.py experiment=dqn/gym
python src/train.py experiment=dqn/gym algorithm.lr=1e-3
python src/train.py experiment=dqn/gym 'logger=[wandb]'  # experiments default to wandb; plain CLI defaults to tensorboard
python src/train.py experiment=dqn/ale             # Atari Pong (40M frames, GPU)
python src/train.py experiment=ddpg/gym            # DDPG continuous control (1M frames)
python src/train.py experiment=a2c/gym             # A2C on-policy continuous control (1M frames)
python src/train.py experiment=ppo/dmc             # PPO on DMC cheetah-run (1M frames)
python src/train.py experiment=ppo/ale             # PPO on Atari-100k JamesBond (100k steps, GPU)
python src/train.py experiment=tdmpc2/dmc          # TD-MPC2 model-based control (1M frames, GPU)
python src/train.py experiment=rainbow/atari100k   # DER on Atari-100k Jamesbond (100k frames, GPU)
python src/train.py experiment=dreamer/atari100k   # DreamerV3 on Atari-100k Jamesbond (GPU)
python src/train.py experiment=dreamer/dmc         # DreamerV3 on DMC cheetah-run, proprio (GPU)
python src/train.py experiment=bbf/atari100k       # BBF on Atari-100k Jamesbond (RR2, GPU)
python src/train.py experiment=bbf/atari100k_rr8   # BBF flagship RR8 (~4x compute)

# Any task within a benchmark is one override; GPU index is not committed:
python src/train.py experiment=dqn/ale environment.task=Breakout trainer.devices=[3]
python src/train.py experiment=tdmpc2/dmc environment.task=walker-walk

python scripts/update_algo_results.py              # refresh algo README benchmark tables (W&B tag: template)

# Cross-algorithm sweep: 6 experiments x 3 seeds, queue-balanced over GPUs.
# Resumable (markers in logs/benchmarks/done/); tags runs `template`.
./scripts/run_benchmarks.sh --dry-run              # print the 18 commands
./scripts/run_benchmarks.sh --smoke                # tiny budgets; validates every spec
./scripts/run_benchmarks.sh --gpus 2,3             # the real sweep

# Comparison figures + rliable, via openrlbenchmark's own rlops CLI.
# First run builds an isolated .venv-openrlbenchmark; output in logs/analysis/.
# Plotted on charts/eval_episodic_return so every algorithm in a figure shows
# the same measurement (canonical_source differs per experiment). `--publish`
# copies PNGs + tables into docs/figures/, which docs/evaluation.md embeds.
./scripts/make_figures.sh                          # every group, tag `template`
./scripts/make_figures.sh --group atari100k        # one comparison group
./scripts/make_figures.sh --tag template-v2 --full --publish   # regenerate docs/evaluation.md figures
pytest tests/test_smoke.py -v

# Evaluate an official TD-MPC2 checkpoint (see src/algorithms/tdmpc2/README.md):
python src/eval.py algorithm=tdmpc2 environment=dmc \
  checkpoint.resume_from=$PWD/checkpoints/cheetah-run-1.pt trainer.accelerator=gpu
```
