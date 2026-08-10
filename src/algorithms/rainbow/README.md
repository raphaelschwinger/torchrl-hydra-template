# Rainbow / Data-Efficient Rainbow (DER)

Rainbow ([Hessel et al. 2018](https://arxiv.org/abs/1710.02298)) combines six
independent extensions of DQN ([Mnih et al. 2015](https://www.nature.com/articles/nature14236))
into one agent: double Q-learning
([van Hasselt et al. 2016](https://arxiv.org/abs/1509.06461)), prioritized
experience replay ([Schaul et al. 2016](https://arxiv.org/abs/1511.05952)),
dueling networks ([Wang et al. 2016](https://arxiv.org/abs/1511.06581)),
multi-step returns, distributional RL / C51
([Bellemare et al. 2017](https://arxiv.org/abs/1707.06887)), and noisy
networks ([Fortunato et al. 2018](https://arxiv.org/abs/1706.10295)).
**Data-Efficient Rainbow** ([van Hasselt et al. 2019, "When to use parametric
models in reinforcement learning?"](https://arxiv.org/abs/1906.05243)) is the
same agent with hyperparameters tuned for the Atari-100k benchmark: longer
n-step returns, more frequent target updates, and a smaller encoder.

`RainbowAlgorithm` extends `DQNAlgorithm` (`src/algorithms/dqn/dqn.py`) and
overrides only what the six extensions require — `setup()` builds a
dueling/noisy/distributional network and a prioritized/multi-step replay
buffer instead of DQN's plain ones, `step()` adds priority updates and
noisy-noise resampling, `get_policy()` controls whether noise stays sampled at
eval time. Every extension is an independent toggle
(`dueling`/`noisy`/`double_dqn`/`distributional`/`prioritized`), so ablating
one is a config override, not a code change.

Both presets share one class; they differ only in their algorithm config:

| | `algorithm=rainbow` (standard) | `experiment=rainbow/atari100k` (DER preset) |
|---|---|---|
| Multi-step $n$ | 3 | 10 |
| Target update period | 8000 grad steps | 8000 grad steps |
| Encoder | dqn (NatureDQN-shaped CNN, SAME padding) | dqn (same) |
| Hidden dim | 512 | 512 |
| Noisy-net $\sigma_0$ | 0.5 | 0.5 |
| Adam $\epsilon$ | 1e-8 | 1.5e-4 |
| Exploration `annealing_frames` | 250,000 | 2,000 |
| PER IS exponent $\beta$ | annealed 0.4 → 1.0 over 100k steps | annealed 0.4 → 1.0 over 100k steps |
| Replay capacity | 1,000,000 | 200,000 |
| Init random frames | 20,000 | 1,600 |

The `der` preset matches Google's canonical DER *schedule* hyperparameters,
as pinned in
[`bigger_better_faster/bbf/configs/der.gin`](https://github.com/google-research/google-research/blob/master/bigger_better_faster/bbf/configs/der.gin)
(the BBF paper's DER baseline — itself van Hasselt et al. 2019 tuned to match
the SPR paper's n-step of 10): `n_steps`, `hard_update_freq`,
`replay_capacity`, `adam_eps`. Encoder, hidden dim and noisy-net $\sigma_0$
are pinned to the *standard* Rainbow values rather than Kaixhin/Rainbow's
smaller data-efficient network — comparing two good and one bad W&B run found
the smaller network/noise combination was not what separated good runs from
bad, so there was no evidence to justify the smaller network (see
"Documented deviations" below). `annealing_frames` is DER-specific and much
shorter than the standard preset's: since epsilon-greedy stacks on top of
noisy nets (`noisy=true` no longer skips it), the standard preset's
250k-frame schedule would leave a 100k-frame DER run acting close to
uniformly randomly for most of training.

## Update rule

Each transition sampled from prioritized replay contributes the C51
cross-entropy loss

$$
\mathcal{L} = - \sum_z \; m(z) \, \log p_\theta(z \mid s_t, a_t),
$$

where $m$ is the n-step Bellman target distribution projected onto the fixed
support $[V_{\min}, V_{\max}]$ (computed by TorchRL's `DistributionalDQNLoss`,
following [Kaixhin/Rainbow's `agent.py`](https://github.com/Kaixhin/Rainbow/blob/master/agent.py)),
and the next-state action is always selected by the online network and
evaluated by the target network (double DQN, built into
`DistributionalDQNLoss`). Sample priorities are the per-sample cross-entropy
loss; the importance-sampling exponent $\beta$ anneals 0.4 → 1.0 and gradients
are clipped to norm 10.

## Pseudocode → implementation

```
for each collector batch:                         # step() (overrides DQNAlgorithm.step)
    extend replay buffer                           #   TensorDictPrioritizedReplayBuffer.extend
                                                     #   (MultiStepTransform folds in n-step returns)
    if warmed up:
        for num_updates:
            if noisy: resample noisy-layer noise    #   q_actor.apply(reset_noise)
            sample prioritized batch                #   replay_buffer.sample(batch_size)
            C51 cross-entropy loss (n-step, double)  #   DistributionalDQNLoss
            clip grad norm 10, Adam step
            hard target update every N grad steps    #   HardUpdate
            update priorities from td_error           #   replay_buffer.update_tensordict_priority
            anneal PER beta                            #   replay_buffer.sampler.beta
```

- `setup()` builds the dueling (`DuelingCnnDQNet`) + noisy (`NoisyLinear`)
  network, wraps it in `DistributionalQValueActor` (C51) or plain
  `QValueActor`, and builds `TensorDictPrioritizedReplayBuffer` with a
  `MultiStepTransform`.
- `step()` mirrors `DQNAlgorithm.step()`'s shape (extend → warm-up gate →
  `num_updates` gradient steps), adding noisy-noise resampling and priority
  updates.
- `get_policy()` is overridden to keep noisy-net noise sampled at evaluation
  time by default (`eval_noise=True`), matching the paper.
- Everything else (`get_collector_config`, `get_explore_policy`,
  `_get_training_state`, `_load_training_state`) is inherited unchanged from
  `DQNAlgorithm` — the attribute names (`q_actor`, `optimizer`,
  `_explore_policy`, `_collected_frames`) match.

## Run

```shell
# Data-Efficient Rainbow on Atari-100k (JamesBond by default)
python src/train.py experiment=rainbow/atari100k
python src/train.py experiment=rainbow/atari100k environment.task=Breakout
```

`configs/algorithm/rainbow.yaml` holds the standard (Dopamine-style) Rainbow
hyperparameters. The **Data-Efficient Rainbow** preset — n-step 10, target
update every 8000 gradient steps, Adam eps 1.5e-4, a 200k replay, and a
2,000-frame epsilon-greedy anneal — is a property of the Atari-100k budget,
so it lives in `configs/experiment/rainbow/atari100k.yaml` rather than in a
second algorithm config (per `AGENTS.md`: no `der.yaml`). To run standard
Rainbow at a longer budget, drop those overrides.

## Environment

`configs/environment/atari100k{,_eval}.yaml` are game-generic: they interpolate
`ALE/${environment.task}-v5`, so `environment.task=Breakout` switches both the
train and the eval env at once. Max-and-skip and episodic-life (train only) are
applied via `gymnasium_wrappers` so life loss is checked after each aggregated
agent step; image preprocessing uses TorchRL transforms. Frame stacking uses
`CatFrames` (same pattern as `ale.yaml`). Training uses life-loss terminals
and unclipped rewards (no `SignTransform` — see "Documented deviations"
below); evaluation uses true game-over episodes and unclipped rewards too.

## Documented deviations

- **No `_partial_` factories** (TD-MPC2 precedent, see `AGENTS.md`): the
  network and replay buffer are built from scalar kwargs inside `setup()`
  rather than injected as Hydra `_partial_` factories. Rainbow's architecture
  (dueling/noisy/distributional shape) and replay buffer (prioritized +
  multi-step) are intrinsically coupled to the toggle kwargs above — swapping
  in an unrelated factory wouldn't compose with them — so a swappable
  `Callable` interface would add failure modes without research payoff.
- **`double_dqn` only applies when `distributional=False`**: TorchRL's
  `DistributionalDQNLoss` always selects the next action with the online
  network and evaluates it with the target network internally (ported
  directly from Kaixhin/Rainbow), so it is unconditionally "double" regardless
  of the toggle.
- **`MultiStepTransform`'s `steps_to_next_obs` key** comes back shaped `[B]`
  instead of `[B, 1]`; `step()` unsqueezes it before calling the loss module
  to avoid it mis-broadcasting against the `[B, 1]`-shaped reward/terminated
  tensors.
- **DER's encoder does not match `der.gin`**: Google's canonical `der.gin`
  points `BBFAgent.network` at `spr_networks.RainbowDQNNetwork` (BBF's
  architecture), not a plain DER-specific net. That network belongs to BBF
  (`src/algorithms/bbf/`), so `experiment=rainbow/atari100k` keeps Rainbow's
  own `dqn` encoder/hidden-dim/noisy-σ instead — matching the standard
  `algorithm=rainbow` preset, not Kaixhin/Rainbow's smaller data-efficient
  CNN either. This preset previously used the smaller Kaixhin-style encoder
  (`encoder_type: data_efficient`, hidden dim 256, $\sigma_0$ 0.1); comparing
  W&B runs before/after that switch found the smaller network was not what
  distinguished good runs from bad (both a good and a bad run shared
  `encoder_type: dqn`), so there was no evidence to keep the smaller,
  untested combination. All schedule hyperparameters (n-step, target update
  period, replay capacity, Adam eps) still match `der.gin` exactly.
- **`SignTransform` (reward clipping to $\{-1, 0, 1\}$) was removed from
  training**: the original preset clipped rewards, matching standard
  DQN/Rainbow practice and C51's fixed `v_min=-10, v_max=10` support.
  Comparing two good runs and one bad run found the best-performing run had
  unclipped rewards, same as the bad run — clipping wasn't what separated
  them — so it was dropped for this preset. If you see C51 value
  misestimation on a game with larger raw score deltas than Jamesbond,
  reconsider re-adding `SignTransform` to
  `configs/environment/atari100k.yaml`.

## Experimental results

**Live W&B table (canonical):** [LatentLab/torchrl-hydra-template — Table](https://wandb.ai/LatentLab/torchrl-hydra-template/table)

| Run | Environment | Config | Seed | Frames | Eval return | Notes |
|-----|-------------|--------|------|--------|-------------|-------|
| — | — | — | — | — | — | No finished runs tagged ``template`` yet — see [W&B table](https://wandb.ai/LatentLab/torchrl-hydra-template/table) |
