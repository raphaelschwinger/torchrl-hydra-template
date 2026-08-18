# BBF — Bigger, Better, Faster

> Schwarzer, Obando-Ceron, Courville, Bellemare, Agarwal, Castro.
> **"Bigger, Better, Faster: Human-level Atari with human-level efficiency."** ICML 2023.
> [arXiv:2305.19452](https://arxiv.org/abs/2305.19452) ·
> [official JAX code](https://github.com/google-research/google-research/tree/master/bigger_better_faster)

Model-free state of the art on the **Atari 100k** benchmark: super-human
aggregate performance (IQM human-normalized score $\approx 1.045$ at replay
ratio 8) from only 100,000 agent interactions ($\approx$ 2 hours of gameplay).

## Why is Atari 100k hard?

Standard deep RL is wildly sample-inefficient — the original DQN trained on
50M frames. With only 100k steps the obvious fix is to reuse each stored
transition more often: increase the **replay ratio** (gradient steps per
environment step). But naively increasing it *hurts* — the network overfits
early replay data and loses plasticity (D'Oro et al. 2023). The entire BBF
recipe is about *making a high replay ratio work*.

## The recipe

BBF = Rainbow-flavoured distributional Q-learning pushed to replay ratio 2–8,
kept stable by six interacting design choices:


| #   | Design choice                            | What it does                                                                                                                                                                                                       | Origin                                            |
| --- | ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------- |
| 1   | **Bigger network**                       | Impala-CNN ResNet encoder at 4× width. Scaling the encoder *only helps* once 2–6 are in place.                                                                                                                     | Espeholt et al. 2018                              |
| 2   | **SPR auxiliary loss**                   | Predict your own future latent representations $k=1..5$ steps ahead through a learned transition model (BYOL-style, no negatives). Dense self-supervised signal next to the sparse reward.                         | Schwarzer et al. 2021                             |
| 3   | **Periodic resets (shrink-and-perturb)** | Every `reset_interval` gradient steps: fully re-initialise heads; interpolate encoder + transition model 50% towards random weights — applied to the online **and** EMA target networks (official `reset_target=True`). Restores plasticity; the replay buffer carries knowledge across. | Ash & Adams 2020; D'Oro et al. 2023 (SR-SPR)      |
| 4   | **Annealed update horizon & discount**   | After each reset, n-step decays exponentially $10 \to 3$ and $\gamma$ rises $0.97 \to 0.997$ over 10k gradient steps. Fast-but-biased learning right after a reset, low-bias at convergence.                       | BBF                                               |
| 5   | **Regularisation everywhere**            | AdamW weight decay 0.1; DrQ augmentation (random $\pm 4$ px shift + intensity jitter) on *every* replayed frame; EMA target network ($\tau=0.005$, updated every gradient step).                                   | Kostrikov et al. 2021 (DrQ)                       |
| 6   | **Rainbow backbone**                     | C51 distributional RL (51 atoms on $[-10, 10]$), Double DQN action selection, dueling heads, prioritized replay (priority $=$ C51 loss). NoisyNets are *removed* (ε-greedy → greedy after 2k steps).               | Hessel et al. 2018; van Hasselt et al. 2019 (DER) |




## Update rule

The value loss is the C51 cross-entropy between the online distribution at
$(s_t, a_t)$ and the projected $n$-step target distribution:

$$
L_{\text{RL}} = -\sum_i m_i  \log p_i(s_t, a_t), \qquad
m = \Pi\left( G^{(n)} + \gamma^{n} Z_{\bar\theta}(s_{t+n}, a^\star) \right),
\quad a^\star = \arg\max_a Q_\theta(s_{t+n}, a)
$$

where $G^{(n)} = \sum_{i<n} \gamma^i r_{t+i}$ (masked at episode cuts) and $\Pi$
is the categorical projection (Bellemare et al. 2017). The SPR loss is the mean
cosine distance between predicted and target future projections:

$$
L_{\text{SPR}} = \frac{1}{K}\sum_{k=1}^{K}
\left\lVert \tilde{g}\big(\hat z_{t+k}\big) - g_{\bar\theta}\big(z_{t+k}\big) \right\rVert_2^2,
\qquad \hat z_{t+k} = h(\hat z_{t+k-1}, a_{t+k-1})
$$

with $\ell_2$-normalised projections (so the squared distance $= 2 - 2\cos$).
The optimiser takes an AdamW step on $w \cdot (L_{\text{RL}} + 5L_{\text{SPR}})$
with per-sample importance weights $w$; priorities are updated from $L_{\text{RL}}$.

## Pseudocode → implementation

```
for each env step t (100,000 total):
    a_t ~ eps-greedy(Q_target)            # eps: 1 -> 0 over 2k steps after warm-up | EGreedyModule
    store (s_t, a_t, r_t, cut_t)          # cut = life loss / terminal / reset | _store -> replay_buffer.extend

    repeat replay_ratio times:
        if grad_steps since reset == reset_interval:
            heads <- random init                  # online + target networks   | _shrink_and_perturb
            encoder, transition <- 0.5*old + 0.5*random  # shrink & perturb
            optimizer state <- fresh (Adam moments kept for encoder/transition)
        n     <- exp-anneal 10 -> 3   over 10k grad steps                        | _current_horizon
        gamma <- exp-anneal .97->.997 over 10k grad steps                        | _current_gamma
        sample prioritized window (s_t..s_t+10, a, r, cut), IS weights w         | PrioritizedSliceSampler
        augment every frame (random shift + intensity)                          | _augment
        L_RL  = C51 cross-entropy(Z_online(s_t,a_t), project(n-step target))     | _project_distribution
        L_SPR = mean_k || norm(pred(proj(ẑ_{t+k}))) - norm(proj_tgt(z_{t+k})) ||²
        AdamW step on w · (L_RL + 5 · L_SPR)                                     | _update
        priorities <- L_RL ; Q_target <- (1-τ)Q_target + τQ_online              | _ema_update_target
```

- `setup()` builds the online + EMA `BBFNetwork`, the `QValueActor` explore/eval
policies — which act with the EMA **target** network in train and eval alike
(official `target_action_selection = True`) — and the replay buffer;
`step()` stores the batch then runs
`round(replay_ratio · frames)` updates; `get_policy()` returns the ε=0.001 eval
policy (`FixedEpsilonGreedy`, which also fires under `ExplorationType.MODE`).
- **Subclasses** `BaseAlgorithm` **directly** (not `RainbowAlgorithm`): BBF overrides
essentially all of the value loss, buffer, target update and network, so
inheriting Rainbow would be nominal.



## Files


| File          | Contents                                                                                                                                                                         |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `bbf.py`      | `BBFAlgorithm`: the update rule above, augmentation, schedules, shrink-and-perturb, EMA target, ε-greedy policies; module-level `_masked_nstep_return` / `_project_distribution` |
| `networks.py` | `BBFNetwork`: Impala-CNN encoder, latent renormalisation, transition model, shared projection, predictor, dueling C51 heads                                                      |




## Documented deviations (from the template conventions & from Dopamine)

Verified against the official JAX release (`configs/BBF.gin`, `spr_networks.py`):

- **Hand-written C51, not** `DistributionalDQNLoss`**.** The categorical projection
and n-step target are computed directly so a *single* EMA target network feeds
both the value loss and SPR. (Rainbow in this repo reuses
`DistributionalDQNLoss`, which manages its own delayed target internally.)
- **torchrl-native buffer, not a hand-rolled sum-tree.** The prioritized
subsequence buffer is a plain `TensorDictReplayBuffer` +
`PrioritizedSliceSampler` (`slice_len = window+1`, `traj_key="traj"` constant
so windows may span episode/life boundaries). n-step returns are computed at
*sample* time — the stored data never bakes in an `n`, which is what makes the
annealed horizon (design choice 4) implementable. `replay_capacity ≥ total env steps` guarantees the ring never wraps, so a window never straddles the write
head. **Priority semantics:** torchrl reduces a slice's sampling priority over
the window (`reduction='max'`); we key each window's priority on its *start*
transition (priority $=$ C51 loss, sampled $\propto \text{loss}^\alpha$).
- **Our PER is more "real" than the official run's.** In the official release,
`set_priority` receives the *batch-mean* DQN loss (`aux_losses["DQNLoss"]` is
a scalar per batch) and `zip`s it against the full index array, so only the
first `batches_to_group` (= 2) of 64 sampled indices are ever updated; every
other transition keeps the max-priority it got on insertion. The official
runs therefore sampled *near-uniformly* despite `replay_scheme='prioritized'`.
This template implements textbook per-sample PER instead; to reproduce the
official effective behaviour, run with `algorithm.prioritized=false`.
- **Life-loss handling.** This template's `EpisodicLifeEnv` flags life loss as
`terminated` at the gym level, so `cut = terminated | done`; there is no
separate `end-of-life` key.
- **Replay-ratio convention.** `replay_ratio` = gradient steps per env step (the
paper's definition). The official gin's `replay_ratio=64` divides by batch size
(64/32 = 2), i.e. the public config is the RR2 variant; the paper's flagship is
RR8 (`configs/experiment/bbf/atari100k_rr8.yaml`).
- **Weight-decay masking:** AdamW decay is applied to weight matrices only
(`ndim > 1`), not biases.
- **The replay buffer is not checkpointed** (resume restores networks, optimizer
and counters only).



## Running

```shell
# Official-config BBF at replay ratio 2 (good default; ~1–2h on a modern GPU)
python src/train.py experiment=bbf/atari100k

# Another Atari-100k game
python src/train.py experiment=bbf/atari100k environment.task=Breakout

# Paper flagship: replay ratio 8 (A100-class GPU recommended)
python src/train.py experiment=bbf/atari100k_rr8

# Re-evaluate a saved checkpoint on the true-score eval env (game over = episode end).
# The 100-episode protocol comes from `evaluation: atari100k`; add
# `logger.0.resume=must` to append the result to the original training run.
python src/eval.py experiment=bbf/atari100k \
    checkpoint.resume_from=logs/train/runs/<run>/checkpoints/last.pt
```

**Comparing against the paper:** the official protocol is a 100-episode eval
at the *end* of training on full (game-over) episodes. Training runs this
automatically (`configs/evaluation/atari100k.yaml` sets
`final_num_episodes: 100`), logs one `charts/eval_episodic_return` row per
episode plus the `eval/return_mean` aggregate, and saves `checkpoints/last.pt`;
`src/eval.py` re-evaluates a checkpoint later. That protocol also sets
`canonical_source: eval`, so `charts/episodic_return` — the metric
openrlbenchmark reads — comes from those rollouts. It has to:
`charts/train_episodic_return` is a per-*life* return (the train env uses
`EpisodicLifeEnv`, which resets `RewardSum` at life loss), so it reads several
times lower than eval scores. Add `evaluation.every_n_steps=10_000` for a
learning curve instead of a single final number.
Official per-seed Jamesbond results from the repo's `scores/RR2_BBF.csv`
(14 seeds): mean ≈ 1125, median 1118, min 573, max 1490 — seed variance is
large even upstream.

Ablation switches (each is one design choice from the table):

```shell
python src/train.py experiment=bbf/atari100k algorithm.spr_weight=0          # no SPR
python src/train.py experiment=bbf/atari100k algorithm.reset_interval=0      # no resets
python src/train.py experiment=bbf/atari100k algorithm.data_augmentation=false
python src/train.py experiment=bbf/atari100k algorithm.width_scale=1         # small net
python src/train.py experiment=bbf/atari100k algorithm.prioritized=false     # uniform replay
python src/train.py experiment=bbf/atari100k \
    algorithm.max_update_horizon=3 algorithm.min_gamma=0.997                  # no annealing
```



## Experimental results

**Live W&B table (canonical):** [LatentLab/torchrl-hydra-template — Table](https://wandb.ai/LatentLab/torchrl-hydra-template/table)


| Run | Environment | Config | Seed | Frames | Eval return | Notes |
|-----|-------------|--------|------|--------|-------------|-------|
| [bbf_Jamesbond_2026-08-03_10-06-24](https://wandb.ai/LatentLab/torchrl-hydra-template/runs/vwbpoxw5) | ALE/Jamesbond-v5 | `experiment=bbf/atari100k` | 1 | 100,000 | 1,251.0 | — |
| [bbf_Jamesbond_2026-08-03_10-11-48](https://wandb.ai/LatentLab/torchrl-hydra-template/runs/dc2e91w2) | ALE/Jamesbond-v5 | `experiment=bbf/atari100k` | 2 | 100,000 | 702.0 | — |
| [bbf_Jamesbond_2026-08-03_11-59-19](https://wandb.ai/LatentLab/torchrl-hydra-template/runs/xxxxqws7) | ALE/Jamesbond-v5 | `experiment=bbf/atari100k` | 3 | 100,000 | 524.5 | — |


*No runs recorded yet. This table is regenerated from W&B runs tagged* `template`
*by* `scripts/update_algo_results.py`*.*