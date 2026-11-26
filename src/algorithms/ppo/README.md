# Proximal Policy Optimization (PPO)

**Paper:** Schulman et al. (2017), [*Proximal Policy Optimization Algorithms*](https://arxiv.org/abs/1707.06347) — clip variant.

On-policy **actor-critic** for **continuous or discrete** control. Each collector batch is
one rollout; advantages are computed once with GAE, then the policy and value networks are
updated for $K$ epochs of shuffled mini-batches under a clipped surrogate objective, and
the data is discarded.

The implementation follows [cleanRL's PPO](https://docs.cleanrl.dev/rl-algorithms/ppo/)
(`ppo_continuous_action.py` / `ppo_atari.py`) and
[*The 37 Implementation Details of PPO*](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/),
built from TorchRL components (`ClipPPOLoss`, `GAE`, `ProbabilisticActor`,
`ActorValueOperator`). See also the
[torchrl SOTA PPO references](https://github.com/pytorch/rl/tree/main/sota-implementations/ppo).

## Key ideas

- **Clipped surrogate objective.** With importance ratio
  $r_t(\theta) = \pi_\theta(a_t \mid s_t) / \pi_{\theta_\text{old}}(a_t \mid s_t)$, PPO maximises
  $\min\bigl(r_t A_t,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\bigr)$, keeping each update
  close to the data-collecting policy without a KL constraint.
- **Multiple epochs per rollout.** Unlike A2C's single pass, PPO reuses each rollout for
  `num_epochs` epochs of shuffled mini-batches — the clipping makes this safe.
- **GAE advantages** (λ=`gae_lambda`), computed once per rollout; value targets
  $V^{\text{target}}_t = A_t + V(s_t)$. GAE is a mathematical bridge between Monte Carlo
  (MC) and temporal-difference (TD) methods, and it fits well with on-policy algorithms:
  each update has only a strictly limited batch of freshly collected transitions, so the
  gradient signal needs to stay exceptionally clean — pure MC would introduce lots of
  variance, while pure TD would introduce significant bias.
- **Clipped value loss** (`clip_value`): the value prediction is clipped around the
  rollout-time value estimate with the same $\epsilon$.
- **Stochastic actor.** Continuous: Normal with state-independent learned log-std
  (`AddStateIndependentNormalScale`), actions clipped to bounds env-side. Discrete:
  Categorical over logits. Collection samples (`ExplorationType.RANDOM`); evaluation uses
  the distribution mode (mean / argmax).
- **Optional shared trunk.** For pixels, `common_network` shares a Nature-CNN trunk between
  actor and critic heads via `ActorValueOperator` (cleanRL's Atari agent).

## Pseudocode

1. Initialise policy $\pi(\cdot \mid s; \theta)$ and value function $V(s; \phi)$.

**For each iteration:**

2. Roll out $N$ steps with $\pi_{\theta_\text{old}}$, storing $\log \pi_{\theta_\text{old}}(a_t \mid s_t)$.
3. Compute GAE advantages $A_t$ and value targets $V^{\text{target}}_t$ (once).

**For each of $K$ epochs, for each shuffled mini-batch:**

4. $r_t(\theta) = \exp\bigl(\log \pi_\theta(a_t \mid s_t) - \log \pi_{\theta_\text{old}}(a_t \mid s_t)\bigr)$.
5. Maximise $\min\bigl(r_t A_t,\ \text{clip}(r_t, 1-\epsilon, 1+\epsilon) A_t\bigr) + \beta\, \mathcal{H}\bigl[\pi(\cdot \mid s_t)\bigr]$ with $A_t$ normalised per mini-batch.
6. Minimise $\bigl(V(s_t; \phi) - V^{\text{target}}_t\bigr)^2$ (optionally $\epsilon$-clipped around the rollout value).
7. One backward pass, global grad-norm clip, optimiser step on the summed actor + critic loss.

## Implementation in this template

| Resource | Path |
|----------|------|
| Algorithm | [`ppo.py`](ppo.py) |
| Shared networks | [`src/components/networks.py`](../../components/networks.py) |
| HPs (state / continuous) | [`configs/algorithm/ppo.yaml`](../../../configs/algorithm/ppo.yaml) |
| Policy (pixels / Atari) | [`configs/algorithm/policy/nature_cnn_categorical.yaml`](../../../configs/algorithm/policy/nature_cnn_categorical.yaml) |
| Experiment: DMC cheetah-run 1M | [`configs/experiment/ppo/dmc.yaml`](../../../configs/experiment/ppo/dmc.yaml) |
| Experiment: DMC100k | [`configs/experiment/ppo/dmc100k.yaml`](../../../configs/experiment/ppo/dmc100k.yaml) |
| Experiment: Atari-100k JamesBond | [`configs/experiment/ppo/ale.yaml`](../../../configs/experiment/ppo/ale.yaml) |

```shell
python src/train.py experiment=ppo/dmc
python src/train.py experiment=ppo/ale
```

### Mapping pseudocode → code

| Pseudocode step | Where in code |
|-----------------|---------------|
| Stochastic policy π | `actor_network()` → `ProbabilisticActor` (+ optional `common_network` trunk via `ActorValueOperator`) |
| Value V(s) | `value_network()` → `ValueOperator` |
| Rollout collection, old log-probs | Trainer `Collector` with `get_explore_policy()`; `return_log_prob=True` |
| GAE advantages (once) | `step()` → `adv_module(batch)` under `no_grad`, before flattening |
| K epochs of shuffled mini-batches | `TensorDictReplayBuffer` + `SamplerWithoutReplacement`, iterated `num_epochs` times |
| Clipped surrogate + value loss | `ClipPPOLoss(clip_epsilon, clip_value, normalize_advantage)` → single `optimizer.step()` |
| LR annealing | `step()` decays Adam lr linearly over `anneal_frames` |

### Implementation details ("the 37 tricks")

Core and continuous-action details from the
[ICLR blog post](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/):

| # | Detail | Where |
|---|--------|-------|
| 1 | Vectorized architecture (rollout of N steps × M envs) | `frames_per_batch` + `trainer.num_envs` |
| 2 | Orthogonal init (√2 hidden, 0.01 policy head, 1.0 value head), zero biases | `src/components/networks.py::orthogonal_init_` |
| 3 | Adam ε = 1e-5 | `adam_eps` |
| 4 | Linear LR annealing to 0 | `anneal_lr` / `anneal_frames` |
| 5 | GAE (with value bootstrapping for truncated rollouts) | `GAE` module |
| 6 | Mini-batch updates (shuffled, without replacement) | `SamplerWithoutReplacement` |
| 7 | Advantage normalization per mini-batch | `ClipPPOLoss(normalize_advantage=True)` |
| 8 | Clipped surrogate objective | `ClipPPOLoss(clip_epsilon)` |
| 9 | Value function loss clipping | `clip_value` |
| 10 | Combined loss (`vf_coef`), one optimizer | `critic_coeff=0.25` ≡ cleanRL `0.5·MSE·0.5` |
| 11 | Entropy bonus | `entropy_coeff` |
| 12 | Global gradient-norm clipping (0.5) | `max_grad_norm` |
| 13 | (Debug) KL early stop | `target_kl` (off by default) |
| C1 | Continuous actions via Normal distribution | `IndependentNormal` |
| C2 | State-independent log std (init 0) | `AddStateIndependentNormalScale` |
| C3 | Action clipping to valid range | env-side `ClipTransform(in_keys_inv=[action])` — stored action/log-prob stay unclipped |
| C4 | Observation normalization | env-side `VecNorm(in_keys=[observation])` |
| C5 | Observation clipping to [-10, 10] | env-side `ClipTransform` |
| A1–A7 | Atari wrappers (noop reset, frame skip, episodic life, reward sign, 84×84 grayscale, frame stack) | `configs/experiment/ppo/ale.yaml` |

**Documented deviations:**

- **No reward normalization** (trick C6): `VecNorm` on rewards standardises (subtracts the
  mean), which is not cleanRL's discounted-return scaling and can change the optimal
  policy. DMC rewards are already in $[0, 1]$; the torchrl SOTA MuJoCo PPO also skips it.
- **GAE computed once per rollout** (cleanRL) rather than re-computed each epoch (torchrl
  SOTA): `clip_value` anchors on the rollout-time `state_value`, which re-computation
  would drift.
- **Atari-100k tuning:** `num_epochs: 10` (cleanRL uses 4) for more sample reuse at the
  100k-step budget (~97 rollouts); everything else matches cleanRL's `ppo_atari.py`.
- **Eval `VecNorm` stats are synced from training:** running statistics are env state
  and are not checkpointed, so the trainer copies the training env's statistics into
  the eval env and freezes them before every evaluation (`evaluation.sync_obs_norm`).
  `ale_eval.yaml` drops `VecNorm` entirely, so there it is a no-op.

## DMC100k (shared DMC protocol)

`experiment=ppo/dmc100k` runs PPO under the protocol of `sac/dmc`, `bro/dmc`
and `tdmpc2/dmc`: 100k env steps = 50k agent steps at action repeat 2, eval
every 10k agent steps on 10 episodes with the deterministic policy, and a
separate 10-episode final eval as the score. The hyperparameters are cleanRL's,
unchanged from `ppo/dmc` (tuned for 1M steps): at this budget a 2048-step
rollout gives only ~24 PPO iterations.

```shell
./scripts/run_sweep.sh --sweep scripts/sweeps/ppo_dmc100k_val.yaml --gpus 0,1,2
```

### Results (2026-11-26, Lena, 3 seeds)

Final eval return, mean (min–max). No paper reports PPO on DMC at 100k env
steps. The closest published numbers are MR.Q's PPO baseline (Fujimoto et al.,
2025, [arXiv:2501.16142](https://arxiv.org/abs/2501.16142), Table 5: Stable
Baselines 3 defaults, action repeat 2, 10 seeds) at **500k agent steps = 1M env
steps, 10x our budget**, so it is an upper reference, not a target.

| Task | Ours, 100k env steps | MR.Q's PPO, 1M env steps |
|---|---|---|
| cheetah-run | 75 (69–81) | 269 [247, 295] |
| walker-walk | 43 (36–49) | 283 [253, 312] |
| finger-spin | 18 (3–48) | 459 [420, 497] |
| reacher-easy | 636 (399–915) | 367 [188, 558] |
| quadruped-walk | 208 (199–214) | 122 [103, 142] |
| cartpole-swingup | 194 (148–257) | 776 [661, 853] |

At 100k env steps PPO is far below the off-policy methods on the same protocol
(SAC: cheetah-run 240, walker-walk 511, finger-spin 471), as expected for an
on-policy method at this budget. reacher-easy and quadruped-walk sit above
MR.Q's 1M-step numbers; with 3 seeds and SB3's different defaults, read that as
noise and implementation spread, not as PPO improving with less data.

**Deterministic vs stochastic policy.** After 24 iterations PPO's
state-independent std is still near its initial value of 1, so the mean action
is a much weaker policy than the sampled one on some tasks. finger-spin seed 1:
the deterministic eval scores 48, the stochastic policy 101 in the same eval
env, matching the training stream (100). The protocol reports the
deterministic score, like the other DMC100k experiments.

**Eval normalisation fix.** These runs use `evaluation.sync_obs_norm: true`
(the new default): before each evaluation the trainer copies the training
env's `VecNorm` statistics into the eval env and freezes them there (SB3's
`sync_envs_normalization`). Before the fix the eval env normalised with its own
cold statistics, and the stochastic policy scored 1.4 on finger-spin in the
eval env against 100 on the training stream. The first sweep without the fix
(`logs/sweeps/ppo-dmc100k-val.cold-vecnorm` on Lena) scored cheetah-run 102,
walker-walk 55, finger-spin 2, reacher-easy 655, quadruped-walk 219,
cartpole-swingup 224. Its numbers on other tasks were not uniformly lower,
because cold statistics adapted to the deterministic policy's own state
distribution, but they did not measure the trained agent.

## Experimental results

**Live W&B table (canonical):** [LatentLab/torchrl-hydra-template — Table](https://wandb.ai/LatentLab/torchrl-hydra-template/table)

| Run | Environment | Config | Seed | Frames | Eval return | Notes |
|-----|-------------|--------|------|--------|-------------|-------|
| [ppo_Jamesbond_2026-08-03_10-01-01](https://wandb.ai/LatentLab/torchrl-hydra-template/runs/79m7ei5q) | ALE/Jamesbond-v5 | `experiment=ppo/ale` | 1 | 100,000 | 52.0 | — |
| [ppo_Jamesbond_2026-08-03_10-01-13](https://wandb.ai/LatentLab/torchrl-hydra-template/runs/s2ik9uaz) | ALE/Jamesbond-v5 | `experiment=ppo/ale` | 2 | 100,000 | 36.5 | — |
| [ppo_Jamesbond_2026-08-03_10-06-24](https://wandb.ai/LatentLab/torchrl-hydra-template/runs/hjahokqn) | ALE/Jamesbond-v5 | `experiment=ppo/ale` | 3 | 100,000 | 29.5 | — |
| [ppo_cheetah-run_2026-08-03_15-13-33](https://wandb.ai/LatentLab/torchrl-hydra-template/runs/riox94hm) | cheetah-run | `experiment=ppo/dmc` | 1 | 1,000,000 | 334.1 | — |
| [ppo_cheetah-run_2026-08-03_15-17-59](https://wandb.ai/LatentLab/torchrl-hydra-template/runs/mkznx8c5) | cheetah-run | `experiment=ppo/dmc` | 2 | 1,000,000 | 551.9 | — |
| [ppo_cheetah-run_2026-08-03_16-06-55](https://wandb.ai/LatentLab/torchrl-hydra-template/runs/qpkb0y13) | cheetah-run | `experiment=ppo/dmc` | 3 | 1,000,000 | 503.2 | — |
