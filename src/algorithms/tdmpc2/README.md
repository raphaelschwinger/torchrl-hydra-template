# TD-MPC2

**Paper:** Hansen, Su & Wang (2024), [*TD-MPC2: Scalable, Robust World Models for Continuous Control*](https://arxiv.org/abs/2310.16828).
**Upstream code:** [nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2) (MIT). This
implementation is adapted from it; each adapted file carries a source-attribution header.

**Model-based** off-policy RL for **continuous** control. A latent world model
(encoder $h$, dynamics $d$, reward $R$, policy prior $p$, Q-ensemble) is learned from
replayed trajectory slices; actions are selected by **MPPI planning** over short latent
rollouts, seeded by the policy prior.

## Background: what is MPC, and why fuse it with TD learning?

**Model Predictive Control (MPC)** is a *closed-loop* planning scheme: at every
environment step, use a model to simulate candidate action sequences over a short
horizon $H$ starting from the *current* observation, pick the best sequence — but
**execute only its first action**. At the next step, the plan is thrown away and
planning starts over from the newly observed state ("receding horizon" control). This
constant replanning is what makes MPC robust to model errors: a plan is only ever
trusted for one step before it is re-checked against reality, so prediction errors
cannot compound over a whole episode. The price is compute — a full optimisation
problem is solved at every single env step.

Pure MPC has a **horizon problem**: the planner can only value what it can simulate.
With a short horizon it is myopic (it would never take an action whose payoff arrives
after step $H$), and extending the horizon makes planning both exponentially harder and
less accurate, since latent-model errors accumulate with rollout length.

**The TD-MPC fusion** solves this by splitting the return estimate in two:

$$\hat{G}(z_0, a_{0:H-1}) \;=\; \underbrace{\sum_{t=0}^{H-1} \gamma^t\, R(z_t, a_t)}_{\text{planned: model rollout}} \;+\; \underbrace{\gamma^H\, Q\bigl(z_H,\, p(z_H)\bigr)}_{\text{learned: TD value function}}$$

MPC handles the *near* future — the first $H$ steps (default $H=3$), where the learned
dynamics are still accurate — and a Q-function trained by ordinary temporal-difference
learning summarises *everything beyond* the horizon as a terminal value. The plan
therefore only needs to look a few steps ahead instead of to the end of the episode:
the TD value turns short-horizon MPC into a far-sighted controller. Conversely, the
planner improves on the raw policy $p$ at test time (it locally searches around it), so
TD-MPC behaves like a policy-improvement operator applied at every action selection.
This is exactly `MPPIPlanner._estimate_value()` in [`planner.py`](planner.py): rewards
are accumulated for `horizon` steps, then the bootstrap term
$\gamma^H Q(z_H, \cdot)$ is added.

## Key ideas

- **Implicit (decoder-free) world model.** The latent $z = h(s)$ is trained only to
  predict *rewards, values and its own future* — no observation reconstruction. A
  consistency loss $\lVert d(z_t, a_t) - h(s_{t+1}) \rVert^2$ grounds the dynamics.
- **SimNorm latents.** The latent is projected onto a product of simplices (softmax over
  groups of `simnorm_dim` entries), which bounds it and stabilises long rollouts.
- **Discrete regression (two-hot).** Reward and value heads predict a categorical
  distribution over `num_bins` symlog-spaced bins and are trained with soft
  cross-entropy — much more robust to reward scale than L2 (as in DreamerV3).
- **Q-ensemble.** `num_q` (default 5) vmapped Q-heads; TD targets use the min of two
  randomly subsampled heads (target network, Polyak `tau`), value estimates in planning
  use the average.
- **MPPI planning.** At every env step, sample `num_samples` action sequences of length
  `horizon` (some rolled out from the policy prior $p$), score each with model rewards
  plus the terminal Q-value (the TD-MPC fusion above), refit a Gaussian to the
  `num_elites` best, iterate, then execute the first action of a sampled elite. The
  plan mean is warm-started from the previous step.
- **Policy prior.** A tanh-squashed Gaussian trained to maximise (running-scale
  normalised) Q-values plus entropy — it seeds the planner and computes TD targets.

## Pseudocode

**For each environment step:**

1. $a_t \leftarrow \text{MPPI}(h(s_t))$ — plan over latent rollouts, warm-started; add
   $\sigma_0$ noise during exploration.
2. Store $(s_t, a_t, r_t, s_{t+1})$ in the slice replay buffer $\mathcal{D}$.
3. Sample $B$ trajectory slices $(s_{0:H}, a_{0:H-1}, r_{0:H-1})$ from $\mathcal{D}$.
4. Latent rollout: $\hat z_{t+1} = d(\hat z_t, a_t)$ with $\hat z_0 = h(s_0)$.
5. Losses, each $\rho^t$-weighted over the horizon:
   - consistency: $\lVert \hat z_{t+1} - h(s_{t+1}) \rVert^2$
   - reward: $\text{soft-CE}\bigl(R(\hat z_t, a_t),\, r_t\bigr)$
   - value: $\text{soft-CE}\bigl(Q(\hat z_t, a_t),\, r_t + \gamma \min Q_{\text{target}}(z_{t+1}, p(z_{t+1}))\bigr)$
6. Update the policy prior $p$ on detached latents:
   $\max_p \mathbb{E}\bigl[Q(z, p(z))/S + \alpha \mathcal{H}(p)\bigr]$ ($S$: running Q scale).
7. Polyak-update the target Q-ensemble.

## Implementation in this template

| Resource | Path |
|----------|------|
| Algorithm | [`tdmpc2.py`](tdmpc2.py) |
| World model | [`world_model.py`](world_model.py) |
| MPPI planner | [`planner.py`](planner.py) |
| Policy wrapper | [`policy.py`](policy.py) |
| Shared components | [`src/components/`](../../components/) (`math.py`, `layers.py`, `scale.py`) |
| HPs | [`configs/algorithm/tdmpc2.yaml`](../../../configs/algorithm/tdmpc2.yaml) |
| Experiment | [`configs/experiment/tdmpc2/dmc.yaml`](../../../configs/experiment/tdmpc2/dmc.yaml) |

```shell
python src/train.py experiment=tdmpc2/dmc
python src/train.py experiment=tdmpc2/dmc environment.task=walker-walk
```

Defaults reproduce the upstream single-task config at `model_size=5` (~5M params:
`enc_dim=256, mlp_dim=512, latent_dim=512, num_q=5`), the setting used for all
single-task results in the paper. Scope: **single-task, state observations** (the
upstream multi-task embeddings, action masks and pixel encoder are not ported).

### Mapping pseudocode → code

| Pseudocode step | Where in code |
|-----------------|---------------|
| MPPI planning | `MPPIPlanner.plan()` (warm-start via `is_init` from `InitTracker`) |
| Store transitions | `step()` → `replay_buffer.extend(...)` (`SliceSampler`, `traj_key="episode"`) |
| Sample slices | `_sample()` → `[horizon+1, B]` obs, `[horizon, B]` action/reward |
| Latent rollout + consistency | `_latent_rollout()` |
| Reward/value soft-CE | `_reward_value_loss()`, targets from `_td_target()` |
| Policy prior update | `update_pi()` (`RunningScale` Q normalisation) |
| Target Q Polyak update | `WorldModel.soft_update_target_Q()` |

### Deviations from the template's factory convention

The world model and replay buffer are built internally from **scalar hyperparameters**
rather than `_partial_` factories: the five subnetworks share
`latent_dim`/`simnorm_dim`/`num_bins` coupling, and compatibility with official
upstream checkpoints pins both the architecture and the parameter names.

### Update schedule

Upstream performs 1 gradient update per env step after 2,500 seed steps, plus a
2,500-update pretraining burst at the seed boundary. This maps to
`frames_per_batch=50` + `num_updates=50` (UTD = 1 at 50-step granularity) and
`init_random_frames=2500` + `pretrain_updates=null` (→ 2,500). The trainer performs no
in-training eval; train-episode returns (`RewardSum`) track eval returns closely, and
checkpoints (every 50k steps) can be evaluated offline with `src/eval.py`.

### Loading official checkpoints

`load_checkpoint()` accepts both this template's `TrainingState` checkpoints and
official `{"model": state_dict}` checkpoints from
[tdmpc2.com/models](https://www.tdmpc2.com/models) (hosted on
[HuggingFace `nicklashansen/tdmpc2`](https://huggingface.co/nicklashansen/tdmpc2));
old-API checkpoints are converted via `api_model_conversion`.

```shell
curl -sL -o checkpoints/cheetah-run-1.pt \
  "https://huggingface.co/nicklashansen/tdmpc2/resolve/main/dmcontrol/cheetah-run-1.pt"
python src/eval.py algorithm=tdmpc2 environment=dmc \
  checkpoint.resume_from=$PWD/checkpoints/cheetah-run-1.pt \
  trainer.accelerator=gpu evaluation.final_num_episodes=10
```

Verified result with the official `cheetah-run-1.pt` (2023-06-03) through this port:
**eval/return_mean 866.9 ± 11.1** (10 episodes).

## Experimental results

**Live W&B table (canonical):** [LatentLab/torchrl-hydra-template — Table](https://wandb.ai/LatentLab/torchrl-hydra-template/table)

| Run | Environment | Config | Seed | Frames | Eval return | Notes |
|-----|-------------|--------|------|--------|-------------|-------|
| [tdmpc2_cheetah-run_2026-08-03_16-09-04](https://wandb.ai/LatentLab/torchrl-hydra-template/runs/nd3ojze5) | cheetah-run | `experiment=tdmpc2/dmc` | 1 | 1,000,000 | 899.7 | — |
| [tdmpc2_cheetah-run_2026-08-03_16-57-43](https://wandb.ai/LatentLab/torchrl-hydra-template/runs/b1wqk7c2) | cheetah-run | `experiment=tdmpc2/dmc` | 2 | 1,000,000 | 920.4 | — |
| [tdmpc2_cheetah-run_2026-08-03_19-51-37](https://wandb.ai/LatentLab/torchrl-hydra-template/runs/gxaltszp) | cheetah-run | `experiment=tdmpc2/dmc` | 3 | 1,000,000 | 901.7 | — |
