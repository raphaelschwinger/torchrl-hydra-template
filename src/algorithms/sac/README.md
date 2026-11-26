# SAC: Soft Actor-Critic

Haarnoja et al. (2018), [arXiv:1812.05905](https://arxiv.org/abs/1812.05905).
Adapted from [ikostrikov/jaxrl](https://github.com/ikostrikov/jaxrl) (MIT), in the
form the BRO code keeps it, because SAC here is the base class of
[BRO](../bro/README.md).

```shell
python src/train.py experiment=sac/dmc environment.task=walker-walk
```

## Pseudocode

```
after init_random_frames random agent steps, per agent step:
  a ~ pi(s) = tanh N(mu(s), sigma(s));  store (s, a, r, s')
  repeat num_updates times on a fresh uniform minibatch:
    critic <- (Q_i(s,a) - y)^2,  y = r + γ (Q_lb'(s', a') − α log π(a'|s')),  a' ~ π(s')
    target <- Polyak(τ)
    actor  <- min α log π(a|s) − Q_lb(s, a),  a ~ π(s)
    α      -> target entropy
```

`Q_lb = mean(Q1, Q2) − pessimism·|Q1 − Q2|/2`; `pessimism: 1` (default) is the
clipped double-Q `min(Q1, Q2)`.

## Defaults (jaxrl SAC)

MLP 256×2 actor and critics with orthogonal init, Adam 3e-4 (actor, critic,
temperature with β1 = 0.5), batch 256, γ = 0.99, τ = 0.005, initial temperature
1, target entropy −|A|/2, one update per agent step, 5 000 random warm-up steps.
`experiment=sac/dmc` runs DMC100k: 50k agent steps at action repeat 2.

## Structure, and how BRO extends it

`SACAlgorithm` keeps the SAC update readable (`_update`: critic → target →
actor → temperature) and exposes the points BRO changes: the trunk classes
(`actor_trunk`, `critic_trunk`), `_build_networks`, `_optimizer_specs`,
`_td_target` / `_critic_loss` / `_reduce_q` (quantile critic), `_update`
(optimistic actor), `_before_updates` (resets) and `_make_explore_policy`.
Networks are built in `setup()` from scalar kwargs (the TD-MPC2 precedent), so
BRO's trunk swap does not need a second factory mechanism.

Runtime knobs, all off by default and shared with BRO: `compile` (+
`compile_scope: step` for one CUDA graph per agent step), `amp`, `tf32`,
`fused_optim`, `buffer_device`.

## Verification

`scripts/sweeps/sac_dmc100k_val.yaml`: six DMC tasks × 3 seeds at 100k env
steps, against the SAC reruns by the TD-MPC2, SimBa and SimbaV2 authors
(`data/external/dmc/scores_per_seed.csv` in the FastRL paper repository).
Those three SAC implementations differ in network size and warm-up, so their
spread at 100k is the target range rather than one number.

### DMC100k port check (2026-11-25)

Final eval return (10 episodes) at 100k env steps, 3 seeds, mean (min–max),
against the SAC reruns at 100k env steps:

| Task | Ours | SimbaV2 authors | TD-MPC2 authors | SimBa authors |
|---|---|---|---|---|
| cheetah-run | 240 (162–284) | 178 (141–215) | 310 (237–362) | 329 (271–416) |
| walker-walk | 511 (439–582) | 449 (366–532) | 708 (510–918) | 783 (605–959) |
| finger-spin | 471 (453–485) | 475 (456–493) | 762 (744–774) | 614 (499–714) |
| reacher-easy | 906 (785–971) | 931 (871–969) | 540 (68–813) | 882 (778–972) |
| quadruped-walk | 153 (120–208) | 114 (30–171) | 55 (20–86) | 76 (16–189) |
| cartpole-swingup | 848 (841–859) | 861 (855–864) | 835 (826–845) | 853 (840–865) |

Ours tracks the SimbaV2 authors' jaxrl-style SAC on every task; the TD-MPC2
and SimBa reruns use other networks and settings.
