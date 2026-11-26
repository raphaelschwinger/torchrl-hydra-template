# BRO: Bigger, Regularized, Optimistic

Nauman et al. (2024), [arXiv:2405.16158](https://arxiv.org/abs/2405.16158).
Ported from the official JAX code,
[naumix/BiggerRegularizedOptimistic](https://github.com/naumix/BiggerRegularizedOptimistic)
(`jaxrl/bro/`, `train_parallel.py`, `configs/bro_default.py`), MIT license.

```shell
python src/train.py experiment=bro/dmc environment.task=walker-walk
python src/train.py experiment=bro/dmc algorithm.num_updates=2   # BRO (Fast), UTD 2
```

## What it is

SAC scaled up and stabilised, so it can run at a high replay ratio.
`BROAlgorithm` subclasses [`SACAlgorithm`](../sac/README.md): `bro.py` holds
only what BRO adds (BroNet trunks, quantile critic, optimistic actor and its
duals, resets), and the SAC update stays in `sac.py`.


| Ingredient | Here |
|---|---|
| **Bigger** critic | BroNet: Dense→LN→ReLU, then 2 residual blocks (Dense→LN→ReLU→Dense→LN, + skip), width 512, two independent critics |
| **Regularized** | LayerNorm everywhere, AdamW (weight decay 1e-4), full parameter **resets** at agent steps 15 001, 50 001, 250 001, … (only the first at UTD 2) |
| **Optimistic** exploration | a second, optimistic actor shifts the conservative actor's mean to maximise `Q_ub = mean(Q1,Q2) + β_ub·|Q1−Q2|/2`, KL-regularised towards it with std ×0.75; β_ub and the KL weight are dual variables driven to a per-dimension KL target of 0.05 |
| Distributional critic | 100 quantiles, quantile-Huber loss (κ=1) |
| Replay ratio | 10 updates per agent step, batch 128, each on its own uniform minibatch |

The policy that is evaluated is `tanh(mean)` of the conservative actor; data is
collected with the optimistic actor.

## Pseudocode

```
after 2 500 random agent steps, per agent step:
  a ~ tanh N(mu_c(s) + dmu_o(s, mu_c), 0.75 sigma_c(s))      # optimistic actor
  store (s, a, r, s')
  if step in reset_steps: re-initialise all networks and optimisers
  repeat 10 times on a fresh minibatch:
    critic   <- quantile-Huber(Z_i(s,a), r + γ(mean_k Z'_k(s', a') − α log π_c(a'|s')))
    target   <- Polyak(τ = 0.005)
    π_c      <- min  α log π_c(a|s) − mean Q(s, a)
    π_o      <- min −mean Q_ub(s, a_o) + λ KL(π_o || π_c)
    α        -> target entropy −|A|/2;  β_ub, λ -> target KL 0.05
```

## Deviations from the official code

- **One seed per process.** The official script vmaps 10 seeds through one
  JAX program; here each seed is its own run (as for every algorithm in the
  template). Vectorising seeds is a runtime question, not a learning one.
- **Resets restore the initial weights**, as the official `init_models(seeds)`
  does by re-using the same PRNG keys.
- **Action repeat 2** (the template's `environment: dmc` stack and the FastRL
  DMC100k protocol: 100k environment steps = 50k agent steps). The official
  script steps DMC with action repeat 1. Reset steps and the 2 500 warm-up
  steps are counted in agent steps, as the official loop counts them.
- Networks are built in `setup()` from scalar kwargs, inherited from SAC
  (the TD-MPC2 precedent): the optimistic actor consumes the conservative
  actor's mean and a reset must restore everything at once.
- The SAC temperature is updated before the optimistic actor (in the official
  code after it). The two touch disjoint parameters, so the result is the same.
- Diagnostics are the last update's values, read only at logging boundaries
  (`pop_train_metrics`), so the update loop has no host sync.

## Verification

Reference: per-seed scores at 100k environment steps for BRO UTD 10 on the 28
DMC tasks, as rerun by the SimbaV2 authors
(`data/external/dmc/scores_per_seed.csv` in the FastRL paper repository).
Port check: `scripts/sweeps/bro_dmc100k_val.yaml`; runtime ablation:
`scripts/sweeps/bro_optimisations_ablation.yaml` (knobs `compile`,
`compile_scope`, `amp`, `tf32`, `fused_optim`, `buffer_device`; all off by default).

### DMC100k port check (2026-11-25)

Final eval return (10 episodes) at 100k env steps, 3 seeds, mean (min–max),
against BRO UTD 10 as rerun by the SimbaV2 authors (5 seeds). cheetah-run ran
the published code path; the other tasks the fast path
(`compile: reduce-overhead`, `compile_scope: step`, same equations), which on
cheetah-run gave the same scores (508, 514).

| Task | Ours (action repeat 2) | Reference |
|---|---|---|
| cheetah-run | 571 (512–608) | 531 (429–639) |
| walker-walk | 956 (941–965) | 963 (956–976) |
| finger-spin | 861 (816–949) | 979 (972–986) |
| reacher-easy | 871 (863–877) | 877 (688–976) |
| quadruped-walk | 730 (603–943) | 810 (642–935) |
| cartpole-swingup | 873 (865–879) | 872 (863–879) |

finger-spin at **action repeat 1** (100k agent steps, as the official code
runs DMC): 970 (940–988). The reference curves count environment steps at
action repeat 1, i.e. twice our agent steps and updates; on fine-control tasks
that shows, on the others it does not.
