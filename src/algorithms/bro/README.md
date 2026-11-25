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

SAC scaled up and stabilised, so it can run at a high replay ratio:

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
- Networks are built in `setup()` from scalar kwargs (the TD-MPC2 precedent):
  the optimistic actor consumes the conservative actor's mean and a reset must
  restore everything at once.
- Diagnostics are the last update's values, read only at logging boundaries
  (`pop_train_metrics`), so the update loop has no host sync.

## Verification

Reference: per-seed scores at 100k environment steps for BRO UTD 10 on the 28
DMC tasks, as rerun by the SimbaV2 authors
(`data/external/dmc/scores_per_seed.csv` in the FastRL paper repository).
The verification sweep is `slurm/sweeps/dmc100k_verify.yaml`.
