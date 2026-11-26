# TD3: Twin Delayed Deep Deterministic policy gradient

Fujimoto, van Hoof & Meger (2018), [arXiv:1802.09477](https://arxiv.org/abs/1802.09477).
Adapted from the authors' reference code, [sfujim/TD3](https://github.com/sfujim/TD3)
(MIT), and written as a subclass of [SAC](../sac/README.md): the replay buffer,
double critic, optimiser plumbing and runtime knobs are SAC's, and `td3.py`
holds only what TD3 changes.

```shell
python src/train.py experiment=td3/dmc environment.task=walker-walk
```

## Pseudocode

```
after init_random_frames random agent steps, per agent step:
  a = clip(π(s) + N(0, σ_explore), −1, 1);  store (s, a, r, s')
  repeat num_updates times on a fresh uniform minibatch:
    a'     = clip(π'(s') + clip(N(0, σ_target), −c, c), −1, 1)      # target policy smoothing
    critic <- Σ_i (Q_i(s,a) − y)²,  y = r + γ min(Q1'(s', a'), Q2'(s', a'))   # clipped double-Q
    every policy_delay critic updates:                               # delayed updates
      actor   <- min −Q1(s, π(s))
      targets <- Polyak(τ) for π' and Q1', Q2'
```

## What TD3 changes in SAC

| SAC | TD3 | Where |
|---|---|---|
| tanh-Gaussian actor | deterministic `tanh(MLP(s))` actor with a target copy | `networks.py::DeterministicActor`, `TD3Networks` |
| sample from the actor to explore | `π(s) + N(0, 0.1)`, clipped | `TD3PolicyModule` |
| entropy term in the target, learned temperature | target policy smoothing, no temperature | `_update_critic` |
| actor maximises `min(Q1, Q2) − α log π` | actor maximises `Q1` | `_update_actor_and_targets` |
| actor, target critic every update | actor, target actor and target critics every `policy_delay` (2) updates | `_update` |
| jaxrl orthogonal init | PyTorch's default `nn.Linear` init (the reference) | `networks.py::TorchMLP` |

SAC gained two small hooks for this, `_make_eval_policy` and
`_compile_update`; SAC and BRO are unchanged by them. `compile` works with
`compile_scope: update` (the critic step and the actor+target step are
compiled separately, since the policy delay branches between updates);
`compile_scope: step` is rejected.

## Defaults (sfujim/TD3)

MLP 256×2 ReLU actor and critics, Adam 3e-4 for both, batch 256, γ = 0.99,
τ = 0.005, exploration noise 0.1, target noise 0.2 clipped at 0.5, policy delay 2,
one update per agent step and 25k random warm-up steps (`start_timesteps`).

`experiment=td3/dmc` runs DMC100k (50k agent steps at action repeat 2) and
**deviates in one place**: the warm-up is 5k agent steps, the same as
`sac/dmc`, because the reference 25k would be half of the 50k-step budget.
DMC's action bounds are [−1, 1], so `max_action = 1`.

## Verification

`scripts/sweeps/td3_dmc100k_val.yaml`: the six DMC tasks of the SAC, BRO and
TD-MPC2 port checks × 3 seeds at 100k env steps, final eval of 10 episodes.

RESULTS_PLACEHOLDER
