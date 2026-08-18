# Synchronous Advantage Actor-Critic (A2C)

**Paper:** Mnih et al. (2016), [*Asynchronous Methods for Deep Reinforcement Learning*](https://arxiv.org/abs/1602.01783) — this template implements the **synchronous** variant (single process, no shared RMSProp).

On-policy **actor-critic** for **continuous** control. Each collector batch is one rollout;
advantages are computed with GAE, then the policy and value networks are updated for one
epoch of mini-batches and the data is discarded.

## Key ideas

- **Policy gradient with value baseline.** The actor maximises expected return; the critic
  $V(s)$ reduces variance of gradient estimates.
- **Advantage estimation (GAE).** Generalized Advantage Estimation (`GAE`, λ=`gae_lambda`)
  balances bias and variance when computing $A_t$ from the rollout.
- **Entropy bonus.** Optional `entropy_coeff` encourages exploration by rewarding policy
  entropy (default 0 for MuJoCo in this template).
- **On-policy schedule.** No long-term replay buffer, no target networks, no warm-up.
  `frames_per_batch / mini_batch_size` mini-batches per iteration, sampled without
  replacement exactly once.
- **Stochastic actor.** `TanhNormal` distribution over bounded actions; collection uses
  `ExplorationType.RANDOM`, evaluation uses the distribution mode (deterministic).

## Pseudocode

1. Initialise policy $\pi(\cdot \mid s; \theta)$ and value function $V(s; \phi)$.

**For each iteration:**

2. Roll out $N$ steps with $\pi$ to obtain trajectory $(s_t, a_t, r_t, s_{t+1})$.
3. Compute GAE advantages $A_t$ and value targets $V^{\text{target}}_t$.

**For each mini-batch** (single epoch, sampled without replacement):

4. Maximise $\log \pi(a_t \mid s_t)\, A_t + \beta\, \mathcal{H}\bigl[\pi(\cdot \mid s_t)\bigr]$.
5. Minimise $\bigl(V(s_t; \phi) - V^{\text{target}}_t\bigr)^2$.
6. One backward pass and optimiser step on the summed actor + critic loss.

## Implementation in this template

| Resource | Path |
|----------|------|
| Algorithm | [`a2c.py`](a2c.py) |
| HPs | [`configs/algorithm/a2c.yaml`](../../../configs/algorithm/a2c.yaml) |
| Experiment | [`configs/experiment/a2c/gym.yaml`](../../../configs/experiment/a2c/gym.yaml) |

```shell
python src/train.py experiment=a2c/gym
```

### Mapping pseudocode → code

| Pseudocode step | Where in code |
|-----------------|---------------|
| Stochastic policy π | `actor_network()` → `ProbabilisticActor` + `TanhNormal` |
| Value V(s) | `value_network()` → `ValueOperator` |
| Rollout collection | Trainer `Collector` with `get_explore_policy()` |
| GAE advantages | `step()` → `adv_module(batch)` under `no_grad` |
| On-policy buffer | `TensorDictReplayBuffer` + `SamplerWithoutReplacement` |
| Actor + critic loss | `A2CLoss` → single `optimizer.step()` |

Reference implementation: [torchrl SOTA A2C MuJoCo](https://github.com/pytorch/rl/blob/main/sota-implementations/a2c/a2c_mujoco.py).

## Experimental results

**Live W&B table (canonical):** [LatentLab/torchrl-hydra-template — Table](https://wandb.ai/LatentLab/torchrl-hydra-template/table)

| Run | Environment | Config | Seed | Frames | Eval return | Notes |
|-----|-------------|--------|------|--------|-------------|-------|
| — | — | — | — | — | — | No finished runs tagged ``template`` yet — see [W&B table](https://wandb.ai/LatentLab/torchrl-hydra-template/table) |
