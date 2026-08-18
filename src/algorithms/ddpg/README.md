# Deep Deterministic Policy Gradient (DDPG)

**Paper:** Lillicrap et al. (2016), [*Continuous control with deep reinforcement learning*](https://arxiv.org/abs/1509.02971).

Off-policy **actor-critic** method for **continuous** action spaces. A deterministic actor
$\mu(s; \theta_\mu)$ proposes actions; a critic $Q(s, a; \theta_Q)$ estimates their
value. Learning uses a replay buffer and slowly moving target networks.

## Key ideas

- **Deterministic policy gradient.** The actor outputs a continuous action directly (no
  sampling from a stochastic policy at execution time for the greedy policy).
- **Actor–critic architecture.** The critic fits the Bellman target; the actor is updated
  to maximise $Q(s, \mu(s))$ w.r.t. actor parameters (policy gradient through the critic).
- **Experience replay.** Same motivation as DQN: decorrelate samples and reuse data.
- **Target networks.** Separate target actor and target critic stabilise bootstrap targets.
  This template uses **Polyak soft updates** (`SoftUpdate`, step size `tau`) rather than
  hard copies.
- **Exploration noise.** Actions during collection are $\mu(s) + \text{noise}$. Default
  is Ornstein–Uhlenbeck process (`OrnsteinUhlenbeckProcessModule`); requires
  `InitTracker` on the environment to reset noise at episode boundaries.
- **Action bounds.** Actor MLP output is passed through `TanhModule` and rescaled to the
  environment's action spec.

## Pseudocode

1. Initialise replay buffer $\mathcal{D}$.
2. Initialise online actor $\mu(s; \theta_\mu)$ and critic $Q(s, a; \theta_Q)$.
3. Initialise target networks: $\theta_{\mu,\text{target}} \leftarrow \theta_\mu$, $\theta_{Q,\text{target}} \leftarrow \theta_Q$.

**For each environment step:**

4. Select $a \leftarrow \mu(s; \theta_\mu) + \mathcal{N}$ (exploration noise).
5. Execute $a$, observe reward $r$ and next state $s'$.
6. Store $(s, a, r, s', \text{done})$ in $\mathcal{D}$.
7. Sample a mini-batch from $\mathcal{D}$.
8. Set $y \leftarrow r + \gamma (1 - \text{done})\, Q_{\text{target}}\bigl(s', \mu_{\text{target}}(s')\bigr)$.
9. Update critic $\theta_Q$ by minimising $\bigl(y - Q(s, a; \theta_Q)\bigr)^2$.
10. Update actor $\theta_\mu$ to maximise $Q\bigl(s, \mu(s; \theta_\mu); \theta_Q\bigr)$.
11. Polyak update: $\theta_{\text{target}} \leftarrow \tau\,\theta_{\text{online}} + (1 - \tau)\,\theta_{\text{target}}$.

## Implementation in this template

| Resource | Path |
|----------|------|
| Algorithm | [`ddpg.py`](ddpg.py) |
| HPs | [`configs/algorithm/ddpg.yaml`](../../../configs/algorithm/ddpg.yaml) |
| Experiment | [`configs/experiment/ddpg/gym.yaml`](../../../configs/experiment/ddpg/gym.yaml) |

```shell
python src/train.py experiment=ddpg/gym
```

### Mapping pseudocode → code

| Pseudocode step | Where in code |
|-----------------|---------------|
| Actor + tanh bounds | `setup()` → `actor_network()` → `TanhModule` |
| Critic Q(s, a) | `value_network()` → `ValueOperator` |
| Exploration noise | `exploration_noise()` → `OrnsteinUhlenbeckProcessModule` |
| Store transitions | `step()` → `replay_buffer.extend(batch)` |
| Critic + actor loss | `DDPGLoss` → combined backward → `group_optimizers` |
| Soft target update | `SoftUpdate(tau=...)` each grad step |

Reference implementation: [torchrl SOTA DDPG](https://github.com/pytorch/rl/blob/main/sota-implementations/ddpg/ddpg.py).

## Experimental results

**Live W&B table (canonical):** [LatentLab/torchrl-hydra-template — Table](https://wandb.ai/LatentLab/torchrl-hydra-template/table)

| Run | Environment | Config | Seed | Frames | Eval return | Notes |
|-----|-------------|--------|------|--------|-------------|-------|
| — | — | — | — | — | — | No finished runs tagged ``template`` yet — see [W&B table](https://wandb.ai/LatentLab/torchrl-hydra-template/table) |
