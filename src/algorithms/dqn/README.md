# Deep Q-Networks (DQN)

**Paper:** Mnih et al. (2015), [*Human-level control through deep reinforcement learning*](https://www.nature.com/articles/nature14236).

Off-policy value-based RL for **discrete** action spaces. A neural network approximates
action values $Q(s, a)$; the agent acts ε-greedy and learns from a replay buffer of past
transitions.

## Key ideas

- **Q-learning with function approximation.** Replace the tabular Q-table with a network
  $Q(s, a; \theta)$ and minimise the Bellman residual on sampled mini-batches.
- **Experience replay.** Store transitions $(s, a, r, s', \text{done})$ in a buffer and
  sample i.i.d. mini-batches. Breaks temporal correlation and stabilises learning.
- **Target network.** Bootstrap targets use a slowly updated copy $\theta_{\text{target}}$
  instead of the online weights, reducing moving-target instability. This template uses
  **hard updates** every `hard_update_freq` gradient steps (`HardUpdate`).
- **ε-greedy exploration.** With probability ε take a random action; otherwise
  $a = \arg\max_a Q(s, a; \theta)$. ε is linearly annealed over `annealing_frames`.
- **Warm-up phase.** Random (or uniformly random) collection for `init_random_frames`
  before any gradient step, so the replay buffer is non-empty.
- **Atari variant** (`experiment=dqn/ale`): pixel observations via `NatureDQN` CNN,
  frame stacking, reward clipping, and a separate eval environment without train-only
  transforms (`EndOfLifeTransform`, `SignTransform`, `VecNorm`).

## Pseudocode

1. Initialise replay buffer $\mathcal{D}$.
2. Initialise Q-network with random weights $\theta$.
3. Initialise target Q-network: $\theta_{\text{target}} \leftarrow \theta$.

**For each environment step:**

4. With probability $\varepsilon$, select a random action $a$; otherwise $a \leftarrow \arg\max_a Q(s, a; \theta)$.
5. Execute $a$, observe reward $r$ and next state $s'$.
6. Store $(s, a, r, s', \text{done})$ in $\mathcal{D}$.
7. Sample a mini-batch from $\mathcal{D}$.
8. Set $y \leftarrow r + \gamma (1 - \text{done}) \max_{a'} Q(s', a'; \theta_{\text{target}})$.
9. Update $\theta$ by minimising $\bigl(y - Q(s, a; \theta)\bigr)^2$.
10. Every $C$ gradient steps, set $\theta_{\text{target}} \leftarrow \theta$.

## Implementation in this template

| Resource | Path |
|----------|------|
| Algorithm | [`dqn.py`](dqn.py) |
| CartPole HPs | [`configs/algorithm/dqn.yaml`](../../../configs/algorithm/dqn.yaml) |
| Atari network | [`configs/algorithm/network/nature_dqn.yaml`](../../../configs/algorithm/network/nature_dqn.yaml) |
| CartPole experiment | [`configs/experiment/dqn/gym.yaml`](../../../configs/experiment/dqn/gym.yaml) |
| Pong experiment | [`configs/experiment/dqn/ale.yaml`](../../../configs/experiment/dqn/ale.yaml) |

```shell
python src/train.py experiment=dqn/gym
python src/train.py experiment=dqn/ale
python src/train.py experiment=dqn/ale environment.task=Breakout
```

### Mapping pseudocode → code

| Pseudocode step | Where in code |
|-----------------|---------------|
| Initialise Q-network | `setup()` → `network()` factory → `QValueActor` |
| Target network | `DQNLoss(delay_value=True)` + `HardUpdate` |
| ε-greedy action | `EGreedyModule` in `get_explore_policy()` |
| Store transitions | `step()` → `replay_buffer.extend(batch)` |
| Sample + Bellman loss | `replay_buffer.sample()` → `DQNLoss` → `optimizer.step()` |
| Hard target sync | `target_updater.step()` every grad step (interval `hard_update_freq`) |

Reference implementations: [torchrl SOTA DQN CartPole](https://github.com/pytorch/rl/blob/main/sota-implementations/dqn/dqn_cartpole.py), [DQN Atari](https://github.com/pytorch/rl/blob/main/sota-implementations/dqn/dqn_atari.py).

## Experimental results

**Live W&B table (canonical):** [LatentLab/torchrl-hydra-template — Table](https://wandb.ai/LatentLab/torchrl-hydra-template/table)

| Run | Environment | Config | Seed | Frames | Eval return | Notes |
|-----|-------------|--------|------|--------|-------------|-------|
| — | — | — | — | — | — | No finished runs tagged ``template`` yet — see [W&B table](https://wandb.ai/LatentLab/torchrl-hydra-template/table) |
