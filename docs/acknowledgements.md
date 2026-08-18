# Acknowledgements

This project builds on the ideas pioneered by
[lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) by
@ashleve and further refined in
[yet-another-lightning-hydra-template](https://github.com/gorodnitskiy/yet-another-lightning-hydra-template)
by @gorodnitskiy. Their work on combining structured Hydra configs with clean
training pipelines served as the foundation; this template adapts that philosophy
to the reinforcement learning setting with TorchRL.

The DQN reference implementation in `src/algorithms/dqn/dqn.py` is modelled on the
torchrl SOTA reference at
[`pytorch/rl/sota-implementations/dqn/dqn_cartpole.py`](https://github.com/pytorch/rl/blob/main/sota-implementations/dqn/dqn_cartpole.py).
The DDPG reference implementation in `src/algorithms/ddpg/ddpg.py` is modelled on the
torchrl SOTA reference at
[`pytorch/rl/sota-implementations/ddpg/ddpg.py`](https://github.com/pytorch/rl/blob/main/sota-implementations/ddpg/ddpg.py).
The A2C reference implementation in `src/algorithms/a2c/a2c.py` is modelled on the
torchrl SOTA reference at
[`pytorch/rl/sota-implementations/a2c/a2c_mujoco.py`](https://github.com/pytorch/rl/blob/main/sota-implementations/a2c/a2c_mujoco.py).
The PPO reference implementation in `src/algorithms/ppo/ppo.py` follows
[cleanRL's PPO](https://docs.cleanrl.dev/rl-algorithms/ppo/) and
[*The 37 Implementation Details of PPO*](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/),
cross-checked against the
[torchrl SOTA PPO references](https://github.com/pytorch/rl/tree/main/sota-implementations/ppo).
The TD-MPC2 implementation in `src/algorithms/tdmpc2/` is adapted from the official
implementation by Nicklas Hansen at
[nicklashansen/tdmpc2](https://github.com/nicklashansen/tdmpc2) (MIT license); it stays
state-dict compatible with the official checkpoints from
[tdmpc2.com/models](https://www.tdmpc2.com/models).
Shared building blocks live in `src/components/` with per-file attribution headers:
`math.py`, `layers.py` and `scale.py` are adapted from nicklashansen/tdmpc2 (MIT),
`distributions.py` from [NM512/r2dreamer](https://github.com/NM512/r2dreamer), and
`optim/laprop.py` from
[Z-T-WANG/LaProp-Optimizer](https://github.com/Z-T-WANG/LaProp-Optimizer) (MIT);
`ema.py` and `optim/agc.py` are template-native.
