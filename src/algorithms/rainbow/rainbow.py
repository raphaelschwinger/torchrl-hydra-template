"""Rainbow: Combining Improvements in Deep Reinforcement Learning.

Hessel et al. (2018), https://arxiv.org/abs/1710.02298

Rainbow combines six independent extensions of DQN (Mnih et al. 2015). This
class extends ``DQNAlgorithm`` and overrides only what those extensions
require; every override is commented with the paper that introduced it.
Each extension is a toggle (``dueling``, ``noisy``, ``double_dqn``,
``distributional``, ``prioritized``) so ablations stay a config change, not a
code change — set any of them to ``False`` to fall back to vanilla DQN
behaviour for that axis.

``configs/experiment/rainbow/atari100k.yaml`` configures this same class as Data-Efficient
Rainbow (van Hasselt et al. 2019), the Atari-100k preset: longer multi-step,
more frequent target updates, and the paper's smaller encoder
(``encoder_type="data_efficient"``).
"""
from __future__ import annotations

from typing import Callable, Literal

import torch
import torch.nn as nn
from tensordict.nn import TensorDictSequential
from torchrl.data import (
    LazyTensorStorage,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.envs import EnvBase
from torchrl.envs.transforms import MultiStepTransform
from torchrl.modules import (
    ConvNet,
    DistributionalQValueActor,
    DuelingCnnDQNet,
    EGreedyModule,
    MLP,
    NoisyLinear,
    QValueActor,
    reset_noise,
)
from torchrl.objectives import DistributionalDQNLoss, DQNLoss, HardUpdate

from src.algorithms.dqn.dqn import DQNAlgorithm

# Conv encoder shapes. "dqn" is the standard NatureDQN encoder (Mnih et al.
# 2015, matches src.components.networks.NatureDQN); "data_efficient" is the smaller
# 2-layer encoder from Data-Efficient Rainbow (van Hasselt et al. 2019),
# tuned for the 100k-frame Atari-100k budget.
_ENCODER_CNN_KWARGS: dict[str, dict] = {
    "dqn": {
        "num_cells": [32, 64, 64],
        "kernel_sizes": [8, 4, 3],
        "strides": [4, 2, 1],
        "activation_class": nn.ReLU,
    },
    "data_efficient": {
        "num_cells": [32, 64],
        "kernel_sizes": [5, 5],
        "strides": [5, 5],
        "activation_class": nn.ReLU,
    },
}


class RainbowAlgorithm(DQNAlgorithm):
    """DQN + double Q-learning + dueling + PER + multi-step + C51 + noisy nets."""

    def __init__(
        self,
        device: torch.device | None = None,
        *,
        obs_key: str = "pixels",
        lr: float = 1e-4,
        gamma: float = 0.99,
        batch_size: int = 32,
        max_grad_norm: float = 10.0,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        annealing_frames: int = 250_000,
        frames_per_batch: int = 4,
        init_random_frames: int = 20_000,
        max_frames_per_traj: int = -1,
        num_updates: int = 4,
        hard_update_freq: int = 8_000,
        replay_capacity: int = 1_000_000,
        encoder_type: Literal["dqn", "data_efficient"] = "dqn",
        hidden_dim: int = 512,
        # --- Wang et al. (2016), "Dueling Network Architectures for Deep RL" ---
        dueling: bool = True,
        # --- Fortunato et al. (2018), "Noisy Networks for Exploration" ---------
        noisy: bool = True,
        noisy_std: float = 0.1,
        eval_noise: bool = True,
        # --- van Hasselt et al. (2016), "Deep RL with Double Q-learning" -------
        # Only takes effect when `distributional=False`: `DistributionalDQNLoss`
        # always selects the next action with the online network and evaluates
        # it with the target network internally, so it is unconditionally
        # "double" regardless of this flag.
        double_dqn: bool = True,
        # --- Bellemare et al. (2017), "A Distributional Perspective on RL" -----
        distributional: bool = True,
        num_atoms: int = 51,
        v_min: float = -10.0,
        v_max: float = 10.0,
        # --- Schaul et al. (2016), "Prioritized Experience Replay" -------------
        prioritized: bool = True,
        prb_alpha: float = 0.5,
        prb_beta_start: float = 0.4,
        prb_beta_end: float = 1.0,
        prb_beta_frames: int = 100_000,
        prb_eps: float = 1e-6,
        # --- Multi-step returns (Sutton 1988; used in Rainbow) -----------------
        n_steps: int = 3,
    ) -> None:
        # DQNAlgorithm's `network`/`replay_buffer` factory defaults are stored
        # but never invoked: `setup()` below is a full override that builds
        # both directly, since Rainbow's architecture/buffer are intrinsically
        # coupled to the toggles above (TD-MPC2 precedent, see algorithm
        # README's "Documented deviations").
        super().__init__(
            device,
            obs_key=obs_key,
            lr=lr,
            gamma=gamma,
            batch_size=batch_size,
            max_grad_norm=max_grad_norm,
            eps_start=eps_start,
            eps_end=eps_end,
            annealing_frames=annealing_frames,
            frames_per_batch=frames_per_batch,
            init_random_frames=init_random_frames,
            max_frames_per_traj=max_frames_per_traj,
            num_updates=num_updates,
            hard_update_freq=hard_update_freq,
        )
        self.replay_capacity = replay_capacity
        self.encoder_type = encoder_type
        self.hidden_dim = hidden_dim
        self.dueling = dueling
        self.noisy = noisy
        self.noisy_std = noisy_std
        self.eval_noise = eval_noise
        self.double_dqn = double_dqn
        self.distributional = distributional
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max
        self.prioritized = prioritized
        self.prb_alpha = prb_alpha
        self.prb_beta_start = prb_beta_start
        self.prb_beta_end = prb_beta_end
        self.prb_beta_frames = prb_beta_frames
        self.prb_eps = prb_eps
        self.n_steps = n_steps

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, make_env: Callable[[], EnvBase]) -> None:
        proof_env = make_env()
        obs_shape = tuple(proof_env.observation_spec[self.obs_key].shape)
        action_spec = proof_env.action_spec
        num_actions = int(action_spec.space.n)
        proof_env.close()

        # 1. Q-network. Dueling (Wang et al. 2016) splits the head into a
        #    state-value and an action-advantage stream; noisy layers
        #    (Fortunato et al. 2018) replace the dense head only — the conv
        #    encoder stays plain, matching the paper. Distributional
        #    (Bellemare et al. 2017) reshapes the output to
        #    [*, num_atoms, num_actions] so raw Q-values become per-atom logits.
        layer_class = NoisyLinear if self.noisy else nn.Linear
        layer_kwargs = {"std_init": self.noisy_std} if self.noisy else None
        out_features = (self.num_atoms, num_actions) if self.distributional else num_actions
        out_features_value = (self.num_atoms, 1) if self.distributional else 1
        cnn_kwargs = dict(_ENCODER_CNN_KWARGS[self.encoder_type])

        if self.dueling:
            q_net = DuelingCnnDQNet(
                out_features=out_features,
                out_features_value=out_features_value,
                cnn_kwargs=cnn_kwargs,
                mlp_kwargs={
                    "num_cells": [self.hidden_dim],
                    "layer_class": layer_class,
                    "layer_kwargs": layer_kwargs,
                },
            )
        else:
            cnn = ConvNet(**cnn_kwargs)
            with torch.no_grad():
                cnn_out = cnn(torch.zeros(1, *obs_shape))
            mlp = MLP(
                in_features=cnn_out.shape[-1],
                out_features=out_features,
                num_cells=[self.hidden_dim],
                activation_class=nn.ReLU,
                layer_class=layer_class,
                layer_kwargs=layer_kwargs,
            )
            q_net = nn.Sequential(cnn, mlp)
        q_net = q_net.to(self.device)
        # DuelingCnnDQNet's advantage/value heads are LazyLinear internally
        # (their input size depends on the conv output, which isn't known
        # until a forward pass); materialize them now so the loss module's
        # functional parameter conversion below doesn't see uninitialized
        # parameters.
        with torch.no_grad():
            q_net(torch.zeros(1, *obs_shape, device=self.device))

        # 2. Actor wrapper.
        if self.distributional:
            support = torch.linspace(self.v_min, self.v_max, self.num_atoms, device=self.device)
            self.q_actor = DistributionalQValueActor(
                module=q_net,
                support=support,
                spec=action_spec,
                in_keys=[self.obs_key],
            ).to(self.device)
        else:
            self.q_actor = QValueActor(
                module=q_net,
                spec=action_spec,
                in_keys=[self.obs_key],
            ).to(self.device)

        # 3. Exploration. Noisy nets (Fortunato et al. 2018) replace
        #    epsilon-greedy entirely: `NoisyLinear` samples fresh weight noise
        #    only in `nn.Module.training` mode (auto-disabled by `.eval()`),
        #    so the same actor serves as both the explore and greedy policy.
        if self.noisy:
            self.greedy_module = None
            self.q_actor.train()
            self._explore_policy = self.q_actor
        else:
            self.greedy_module = EGreedyModule(
                spec=action_spec,
                eps_init=self.eps_start,
                eps_end=self.eps_end,
                annealing_num_steps=self.annealing_frames,
                device=self.device,
            )
            self._explore_policy = TensorDictSequential(self.q_actor, self.greedy_module)

        # 4. Replay buffer. Prioritized sampling (Schaul et al. 2016) biases
        #    sampling toward high-TD-error transitions; the importance-sampling
        #    exponent beta is annealed 0.4 -> 1.0 in step() below, following
        #    the paper. Multi-step returns (as used in Rainbow; n-step
        #    bootstrapping traces to Sutton 1988) are applied at write time via
        #    `MultiStepTransform`, which is unbiased by collector-batch
        #    boundaries (unlike the collector-side `MultiStep` postproc).
        storage = LazyTensorStorage(max_size=self.replay_capacity, device="cpu")
        transform = MultiStepTransform(n_steps=self.n_steps, gamma=self.gamma) if self.n_steps > 1 else None
        if self.prioritized:
            self.replay_buffer = TensorDictPrioritizedReplayBuffer(
                alpha=self.prb_alpha,
                beta=self.prb_beta_start,
                eps=self.prb_eps,
                storage=storage,
                transform=transform,
            )
        else:
            self.replay_buffer = TensorDictReplayBuffer(storage=storage, transform=transform)

        # 5. Loss. `DistributionalDQNLoss` computes the C51 categorical
        #    projection (Bellemare et al. 2017) and always uses double-DQN
        #    action selection internally. Otherwise plain `DQNLoss` with the
        #    `double_dqn` toggle (van Hasselt et al. 2016).
        if self.distributional:
            self.loss_module = DistributionalDQNLoss(
                self.q_actor, gamma=self.gamma, delay_value=True
            )
        else:
            self.loss_module = DQNLoss(
                value_network=self.q_actor,
                loss_function="l2",
                delay_value=True,
                double_dqn=self.double_dqn,
            )
            self.loss_module.make_value_estimator(gamma=self.gamma)
        self.loss_module = self.loss_module.to(self.device)
        self.target_updater = HardUpdate(
            self.loss_module, value_network_update_interval=self.hard_update_freq
        )
        self.optimizer = torch.optim.Adam(self.q_actor.parameters(), lr=self.lr)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def step(self, batch) -> dict[str, float]:
        batch = batch.reshape(-1)
        if self.greedy_module is not None:
            self.greedy_module.step(batch.numel())
        self.replay_buffer.extend(batch)
        self._collected_frames += batch.numel()

        if self._collected_frames < self.init_random_frames:
            return {"train/epsilon": float(self.greedy_module.eps) if self.greedy_module else 0.0}

        losses = torch.zeros(self.num_updates, device=self.device)
        for j in range(self.num_updates):
            if self.noisy:
                # Fortunato et al. (2018): resample noisy-layer noise once per
                # gradient step (the reference schedule), not once per action.
                self.q_actor.apply(reset_noise)

            sample = self.replay_buffer.sample(self.batch_size).to(self.device)
            # MultiStepTransform writes "steps_to_next_obs" with shape [B]
            # instead of [B, 1]; DQNLoss/DistributionalDQNLoss broadcast it
            # directly against [B, 1]-shaped reward/terminated, so a bare [B]
            # silently mis-broadcasts into [B, B]. Align the trailing dim.
            steps_key = "steps_to_next_obs"
            if steps_key in sample.keys() and sample.get(steps_key).dim() == 1:
                sample.set(steps_key, sample.get(steps_key).unsqueeze(-1))
            loss = self.loss_module(sample)["loss"]

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_actor.parameters(), self.max_grad_norm)
            self.optimizer.step()
            self.target_updater.step()

            if self.prioritized:
                # Schaul et al. (2016): re-prioritize sampled transitions from
                # the per-sample TD error the loss module wrote into `sample`.
                self.replay_buffer.update_tensordict_priority(sample)
                self._anneal_prb_beta()

            losses[j] = loss.detach()

        return {
            "train/q_loss": losses.mean().item(),
            "train/epsilon": float(self.greedy_module.eps) if self.greedy_module else 0.0,
        }

    def _anneal_prb_beta(self) -> None:
        """Linearly anneal the PER importance-sampling exponent (Schaul et al. 2016)."""
        if self.prb_beta_frames <= 0:
            return
        fraction = min(1.0, self._collected_frames / self.prb_beta_frames)
        self.replay_buffer.sampler.beta = (
            self.prb_beta_start + (self.prb_beta_end - self.prb_beta_start) * fraction
        )

    # ------------------------------------------------------------------
    # Policy access
    # ------------------------------------------------------------------

    def get_policy(self):
        if self.noisy:
            # Fortunato et al. (2018): official Rainbow evaluates with noise
            # still sampled by default (`eval_noise=True`); NoisyLinear reads
            # `nn.Module.training` to decide whether to sample or use the
            # mean weights. This mutates shared state, which periodic
            # evaluation would otherwise leak into training —
            # `BaseTrainer.evaluate()` snapshots and restores every algorithm
            # module's `.training` flag around the rollout.
            self.q_actor.train(mode=self.eval_noise)
        return self.q_actor
