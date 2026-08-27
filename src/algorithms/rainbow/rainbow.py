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

import math
from types import MethodType
from typing import Callable, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictSequential
from torchrl.data import (
    LazyTensorStorage,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.envs import EnvBase
from torchrl.envs.transforms import MultiStepTransform
from torchrl.envs.transforms.rb_transforms import _multi_step_func
from torchrl.modules import (
    ConvNet,
    DistributionalQValueActor,
    DuelingCnnDQNet,
    EGreedyModule,
    MLP,
    NoisyLinear,
    QValueActor,
)
from torchrl.objectives import DistributionalDQNLoss, DQNLoss, HardUpdate

from src.algorithms.dqn.dqn import DQNAlgorithm
from src.components.exploration import FixedEpsilonGreedy

# Conv encoder shapes. "dqn" follows the BBF/Dopamine Atari encoder with
# Flax/JAX-style SAME padding; "data_efficient" is the smaller 2-layer encoder
# from Data-Efficient Rainbow (van Hasselt et al. 2019), tuned for the 100k-frame
# Atari-100k budget.
_ENCODER_CNN_KWARGS: dict[str, dict] = {
    "dqn": {
        "same_padding": True,
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


class _FlattenFeatures(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() <= 1:
            return x
        return x.flatten(1)


def _pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        return value
    return (value, value)


class _SamePadConv2d(nn.Module):
    """Conv2d with Flax/JAX-style SAME padding for Atari encoders."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int],
    ) -> None:
        super().__init__()
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=0,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_h, in_w = x.shape[-2:]
        out_h = math.ceil(in_h / self.stride[0])
        out_w = math.ceil(in_w / self.stride[1])
        pad_h = max((out_h - 1) * self.stride[0] + self.kernel_size[0] - in_h, 0)
        pad_w = max((out_w - 1) * self.stride[1] + self.kernel_size[1] - in_w, 0)
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        if pad_h or pad_w:
            x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
        return self.conv(x)


class _SamePaddingConvNet(nn.Module):
    """Nature-DQN CNN with Flax/JAX SAME padding."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_cells: list[int],
        kernel_sizes: list[int],
        strides: list[int],
        activation_class: type[nn.Module] = nn.ReLU,
        activation_kwargs: dict | list[dict] | None = None,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        current_channels = in_channels
        for i, (out_channels, kernel_size, stride) in enumerate(
            zip(num_cells, kernel_sizes, strides)
        ):
            layers.append(
                _SamePadConv2d(
                    current_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                )
            )
            if isinstance(activation_kwargs, list):
                kwargs = activation_kwargs[i] if i < len(activation_kwargs) else {}
            else:
                kwargs = activation_kwargs or {}
            layers.append(activation_class(**kwargs))
            current_channels = out_channels
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_shape = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])
        out = self.layers(x).flatten(1)
        if batch_shape:
            return out.reshape(*batch_shape, out.shape[-1])
        return out.reshape(out.shape[-1])


class InclusiveDoneMultiStepTransform(MultiStepTransform):
    """TorchRL MultiStepTransform with n-step bootstrap terminals.

    TorchRL's transform keeps the original done keys and exposes a separate
    ``nonterminal`` key. DQN losses consume ``next.done`` though, so a terminal
    on the last reward inside the n-step target would otherwise still bootstrap.
    """

    def _inv_call(self, tensordict: TensorDict) -> TensorDict | None:
        if not self._validated:
            self._validate()

        total_cat = self._append_tensordict(tensordict)
        if total_cat.shape[-1] <= self.n_steps:
            return None

        out = _multi_step_func(
            total_cat,
            done_key=self.done_key,
            done_keys=self.done_keys,
            reward_keys=self.reward_keys,
            mask_key=self.mask_key,
            n_steps=self.n_steps,
            gamma=self.gamma,
        )
        out = out[..., : -self.n_steps]
        for done_key in self.done_keys:
            existing = out.get(("next", done_key), default=None)
            if existing is None:
                continue
            inclusive_done = self._inclusive_done(total_cat, done_key)[..., : -self.n_steps]
            value = inclusive_done
            while value.ndim < existing.ndim:
                value = value.unsqueeze(-1)
            out.set(("next", done_key), value.expand_as(existing).to(existing.dtype))
        return out

    def _inclusive_done(self, tensordict: TensorDict, done_key: str) -> torch.Tensor:
        done = tensordict.get(("next", done_key)).bool()
        if done.shape != tensordict.shape:
            if done.shape[-1] == 1 and done.shape[:-1] == tensordict.shape:
                done = done.squeeze(-1)
            else:
                done = done.reshape(tensordict.shape)
        padded = F.pad(done.to(torch.int8), (0, self.n_steps - 1), value=0)
        return padded.unfold(-1, self.n_steps, 1).bool().any(dim=-1)


class _CnnRainbowQNet(nn.Module):
    """Rainbow head for explicit layer classes that TorchRL cannot lazy-build."""

    def __init__(
        self,
        *,
        obs_shape: tuple[int, ...],
        cnn_kwargs: dict,
        same_padding: bool = False,
        num_actions: int,
        hidden_dim: int,
        distributional: bool,
        num_atoms: int,
        dueling: bool,
        layer_class: type[nn.Module],
        layer_kwargs: dict | None,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.num_atoms = num_atoms
        self.distributional = distributional
        self.dueling = dueling
        if same_padding:
            self.encoder = _SamePaddingConvNet(
                in_channels=obs_shape[0],
                **cnn_kwargs,
            )
        else:
            self.encoder = ConvNet(**cnn_kwargs)
        with torch.no_grad():
            latent = self.encoder(torch.zeros(1, *obs_shape))
        kwargs = layer_kwargs or {}
        self.projection = nn.Sequential(
            _FlattenFeatures(),
            layer_class(int(latent.flatten(1).shape[-1]), hidden_dim, **kwargs),
        )
        head_out = num_actions * num_atoms if distributional else num_actions
        self.advantage = layer_class(hidden_dim, head_out, **kwargs)
        self.value = None
        if dueling:
            value_out = num_atoms if distributional else 1
            self.value = layer_class(hidden_dim, value_out, **kwargs)

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.projection(self.encoder(pixels)))
        if self.distributional:
            adv = self.advantage(h).view(-1, self.num_actions, self.num_atoms)
            if self.dueling and self.value is not None:
                value = self.value(h).view(-1, 1, self.num_atoms)
                logits = value + adv - adv.mean(dim=1, keepdim=True)
            else:
                logits = adv
            return logits.transpose(1, 2)

        q_values = self.advantage(h)
        if self.dueling and self.value is not None:
            value = self.value(h)
            q_values = value + q_values - q_values.mean(dim=1, keepdim=True)
        return q_values


class RainbowAlgorithm(DQNAlgorithm):
    """DQN + double Q-learning + dueling + PER + multi-step + C51 + noisy nets."""

    def __init__(
        self,
        device: torch.device | None = None,
        *,
        obs_key: str = "pixels",
        lr: float = 1e-4,
        adam_eps: float = 1e-8,
        weight_decay: float = 0.0,
        gamma: float = 0.99,
        batch_size: int = 32,
        max_grad_norm: float = 10.0,
        eps_start: float = 1.0,
        eps_end: float = 0.01,
        eps_eval: float = 0.001,
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
        self.eps_eval = eps_eval
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
        self.adam_eps = adam_eps
        self.weight_decay = weight_decay

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, make_env: Callable[[], EnvBase]) -> None:
        proof_env = make_env()
        obs_shape = tuple(proof_env.observation_spec[self.obs_key].shape)
        action_spec = proof_env.action_spec
        self.action_spec = action_spec
        num_actions = int(action_spec.space.n)
        proof_env.close()

        # 1. Q-network. Dueling (Wang et al. 2016) splits the head into a
        #    state-value and an action-advantage stream; noisy layers
        #    (Fortunato et al. 2018) replace the dense head only — the conv
        #    encoder stays plain, matching the paper. Distributional
        #    (Bellemare et al. 2017) reshapes the output to
        #    [*, num_atoms, num_actions] so raw Q-values become per-atom logits.
        layer_class = NoisyLinear if self.noisy else nn.Linear
        layer_kwargs = (
            {
                "std_init": self.noisy_std,
            }
            if self.noisy
            else None
        )
        out_features = (self.num_atoms, num_actions) if self.distributional else num_actions
        out_features_value = (self.num_atoms, 1) if self.distributional else 1
        cnn_kwargs = dict(_ENCODER_CNN_KWARGS[self.encoder_type])
        same_padding = bool(cnn_kwargs.pop("same_padding", False))
        if same_padding:
            q_net = _CnnRainbowQNet(
                obs_shape=obs_shape,
                cnn_kwargs=cnn_kwargs,
                same_padding=same_padding,
                num_actions=num_actions,
                hidden_dim=self.hidden_dim,
                distributional=self.distributional,
                num_atoms=self.num_atoms,
                dueling=self.dueling,
                layer_class=layer_class,
                layer_kwargs=layer_kwargs,
            )
        elif self.dueling:
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
        if self.noisy:
            _sample_noisy_linear_on_forward(q_net)

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

        # 3. Exploration. DER keeps epsilon-greedy on top of NoisyNet.
        self.greedy_module = EGreedyModule(
            spec=action_spec,
            eps_init=self.eps_start,
            eps_end=self.eps_end,
            annealing_num_steps=self.annealing_frames,
            device=self.device,
        )
        if self.noisy:
            self.q_actor.train()
        self._explore_policy = TensorDictSequential(
            self.q_actor,
            self.greedy_module,
            _SqueezeUnbatchedActionModule(),
        )

        # 4. Replay buffer. Prioritized sampling (Schaul et al. 2016) biases
        #    sampling toward high-TD-error transitions; the importance-sampling
        #    exponent beta is annealed 0.4 -> 1.0 in step() below, following
        #    the paper. Multi-step returns (as used in Rainbow; n-step
        #    bootstrapping traces to Sutton 1988) are applied at write time via
        #    `MultiStepTransform`, which is unbiased by collector-batch
        #    boundaries (unlike the collector-side `MultiStep` postproc).
        storage = LazyTensorStorage(max_size=self.replay_capacity, device="cpu")
        transform = InclusiveDoneMultiStepTransform(n_steps=self.n_steps, gamma=self.gamma) if self.n_steps > 1 else None
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
        self.optimizer = self._make_optimizer()

    def _make_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.q_actor.parameters(),
            lr=self.lr,
            eps=self.adam_eps,
            weight_decay=self.weight_decay,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def step(self, batch) -> dict[str, float]:
        batch = batch.reshape(-1)
        _squeeze_policy_singletons(batch)
        _canonicalize_one_hot_action(batch, self.action_spec)
        self.replay_buffer.extend(batch)
        self._collected_frames += batch.numel()

        if self._collected_frames < self.init_random_frames:
            return {"train/epsilon": 1.0}
        self.greedy_module.step(batch.numel())

        losses = torch.zeros(self.num_updates, device=self.device)
        for j in range(self.num_updates):
            sample = self.replay_buffer.sample(self.batch_size).to(self.device)
            _canonicalize_one_hot_action(sample, self.action_spec)
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
        # NoisyLinear reads `nn.Module.training` to decide whether to sample
        # fresh weight noise or use the mean weights; `.eval()` also disables
        # dropout-like behaviour in any other submodule. This mutates shared
        # state, which periodic evaluation would otherwise leak into training
        # — `BaseTrainer.evaluate()` snapshots and restores every algorithm
        # module's `.training` flag around the rollout.
        self.q_actor.eval()
        return TensorDictSequential(
            self.q_actor,
            FixedEpsilonGreedy(self.action_spec, self.eps_eval),
            _SqueezePolicySingletonsModule(),
        )


def _sample_noisy_linear_on_forward(module: nn.Module) -> None:
    def forward_with_fresh_noise(layer: NoisyLinear, input: torch.Tensor) -> torch.Tensor:
        if not layer.training:
            return F.linear(input, layer.weight_mu, layer.bias_mu)
        epsilon_in = layer._scale_noise(layer.in_features)
        epsilon_out = layer._scale_noise(layer.out_features)
        weight = layer.weight_mu + layer.weight_sigma * epsilon_out.outer(epsilon_in)
        bias = None
        if layer.bias_mu is not None:
            bias = layer.bias_mu + layer.bias_sigma * epsilon_out
        return F.linear(input, weight, bias)

    for child in module.modules():
        if isinstance(child, NoisyLinear):
            child.forward = MethodType(forward_with_fresh_noise, child)


def _squeeze_policy_singletons(batch) -> None:
    """Keep collector output shapes stable before writing them to replay."""
    for key in ("action", "action_value"):
        value = batch.get(key, default=None)
        if value is not None and value.dim() > 2 and value.shape[-2] == 1:
            batch.set(key, value.squeeze(-2))


def _canonicalize_one_hot_action(batch, action_spec) -> None:
    action = batch.get("action", default=None)
    if action is None or action.dim() == 0:
        return
    spec_shape = tuple(getattr(action_spec, "shape", ()))
    if not spec_shape:
        return
    num_actions = int(spec_shape[-1])
    if num_actions <= 1 or action.shape[-1] != num_actions:
        return
    indices = action.argmax(dim=-1)
    canonical = F.one_hot(indices, num_classes=num_actions).to(dtype=action.dtype)
    batch.set("action", canonical)


class _SqueezePolicySingletonsModule(TensorDictModuleBase):
    """Normalize policy output shapes before the collector stacks them."""

    def __init__(self) -> None:
        self.in_keys = []
        self.out_keys = []
        super().__init__()

    def forward(self, tensordict: TensorDict) -> TensorDict:
        _squeeze_policy_singletons(tensordict)
        return tensordict


class _SqueezeUnbatchedActionModule(TensorDictModuleBase):
    """Match the unbatched action spec shape during collection."""

    def __init__(self) -> None:
        self.in_keys = []
        self.out_keys = []
        super().__init__()

    def forward(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict.get("action", default=None)
        if (
            action is not None
            and len(tensordict.batch_size) == 0
            and action.dim() > 1
            and action.shape[0] == 1
        ):
            tensordict.set("action", action.squeeze(0))
        return tensordict
