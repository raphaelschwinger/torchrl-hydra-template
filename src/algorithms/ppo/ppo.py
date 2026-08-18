"""Proximal Policy Optimization (PPO, clip variant).

Schulman et al. (2017), "Proximal Policy Optimization Algorithms."
https://arxiv.org/abs/1707.06347

Pseudocode:
    Initialise policy pi(.|s; theta) and value V(s; phi)
    For each iteration:
        Roll out N steps with pi_old -> (s_t, a_t, r_t, log pi_old(a_t|s_t))
        Compute GAE advantages A_t and value targets V^target_t (once)
        For each of K epochs, for each shuffled minibatch:
            r_t(theta) = pi(a_t|s_t) / pi_old(a_t|s_t)
            Maximise min(r_t A_t, clip(r_t, 1-eps, 1+eps) A_t) + beta H[pi]
            Minimise (V(s_t; phi) - V^target_t)^2  (optionally eps-clipped)
            One backward + step with summed (actor + critic) loss

Implementation follows cleanRL's ``ppo_continuous_action.py`` /
``ppo_atari.py`` and "The 37 Implementation Details of PPO"
(https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/),
built from TorchRL components (``ClipPPOLoss``, ``GAE``,
``ProbabilisticActor``, ``ActorValueOperator``). See the package README for
the trick-by-trick mapping and the (two) documented deviations.
"""
from __future__ import annotations

import functools
from typing import Callable

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.tensor_specs import CategoricalBox
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType
from torchrl.modules import (
    ActorValueOperator,
    IndependentNormal,
    ProbabilisticActor,
    ValueOperator,
)
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

from src.algorithms.base import BaseAlgorithm, CollectorConfig, TrainingState
from src.components.networks import make_mlp_value, make_normal_mlp_actor


class PPOAlgorithm(BaseAlgorithm):
    """PPO (clip) for continuous or discrete control.

    On-policy: each collected rollout is consumed in ``num_epochs`` epochs of
    shuffled mini-batch updates and discarded. No long-term replay buffer, no
    target networks, no warm-up phase.

    One class covers both benchmark setups; the network factories decide the
    architecture (``algorithm/policy=mlp_normal`` vs ``nature_cnn_categorical``):

    - **State inputs** (default): separate actor / critic MLPs on ``obs_key``.
      The actor outputs the mean of a Normal with a state-independent learned
      log-std (cleanRL's continuous-action policy). Discrete action spaces get
      a Categorical over logits instead.
    - **Pixel inputs**: pass ``common_network`` to share a trunk (Nature CNN)
      between an actor head and a critic head via ``ActorValueOperator``.

    Defaults are cleanRL's ``ppo_continuous_action.py`` (MuJoCo/DMC-style,
    1M frames). ``critic_coeff=0.25`` is cleanRL's ``0.5 * MSE * vf_coef(0.5)``
    expressed against TorchRL's un-halved ``l2`` critic loss.
    """

    def __init__(
        self,
        device: torch.device | None = None,
        *,
        # --- Design choices: factories injected as Callables ---------------
        # Factories are called as ``factory(in_shape, action_dim)`` in
        # ``setup``. Without ``common_network``, ``in_shape`` is the raw
        # observation shape; with it, actor/value factories build heads on the
        # trunk's feature vector instead.
        common_network: Callable[[tuple[int, ...], int], nn.Module] | None = None,
        actor_network: Callable[[tuple[int, ...], int], nn.Module] = functools.partial(
            make_normal_mlp_actor,
            num_cells=[64, 64],
            activation_class=nn.Tanh,
        ),
        value_network: Callable[[tuple[int, ...], int], nn.Module] = functools.partial(
            make_mlp_value,
            num_cells=[64, 64],
            activation_class=nn.Tanh,
        ),
        obs_key: str = "observation",
        # --- Optimisation --------------------------------------------------
        lr: float = 3e-4,
        adam_eps: float = 1e-5,
        anneal_lr: bool = True,
        anneal_frames: int = 1_000_000,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        num_epochs: int = 10,
        mini_batch_size: int = 64,
        clip_epsilon: float = 0.2,
        clip_value: bool = True,
        entropy_coeff: float = 0.0,
        critic_coeff: float = 0.25,
        loss_critic_type: str = "l2",
        normalize_advantage: bool = True,
        max_grad_norm: float = 0.5,
        target_kl: float | None = None,
        # --- Data collection ----------------------------------------------
        frames_per_batch: int = 2048,
        max_frames_per_traj: int = -1,
    ) -> None:
        super().__init__(device)
        self._make_common_network = common_network
        self._make_actor_network = actor_network
        self._make_value_network = value_network
        self.obs_key = obs_key
        self.lr = lr
        self.adam_eps = adam_eps
        self.anneal_lr = anneal_lr
        self.anneal_frames = anneal_frames
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.num_epochs = num_epochs
        self.mini_batch_size = mini_batch_size
        self.clip_epsilon = clip_epsilon
        self.clip_value = clip_value
        self.entropy_coeff = entropy_coeff
        self.critic_coeff = critic_coeff
        self.loss_critic_type = loss_critic_type
        self.normalize_advantage = normalize_advantage
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.frames_per_batch = frames_per_batch
        self.max_frames_per_traj = max_frames_per_traj
        self._collected_frames = 0

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self, make_env: Callable[[], EnvBase]) -> None:
        # Read env specs from a short-lived proof environment.
        proof_env = make_env()
        obs_shape = tuple(proof_env.observation_spec[self.obs_key].shape)
        action_spec = proof_env.action_spec

        # Discrete -> Categorical over logits; continuous -> Normal(loc, scale)
        # with a state-independent learned log-std (cleanRL's policy heads).
        if isinstance(action_spec.space, CategoricalBox):
            action_dim = int(action_spec.space.n)
            dist_class = torch.distributions.Categorical
            dist_kwargs: dict = {}
            dist_keys = ["logits"]
        else:
            action_dim = int(action_spec.shape[-1])
            dist_class = IndependentNormal
            dist_kwargs = {}
            dist_keys = ["loc", "scale"]

        # 1. Actor and critic, optionally sharing a trunk.
        if self._make_common_network is not None:
            # Shared trunk (e.g. Nature CNN): obs -> features, then one head
            # each for policy logits and state value (cleanRL's Atari agent).
            trunk = self._make_common_network(obs_shape, action_dim).to(self.device)
            with torch.no_grad():
                feat_dim = trunk(torch.zeros(1, *obs_shape, device=self.device)).shape[-1]
            common_module = TensorDictModule(
                trunk, in_keys=[self.obs_key], out_keys=["common_features"]
            )
            actor_module = TensorDictModule(
                self._make_actor_network((feat_dim,), action_dim).to(self.device),
                in_keys=["common_features"],
                out_keys=dist_keys,
            )
            value_module = ValueOperator(
                self._make_value_network((feat_dim,), action_dim).to(self.device),
                in_keys=["common_features"],
            )
        else:
            common_module = None
            actor_module = TensorDictModule(
                self._make_actor_network(obs_shape, action_dim).to(self.device),
                in_keys=[self.obs_key],
                out_keys=dist_keys,
            )
            value_module = ValueOperator(
                self._make_value_network(obs_shape, action_dim).to(self.device),
                in_keys=[self.obs_key],
            )

        policy_module = ProbabilisticActor(
            module=actor_module,
            spec=action_spec,
            in_keys=dist_keys,
            distribution_class=dist_class,
            distribution_kwargs=dist_kwargs,
            return_log_prob=True,  # pi_old log-probs, stored during collection
            default_interaction_type=ExplorationType.RANDOM,
        )

        if common_module is not None:
            operator = ActorValueOperator(common_module, policy_module, value_module)
            self.actor = operator.get_policy_operator()
            self.critic = operator.get_value_operator()
        else:
            self.actor = policy_module
            self.critic = value_module

        # 2. GAE advantage estimator. Applied once per rollout (under
        #    no_grad); its ``advantage`` / ``value_target`` / ``state_value``
        #    outputs stay fixed across the epoch loop, as in cleanRL.
        self.adv_module = GAE(
            gamma=self.gamma,
            lmbda=self.gae_lambda,
            value_network=self.critic,
            average_gae=False,
            device=self.device,
        )

        # 3. Clipped PPO loss. ``normalize_advantage`` standardises per
        #    minibatch; ``clip_value`` clips the value prediction around the
        #    rollout-time ``state_value`` with the same epsilon (cleanRL's
        #    ``clip_vloss``).
        self.loss_module = ClipPPOLoss(
            actor_network=self.actor,
            critic_network=self.critic,
            clip_epsilon=self.clip_epsilon,
            entropy_coeff=self.entropy_coeff,
            critic_coeff=self.critic_coeff,
            loss_critic_type=self.loss_critic_type,
            normalize_advantage=self.normalize_advantage,
            clip_value=self.clip_epsilon if self.clip_value else None,
        )

        # 4. On-policy mini-batch buffer: holds exactly one rollout; sampling
        #    without replacement reshuffles on every pass, so iterating it
        #    ``num_epochs`` times yields K epochs of fresh minibatches.
        self.data_buffer = TensorDictReplayBuffer(
            storage=LazyTensorStorage(self.frames_per_batch, device=self.device),
            sampler=SamplerWithoutReplacement(),
            batch_size=self.mini_batch_size,
        )

        # 5. Single Adam over actor + critic (cleanRL uses one optimizer).
        #    With a shared trunk the actor and critic views expose the same
        #    trunk parameters; dedupe so Adam updates each tensor once.
        params = list(dict.fromkeys(self.loss_module.parameters()))
        self.optimizer = torch.optim.Adam(params, lr=self.lr, eps=self.adam_eps)

    def get_collector_config(self) -> CollectorConfig:
        return CollectorConfig(
            frames_per_batch=self.frames_per_batch,
            init_random_frames=0,
            max_frames_per_traj=self.max_frames_per_traj,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def step(self, batch: TensorDict) -> dict[str, float]:
        """One collector iteration: GAE once -> K epochs of clipped updates."""
        # Linearly decay the lr to 0 over ``anneal_frames`` (based on frames
        # collected *before* this rollout, so the first update uses full lr).
        alpha = 1.0
        if self.anneal_lr:
            alpha = max(0.0, 1.0 - self._collected_frames / self.anneal_frames)
            for group in self.optimizer.param_groups:
                group["lr"] = self.lr * alpha
        self._collected_frames += batch.numel()

        # Compute GAE on the unflattened rollout (time must be the trailing
        # batch dim, e.g. [num_envs, T]) so multi-env batches stay correct.
        with torch.no_grad():
            batch = self.adv_module(batch)

        # Reset and load the on-policy buffer for this iteration only.
        self.data_buffer.empty()
        self.data_buffer.extend(batch.reshape(-1))

        sums: dict[str, float] = {}
        num_updates = 0
        kl = 0.0
        for _ in range(self.num_epochs):
            for mb in self.data_buffer:
                loss_td = self.loss_module(mb.to(self.device))
                loss = loss_td["loss_objective"] + loss_td["loss_critic"]
                if "loss_entropy" in loss_td.keys():
                    loss = loss + loss_td["loss_entropy"]

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.loss_module.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                num_updates += 1
                for key in (
                    "loss_objective",
                    "loss_critic",
                    "loss_entropy",
                    "entropy",
                    "clip_fraction",
                    "kl_approx",
                    "ESS",
                ):
                    if key in loss_td.keys():
                        sums[key] = sums.get(key, 0.0) + loss_td[key].detach().mean().item()
                kl = loss_td.get("kl_approx", torch.zeros(())).detach().mean().item()

            # Optional early stop when the policy drifts too far (cleanRL's
            # ``target_kl``; checked once per epoch).
            if self.target_kl is not None and kl > self.target_kl:
                break

        metrics = {f"train/{k}": v / num_updates for k, v in sums.items()}
        metrics["train/lr"] = self.lr * alpha
        return metrics

    # ------------------------------------------------------------------
    # Policy access
    # ------------------------------------------------------------------

    def get_policy(self) -> TensorDictModule:
        # The trainer wraps eval rollouts in ``set_exploration_type(MODE)``:
        # Categorical -> argmax, IndependentNormal -> mean (deterministic).
        return self.actor

    def get_explore_policy(self) -> TensorDictModule:
        # Default interaction type is RANDOM, so collection samples from pi.
        return self.actor

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _get_training_state(self) -> TrainingState:
        return TrainingState(
            step=0,
            policy_state_dict=self.loss_module.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict(),
            extra={"collected_frames": self._collected_frames},
        )

    def _load_training_state(self, state: TrainingState) -> None:
        self.loss_module.load_state_dict(state.policy_state_dict)
        self.optimizer.load_state_dict(state.optimizer_state_dict)
        if state.extra is not None:
            self._collected_frames = int(state.extra.get("collected_frames", 0))
