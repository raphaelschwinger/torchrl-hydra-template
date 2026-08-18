# Adapted from https://github.com/nicklashansen/tdmpc2 (tdmpc2/common/world_model.py
# and api_model_conversion from tdmpc2/common/layers.py), MIT license.
# Changes: single-task only (multi-task embeddings, action masks, termination head
# and rgb encoder removed); `cfg` replaced by explicit constructor arguments.
# Attribute names and the `init()` parameter aliasing are preserved verbatim so
# official TD-MPC2 checkpoints remain state-dict compatible.
"""TD-MPC2 implicit world model (single-task, state observations)."""
from __future__ import annotations

from copy import deepcopy

import torch
import torch.nn as nn
from tensordict import TensorDict
from tensordict.nn import TensorDictParams

from src.components import layers, math


class WorldModel(nn.Module):
    """Latent encoder, dynamics, reward, policy prior, and Q-ensemble."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        latent_dim: int = 512,
        mlp_dim: int = 512,
        enc_dim: int = 256,
        num_enc_layers: int = 2,
        num_q: int = 5,
        dropout: float = 0.01,
        simnorm_dim: int = 8,
        num_bins: int = 101,
        vmin: float = -10.0,
        vmax: float = 10.0,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        tau: float = 0.01,
    ) -> None:
        super().__init__()
        self.num_q = num_q
        self.num_bins, self.vmin, self.vmax = num_bins, vmin, vmax
        self.tau = tau
        # "state" key mirrors upstream's per-obs-type encoder dict (checkpoint compat).
        enc = layers.mlp(
            obs_dim, max(num_enc_layers - 1, 1) * [enc_dim], latent_dim,
            act=layers.SimNorm(simnorm_dim),
        )
        self._encoder = nn.ModuleDict({"state": enc})
        self._dynamics = layers.mlp(
            latent_dim + action_dim, 2 * [mlp_dim], latent_dim, act=layers.SimNorm(simnorm_dim)
        )
        self._reward = layers.mlp(latent_dim + action_dim, 2 * [mlp_dim], max(num_bins, 1))
        self._pi = layers.mlp(latent_dim, 2 * [mlp_dim], 2 * action_dim)
        self._Qs = layers.Ensemble(
            [
                layers.mlp(latent_dim + action_dim, 2 * [mlp_dim], max(num_bins, 1), dropout=dropout)
                for _ in range(num_q)
            ]
        )
        self.apply(layers.weight_init)
        layers.zero_([self._reward[-1].weight, self._Qs.params["2", "weight"]])

        self.register_buffer("log_std_min", torch.tensor(log_std_min))
        self.register_buffer("log_std_dif", torch.tensor(log_std_max) - self.log_std_min)
        self.init()

    def init(self) -> None:
        """(Re)create detached / target Q parameter views after construction or `.to()`."""
        self._detach_Qs_params = TensorDictParams(self._Qs.params.data, no_convert=True)
        self._target_Qs_params = TensorDictParams(self._Qs.params.data.clone(), no_convert=True)

        # Create modules on meta device so parameters are not duplicated ...
        with self._detach_Qs_params.data.to("meta").to_module(self._Qs.module):
            self._detach_Qs = deepcopy(self._Qs)
            self._target_Qs = deepcopy(self._Qs)

        # ... then alias the params in (avoids duplicated tensors in the state dict).
        delattr(self._detach_Qs, "params")
        self._detach_Qs.__dict__["params"] = self._detach_Qs_params
        delattr(self._target_Qs, "params")
        self._target_Qs.__dict__["params"] = self._target_Qs_params

    @property
    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.init()
        return self

    def train(self, mode: bool = True):
        """Keep target Q-networks in eval mode."""
        super().train(mode)
        self._target_Qs.train(False)
        return self

    def soft_update_target_Q(self) -> None:
        """Soft-update target Q-networks using Polyak averaging."""
        self._target_Qs_params.lerp_(self._detach_Qs_params, self.tau)

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        """Encode a state observation into its latent representation."""
        return self._encoder["state"](obs)

    def next(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Predict the next latent state given current latent state and action."""
        return self._dynamics(torch.cat([z, a], dim=-1))

    def reward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """Predict instantaneous (single-step) reward logits."""
        return self._reward(torch.cat([z, a], dim=-1))

    def pi(self, z: torch.Tensor) -> tuple[torch.Tensor, TensorDict]:
        """Sample an action from the Gaussian policy prior (tanh-squashed)."""
        mean, log_std = self._pi(z).chunk(2, dim=-1)
        log_std = math.log_std(log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mean)

        log_prob = math.gaussian_logprob(eps, log_std)
        scaled_log_prob = log_prob * eps.shape[-1]

        # Reparameterization trick
        action = mean + eps * log_std.exp()
        mean, action, log_prob = math.squash(mean, action, log_prob)

        entropy_scale = scaled_log_prob / (log_prob + 1e-8)
        info = TensorDict(
            {
                "mean": mean,
                "log_std": log_std,
                "action_prob": 1.0,
                "entropy": -log_prob,
                "scaled_entropy": -log_prob * entropy_scale,
            }
        )
        return action, info

    def Q(
        self,
        z: torch.Tensor,
        a: torch.Tensor,
        return_type: str = "min",
        target: bool = False,
        detach: bool = False,
    ) -> torch.Tensor:
        """Predict state-action value.

        ``return_type``: ``min``/``avg`` of two randomly subsampled Q-heads
        (decoded to scalars), or ``all`` raw logits. ``target`` uses the target
        ensemble; ``detach`` uses the gradient-detached view of the online ensemble.
        """
        assert return_type in {"min", "avg", "all"}

        z = torch.cat([z, a], dim=-1)
        if target:
            qnet = self._target_Qs
        elif detach:
            qnet = self._detach_Qs
        else:
            qnet = self._Qs
        out = qnet(z)

        if return_type == "all":
            return out

        qidx = torch.randperm(self.num_q, device=out.device)[:2]
        Q = math.two_hot_inv(out[qidx], self.num_bins, self.vmin, self.vmax)
        if return_type == "min":
            return Q.min(0).values
        return Q.sum(0) / 2


def _rename_ensemble_keys(source_state_dict: dict) -> dict:
    """Remap old-API ``_Qs.params.<int>`` / ``_target_Qs.params.<int>`` keys."""
    name_map = ["weight", "bias", "ln.weight", "ln.bias"]
    new_state_dict = dict()
    for key, val in list(source_state_dict.items()):
        if key.startswith("_Qs."):
            num = key[len("_Qs.params."):]
            new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
            del source_state_dict[key]
            new_state_dict["_Qs.params." + new_key] = val
            new_state_dict["_detach_Qs_params." + new_key] = val
        elif key.startswith("_target_Qs."):
            num = key[len("_target_Qs.params."):]
            new_key = str(int(num) // 4) + "." + name_map[int(num) % 4]
            del source_state_dict[key]
            new_state_dict["_target_Qs_params." + new_key] = val
    return new_state_dict


def api_model_conversion(target_state_dict: dict, source_state_dict: dict) -> dict:
    """Convert a checkpoint from the old TD-MPC2 API to the torch.compile-compatible API.

    Official checkpoints from tdmpc2.com predate the vmapped ensemble layout;
    this remaps their ``_Qs.params.<int>`` keys. Already-converted checkpoints
    pass through unchanged.
    """
    # check whether checkpoint is already in the new format
    if "_detach_Qs_params.0.weight" in source_state_dict:
        return source_state_dict

    new_state_dict = _rename_ensemble_keys(source_state_dict)

    # add batch_size and device from target_state_dict to new_state_dict
    for prefix in ("_Qs.", "_detach_Qs_", "_target_Qs_"):
        for key in ("__batch_size", "__device"):
            new_key = prefix + "params." + key
            new_state_dict[new_key] = target_state_dict[new_key]

    # check that every key in new_state_dict is in target_state_dict
    for key in new_state_dict.keys():
        assert key in target_state_dict, f"key {key} not in target_state_dict"
    # check that all Qs keys in target_state_dict are in new_state_dict
    for key in target_state_dict.keys():
        if "Qs" in key:
            assert key in new_state_dict, f"key {key} not in new_state_dict"
    # check that source_state_dict contains no Qs keys
    for key in source_state_dict.keys():
        assert "Qs" not in key, f"key {key} contains 'Qs'"

    # copy log_std_min and log_std_max from target_state_dict to new_state_dict
    new_state_dict["log_std_min"] = target_state_dict["log_std_min"]
    new_state_dict["log_std_dif"] = target_state_dict["log_std_dif"]

    # copy new_state_dict to source_state_dict
    source_state_dict.update(new_state_dict)

    return source_state_dict
