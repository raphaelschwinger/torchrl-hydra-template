# Adapted from https://github.com/nicklashansen/tdmpc2 (tdmpc2/common/math.py),
# MIT license. Changes: `cfg` parameters replaced by explicit arguments;
# multi-task / termination helpers removed.
"""Math utilities for discrete regression and squashed Gaussian policies.

Implements symlog two-hot encoding (discrete regression over a fixed support,
as in DreamerV3 / TD-MPC2) and helpers for tanh-squashed Gaussian policies.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_ce(
    pred: torch.Tensor, target: torch.Tensor, num_bins: int, vmin: float, vmax: float
) -> torch.Tensor:
    """Cross entropy between predicted logits and soft two-hot targets."""
    pred = F.log_softmax(pred, dim=-1)
    target = two_hot(target, num_bins, vmin, vmax)
    return -(target * pred).sum(-1, keepdim=True)


def log_std(x: torch.Tensor, low: torch.Tensor, dif: torch.Tensor) -> torch.Tensor:
    """Map an unbounded tensor into ``[low, low + dif]`` via tanh."""
    return low + 0.5 * dif * (torch.tanh(x) + 1)


def gaussian_logprob(eps: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """Gaussian log probability of noise ``eps`` under ``N(0, exp(log_std))``."""
    residual = -0.5 * eps.pow(2) - log_std
    log_prob = residual - 0.9189385175704956  # 0.5 * log(2 * pi)
    return log_prob.sum(-1, keepdim=True)


def squash(mu, pi, log_pi):
    """Apply tanh squashing to mean/sample and correct the log-probability."""
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    squashed_pi = torch.log(F.relu(1 - pi.pow(2)) + 1e-6)
    log_pi = log_pi - squashed_pi.sum(-1, keepdim=True)
    return mu, pi, log_pi


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithm. Adapted from https://github.com/danijar/dreamerv3."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Symmetric exponential (inverse of :func:`symlog`)."""
    return torch.sign(x) * torch.expm1(torch.abs(x))


def two_hot(x: torch.Tensor, num_bins: int, vmin: float, vmax: float) -> torch.Tensor:
    """Convert a batch of scalars to soft two-hot targets for discrete regression."""
    if num_bins == 0:
        return x
    elif num_bins == 1:
        return symlog(x)
    bin_size = (vmax - vmin) / (num_bins - 1)
    x = torch.clamp(symlog(x), vmin, vmax).squeeze(1)
    bin_idx = torch.floor((x - vmin) / bin_size)
    bin_offset = ((x - vmin) / bin_size - bin_idx).unsqueeze(-1)
    soft_two_hot = torch.zeros(x.shape[0], num_bins, device=x.device, dtype=x.dtype)
    bin_idx = bin_idx.long()
    soft_two_hot = soft_two_hot.scatter(1, bin_idx.unsqueeze(1), 1 - bin_offset)
    soft_two_hot = soft_two_hot.scatter(1, (bin_idx.unsqueeze(1) + 1) % num_bins, bin_offset)
    return soft_two_hot


def two_hot_inv(x: torch.Tensor, num_bins: int, vmin: float, vmax: float) -> torch.Tensor:
    """Convert a batch of soft two-hot encoded vectors (logits) to scalars."""
    if num_bins == 0:
        return x
    elif num_bins == 1:
        return symexp(x)
    dreg_bins = torch.linspace(vmin, vmax, num_bins, device=x.device, dtype=x.dtype)
    x = F.softmax(x, dim=-1)
    x = torch.sum(x * dreg_bins, dim=-1, keepdim=True)
    return symexp(x)


def gumbel_softmax_sample(p: torch.Tensor, temperature: float = 1.0, dim: int = 0):
    """Sample an index from the Gumbel-Softmax distribution over probabilities ``p``."""
    logits = p.log()
    gumbels = (
        -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
        .exponential_()
        .log()
    )  # ~Gumbel(0,1)
    gumbels = (logits + gumbels) / temperature  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)
    return y_soft.argmax(-1)
