"""Unit tests for BBF's buffer and sample-time computations.

These run without an environment or ``ale_py`` — they exercise the riskiest,
torchrl-native piece of BBF (the ``PrioritizedSliceSampler`` window buffer) plus
the two pure functions used at sample time (n-step masking and the C51
categorical projection).
"""
from __future__ import annotations

import torch
from tensordict import TensorDict

from src.algorithms.bbf.bbf import (
    BBFAlgorithm,
    _masked_nstep_return,
    _project_distribution,
)


# ----------------------------------------------------------------------
# _masked_nstep_return
# ----------------------------------------------------------------------


def test_nstep_return_no_cut():
    # Rewards 1,2,3,4,5 with no cut; n=3, gamma=0.5.
    reward = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    cut = torch.zeros(1, 5, dtype=torch.bool)
    returns, bootstrap, alive = _masked_nstep_return(reward, cut, gamma=0.5, n=3)
    # 1 + 0.5*2 + 0.25*3 = 2.75
    assert torch.allclose(returns, torch.tensor([2.75]))
    assert bootstrap.item() == 1.0  # no cut within horizon -> bootstrap
    assert torch.allclose(alive[0], torch.ones(5))


def test_nstep_return_varies_with_n():
    reward = torch.ones(1, 11)
    cut = torch.zeros(1, 11, dtype=torch.bool)
    r3, _, _ = _masked_nstep_return(reward, cut, gamma=1.0, n=3)
    r10, _, _ = _masked_nstep_return(reward, cut, gamma=1.0, n=10)
    assert r3.item() == 3.0 and r10.item() == 10.0  # same data, different horizon


def test_nstep_return_masks_at_cut():
    # Cut at index 1: reward at the boundary (idx 1) counts, later rewards do not.
    reward = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])
    cut = torch.tensor([[False, True, False, False, False]])
    returns, bootstrap, alive = _masked_nstep_return(reward, cut, gamma=1.0, n=4)
    # r0 (alive) + r1 (boundary still counts) ; r2, r3 masked out
    assert returns.item() == 3.0
    assert bootstrap.item() == 0.0  # cut within horizon -> no bootstrap
    assert torch.allclose(alive[0], torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0]))


# ----------------------------------------------------------------------
# _project_distribution
# ----------------------------------------------------------------------


def test_projection_sums_to_one():
    support = torch.linspace(-10, 10, 51)
    next_dist = torch.softmax(torch.randn(8, 51), dim=-1)
    returns = torch.randn(8)
    bootstrap = torch.ones(8)
    proj = _project_distribution(next_dist, returns, bootstrap, support, gamma_n=0.9)
    assert proj.shape == (8, 51)
    assert torch.allclose(proj.sum(-1), torch.ones(8), atol=1e-5)


def test_projection_deterministic_return_lands_on_atom():
    # No bootstrap, a scalar return of exactly 0 -> all mass on the centre atom.
    support = torch.linspace(-10, 10, 51)  # atom 25 == 0.0
    next_dist = torch.softmax(torch.randn(1, 51), dim=-1)
    returns = torch.zeros(1)
    bootstrap = torch.zeros(1)
    proj = _project_distribution(next_dist, returns, bootstrap, support, gamma_n=0.9)
    assert proj[0].argmax().item() == 25
    assert torch.allclose(proj[0, 25], torch.tensor(1.0), atol=1e-5)


# ----------------------------------------------------------------------
# torchrl PrioritizedSliceSampler window buffer
# ----------------------------------------------------------------------


def _make_algo(prioritized: bool, capacity: int = 256) -> BBFAlgorithm:
    algo = BBFAlgorithm(
        device=torch.device("cpu"),
        replay_capacity=capacity,
        prioritized=prioritized,
        batch_size=8,
        max_update_horizon=3,
        spr_depth=2,
    )
    algo.window = max(algo.max_update_horizon, algo.spr_depth)  # = 3
    return algo


def _fill(buf, n: int, obs_shape=(4, 8, 8)) -> None:
    # pixels encode the global step index in every element so we can verify
    # window contiguity after sampling.
    pixels = (
        torch.arange(n).view(n, 1, 1, 1).expand(n, *obs_shape).to(torch.uint8).clone()
    )
    td = TensorDict(
        {
            "pixels": pixels,
            "action": torch.arange(n) % 3,
            "reward": torch.ones(n),
            "cut": torch.zeros(n, dtype=torch.bool),
            "traj": torch.zeros(n, dtype=torch.long),
        },
        batch_size=[n],
    )
    buf.extend(td)


def test_buffer_samples_contiguous_full_windows():
    algo = _make_algo(prioritized=True)
    buf = algo._make_replay_buffer(algo.window)
    _fill(buf, 40)
    sl = algo.window + 1  # 4
    sample = buf.sample(algo.batch_size * sl).reshape(algo.batch_size, sl)
    assert sample.batch_size == torch.Size([algo.batch_size, sl])
    # pixel value == global index; each row must be consecutive indices.
    idx = sample.get("pixels")[..., 0, 0, 0].long()  # (B, sl)
    diffs = idx[:, 1:] - idx[:, :-1]
    assert torch.all(diffs == 1), f"windows not contiguous:\n{idx}"
    # never crosses the write head (all indices < stored size).
    assert int(idx.max()) < 40


def test_buffer_uint8_roundtrip():
    algo = _make_algo(prioritized=True)
    buf = algo._make_replay_buffer(algo.window)
    _fill(buf, 40)
    sl = algo.window + 1
    sample = buf.sample(algo.batch_size * sl).reshape(algo.batch_size, sl)
    obs = sample.get("pixels").float() / 255.0
    assert obs.dtype == torch.float32 and 0.0 <= float(obs.min()) and float(obs.max()) <= 1.0


def test_buffer_prioritization_skews_sampling():
    torch.manual_seed(0)
    algo = _make_algo(prioritized=True)
    buf = algo._make_replay_buffer(algo.window)
    _fill(buf, 60)
    sl = algo.window + 1
    # Boost a handful of start indices, zero the rest.
    hot = torch.tensor([5, 6, 7, 8, 9])
    all_idx = torch.arange(60)
    buf.update_priority(all_idx, torch.full((60,), 1e-6))
    buf.update_priority(hot, torch.ones(len(hot)))
    counts = torch.zeros(60)
    for _ in range(50):
        sample = buf.sample(algo.batch_size * sl).reshape(algo.batch_size, sl)
        starts = sample.get("pixels")[:, 0, 0, 0, 0].long()
        counts.index_add_(0, starts, torch.ones(algo.batch_size))
    # Sampling must concentrate on windows overlapping the boosted region.
    hot_region = counts[3:10].sum()
    assert hot_region > 0.6 * counts.sum(), (hot_region, counts.sum())
    # IS weights present and normalised to <= 1.
    sample = buf.sample(algo.batch_size * sl).reshape(algo.batch_size, sl)
    w = sample.get("priority_weight")[:, 0]
    assert float(w.max()) <= 1.0 + 1e-6 and float(w.min()) > 0.0


def test_uniform_buffer_has_no_priority_weight():
    algo = _make_algo(prioritized=False)
    buf = algo._make_replay_buffer(algo.window)
    _fill(buf, 40)
    sl = algo.window + 1
    sample = buf.sample(algo.batch_size * sl).reshape(algo.batch_size, sl)
    assert "priority_weight" not in sample.keys()
