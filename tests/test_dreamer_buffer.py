"""Dreamer replay alignment: each training step must pair a_{t-1} with its own outcome."""

from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from src.algorithms.dreamer.buffer import Buffer

A = 3  # action dim


def _cfg(batch_length=2):
    return SimpleNamespace(
        device="cpu", storage_device="cpu", batch_size=1, batch_length=batch_length,
        max_size=100, num_envs=1, online=False, pin_memory=False,
    )


def _obs(i):
    return torch.full((3, 4, 4), i / 255.0)


def _batch(steps):
    """TorchRL rows. steps: (obs id, is_first, reward of a_t, terminated, next obs id)."""
    n = len(steps)
    return TensorDict(
        {
            "image": torch.stack([_obs(o) for o, *_ in steps]),
            "is_first": torch.tensor([[f] for _, f, *_ in steps]),
            # action a_t encodes the obs it was taken from
            "action": torch.tensor([[float(o)] * A for o, *_ in steps]),
            "stoch": torch.zeros(n, 2, 2),
            "deter": torch.zeros(n, 4),
            "next": TensorDict(
                {
                    "image": torch.stack([_obs(nx) for *_, nx in steps]),
                    "is_first": torch.zeros(n, 1, dtype=torch.bool),
                    "reward": torch.tensor([[r] for _, _, r, _, _ in steps]),
                    "terminated": torch.tensor([[t] for _, _, _, t, _ in steps]),
                    "truncated": torch.zeros(n, 1, dtype=torch.bool),
                    "done": torch.tensor([[t] for _, _, _, t, _ in steps]),
                },
                batch_size=(n,),
            ),
        },
        batch_size=(n,),
    )


def test_rows_carry_the_reward_that_arrived_with_the_observation():
    buf = Buffer(_cfg())
    # Episode: obs 0..3, a_k earns 10*(k+1), crash after a_3 into obs 4; reset to obs 5.
    buf.add_transition(_batch([(0, True, 10.0, False, 1), (1, False, 20.0, False, 2)]))
    buf.add_transition(_batch([(2, False, 30.0, False, 3), (3, False, 40.0, True, 4)]))
    buf.add_transition(_batch([(5, True, 50.0, False, 6), (6, False, 60.0, False, 7)]))
    td = buf._buffer[: len(buf._buffer)]

    assert td["image"][:, 0, 0, 0].tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert td["reward"].reshape(-1).tolist() == [0, 10, 20, 30, 40, 0, 50]
    assert td["is_terminal"].reshape(-1).tolist() == [False, False, False, False, True, False, False]
    assert td["is_last"].reshape(-1).tolist() == [False, False, False, False, True, False, False]
    assert td["is_first"].reshape(-1).tolist() == [True, False, False, False, False, True, False]
    # The terminal row has no action taken from it.
    assert td["action"][:, 0].tolist() == [0, 1, 2, 3, 0, 5, 6]


def test_sample_pairs_previous_action_with_its_reward():
    buf = Buffer(_cfg(batch_length=3))
    buf.add_transition(_batch([(0, True, 10.0, False, 1), (1, False, 20.0, False, 2)]))
    buf.add_transition(_batch([(2, False, 30.0, False, 3), (3, False, 40.0, True, 4)]))
    for _ in range(20):
        data, _, _ = buf.sample()
        obs = data["image"][0, :, 0, 0, 0].mul(255).round()
        for t, o in enumerate(obs.tolist()):
            assert data["action"][0, t, 0].item() == o - 1  # a_{t-1} was taken from obs o-1
            assert data["reward"][0, t, 0].item() == 10.0 * o  # reward a_{t-1} earned
        assert data["is_terminal"][0, :, 0].tolist() == [o == 4 for o in obs.tolist()]


def test_rejects_multiple_envs():
    cfg = _cfg()
    cfg.num_envs = 2
    with pytest.raises(NotImplementedError):
        Buffer(cfg)
