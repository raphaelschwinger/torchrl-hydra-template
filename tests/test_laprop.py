"""LaProp foreach path vs original per-param path: bitwise equivalence.

The foreach implementation batches the same ops in the same order, so the two
paths must produce bit-identical parameters and optimizer state. All model
parameters are float32 (bf16 is used as autocast compute dtype only, never for
parameter storage), so f32 is the only dtype exercised. Runs without gym/env.
"""

import copy

import torch

from src.components.optim.laprop import LaProp


def _make_params(seed: int):
    g = torch.Generator().manual_seed(seed)
    shapes = [(64, 32), (128,), (16, 8), (1024,)]
    return [torch.nn.Parameter(torch.randn(shape, generator=g)) for shape in shapes]


def _set_grads(params, seed: int):
    g = torch.Generator().manual_seed(seed)
    for p in params:
        p.grad = torch.randn(p.shape, generator=g)


def _run(opt, params, steps=5):
    for step in range(steps):
        # Vary lr per step like the LambdaLR warmup schedule does.
        lr = 4e-5 * (step + 1) / steps
        opt.param_groups[0]["lr"] = lr
        _set_grads(params, seed=100 + step)
        opt.step()


def _assert_equal(params_a, opt_a, params_b, opt_b):
    for pa, pb in zip(params_a, params_b):
        assert torch.equal(pa.data, pb.data), "parameter mismatch between paths"
        sa, sb = opt_a.state[pa], opt_b.state[pb]
        assert sa["step"] == sb["step"]
        assert sa["exp_avg_lr_1"] == sb["exp_avg_lr_1"]
        assert sa["exp_avg_lr_2"] == sb["exp_avg_lr_2"]
        assert torch.equal(sa["exp_avg"], sb["exp_avg"])
        assert torch.equal(sa["exp_avg_sq"], sb["exp_avg_sq"])


def test_foreach_matches_per_param():
    """opt.step() via the foreach path vs. calling _step_group_per_param directly."""
    params_a = _make_params(0)
    params_b = copy.deepcopy(params_a)
    opt_a = LaProp(params_a, lr=4e-5, betas=(0.9, 0.999), eps=1e-20)
    opt_b = LaProp(params_b, lr=4e-5, betas=(0.9, 0.999), eps=1e-20)

    for step in range(5):
        lr = 4e-5 * (step + 1) / 5
        opt_a.param_groups[0]["lr"] = lr
        opt_b.param_groups[0]["lr"] = lr
        _set_grads(params_a, seed=100 + step)
        _set_grads(params_b, seed=100 + step)
        opt_a.step()  # default: foreach
        opt_b._step_group_per_param(opt_b.param_groups[0])  # original path

    _assert_equal(params_a, opt_a, params_b, opt_b)


def test_foreach_flag_matches_disabled_flag():
    """The public foreach=True/False constructor switch (perf.foreach_laprop)."""
    params_a = _make_params(2)
    params_b = copy.deepcopy(params_a)
    opt_a = LaProp(params_a, lr=4e-5, foreach=True)
    opt_b = LaProp(params_b, lr=4e-5, foreach=False)

    _run(opt_a, params_a)
    _run(opt_b, params_b)

    _assert_equal(params_a, opt_a, params_b, opt_b)


def test_foreach_skips_gradless_params():
    params = _make_params(1)
    opt = LaProp(params, lr=4e-5)
    _set_grads(params, seed=7)
    params[1].grad = None  # this param must be skipped, state untouched
    before = params[1].data.clone()
    opt.step()
    assert torch.equal(params[1].data, before)
    assert len(opt.state[params[1]]) == 0
    # The stepped params advanced normally.
    assert opt.state[params[0]]["step"] == 1
