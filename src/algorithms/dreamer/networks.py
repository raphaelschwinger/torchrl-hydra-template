"""
This file is from the R2Dreamer Repository: https://github.com/NM512/r2dreamer
It is modified to integrate with our Hydra Pipeline and TorchRL
Changes are marked with Comments #!
"""

import math
import re
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn

import src.components.distributions as dists  #! R2Dreamer used bare `import distributions as dists`
from src.algorithms.dreamer.tools import weight_init_  #!


class LambdaLayer(nn.Module):
    """Wrap an arbitrary callable into an ``nn.Module``."""

    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BlockLinear(nn.Module):
    """Block-wise linear layer.

    Weight layout is chosen to cooperate with PyTorch's fan-in/fan-out
    calculation used by initializers.
    """

    def __init__(self, in_ch: int, out_ch: int, blocks: int, outscale: float = 1.0):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.blocks = int(blocks)
        self.outscale = float(outscale)

        # Store weight in a layout that works with torch's fan calculation.
        # (O/G, I/G, G)
        self.weight = nn.Parameter(
            torch.empty(
                self.out_ch // self.blocks, self.in_ch // self.blocks, self.blocks
            )
        )
        self.bias = nn.Parameter(torch.empty(self.out_ch))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (..., I)
        batch_shape = x.shape[:-1]
        # Reshape to expose block dimension.
        # (..., I) -> (..., G, I/G)
        x = x.view(*batch_shape, self.blocks, self.in_ch // self.blocks)

        # Block-wise multiplication
        # (..., G, I/G), (O/G, I/G, G) -> (..., G, O/G)
        x = torch.einsum("...gi,oig->...go", x, self.weight)
        # Merge block dimension back.
        # (..., G, O/G) -> (..., O)
        x = x.reshape(*batch_shape, self.out_ch)
        return x + self.bias


class Conv2dSamePad(nn.Conv2d):
    """A Conv2d layer that emulates TensorFlow's 'SAME' padding.

    For stride 1 with an odd kernel and no dilation (the only configuration
    used in this codebase) SAME padding equals the constant symmetric
    ``k // 2``, which is passed to cuDNN directly. The runtime ``F.pad``
    fallback below materialises a padded copy of the activations on every
    forward/backward, which is measurably slower.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._static_same = (
            all(s == 1 for s in self.stride)
            and all(k % 2 == 1 for k in self.kernel_size)
            and all(d == 1 for d in self.dilation)
        )
        if self._static_same:
            self.padding = tuple(k // 2 for k in self.kernel_size)

    def _calc_same_pad(self, i: int, k: int, s: int, d: int) -> int:
        i_div_s_ceil = (i + s - 1) // s
        return max((i_div_s_ceil - 1) * s + (k - 1) * d + 1 - i, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._static_same:
            return super().forward(x)
        ih, iw = x.size()[-2:]
        pad_h = self._calc_same_pad(
            ih, self.kernel_size[0], self.stride[0], self.dilation[0]
        )
        pad_w = self._calc_same_pad(
            iw, self.kernel_size[1], self.stride[1], self.dilation[1]
        )

        if pad_h > 0 or pad_w > 0:
            x = F.pad(
                x,
                [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2],
            )

        return F.conv2d(
            x,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class RMSNorm2D(nn.RMSNorm):
    """RMSNorm over channel-last format applied to 4D tensors."""

    def __init__(self, ch: int, eps: float = 1e-3, dtype=None):
        super().__init__(ch, eps=eps, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply RMSNorm over the channel dimension.
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class MultiEncoder(nn.Module):
    def __init__(
        self,
        config,
        shapes,
    ):
        super().__init__()
        excluded = ("is_first", "is_last", "is_terminal", "reward")
        shapes = {
            k: v
            for k, v in shapes.items()
            if k not in excluded and not k.startswith("log_")
        }
        self.cnn_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) == 3 and re.match(config.cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(config.mlp_keys, k)
        }
        print("Encoder CNN shapes:", self.cnn_shapes)
        print("Encoder MLP shapes:", self.mlp_shapes)

        self.out_dim = 0
        self.selectors = []
        self.encoders = []
        if self.cnn_shapes:
            # TorchRL stores images in channel-first (C, H, W); first dim is channels  #! R2Dreamer assumed channel-last (H, W, C): `input_ch = sum([v[-1] for v in ...])`
            input_ch = sum([v[0] for v in self.cnn_shapes.values()])  #!
            some_v = tuple(self.cnn_shapes.values())[0]  #!
            input_shape = (some_v[1], some_v[2], input_ch)  #!
            self.encoders.append(ConvEncoder(config.cnn, input_shape))
            self.selectors.append(  #! R2Dreamer concatenated on last dim (-1); channel-first needs dim=-3 (C axis)
                lambda obs: torch.cat([obs[k] for k in self.cnn_shapes], dim=-3)  #!
            )
            self.out_dim += self.encoders[-1].out_dim
        if self.mlp_shapes:
            inp_dim = sum([sum(v) for v in self.mlp_shapes.values()])
            self.encoders.append(MLP(config.mlp, inp_dim))
            self.selectors.append(
                lambda obs: torch.cat([obs[k] for k in self.mlp_shapes], -1)
            )
            self.out_dim += self.encoders[-1].out_dim
        self.encoders = nn.ModuleList(self.encoders)

        if len(self.encoders) > 1:
            self.fuser = lambda x: torch.cat(x, dim=-1)
        elif len(self.encoders) == 1:
            self.fuser = lambda x: x[0]
        else:
            raise NotImplementedError

        self.apply(weight_init_)

    def forward(self, obs):
        """Encode a dict of observations."""
        # dict of (B, T, *)
        return self.fuser(
            [enc(sel(obs)) for enc, sel in zip(self.encoders, self.selectors)]
        )


class MultiDecoder(nn.Module):
    def __init__(self, config, deter, flat_stoch, shapes):
        super().__init__()
        excluded = ("is_first", "is_last", "is_terminal")
        shapes = {k: v for k, v in shapes.items() if k not in excluded}
        self.cnn_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) == 3 and re.match(config.cnn_keys, k)
        }
        self.mlp_shapes = {
            k: v
            for k, v in shapes.items()
            if len(v) in (1, 2) and re.match(config.mlp_keys, k)
        }
        print("Decoder CNN shapes:", self.cnn_shapes)
        print("Decoder MLP shapes:", self.mlp_shapes)
        self.all_keys = list(self.mlp_shapes.keys()) + list(self.cnn_shapes.keys())

        # Unlike the encoder, each decoder is initialized independently.
        if self.cnn_shapes:
            # TorchRL shapes are (C, H, W); channels are first dim  #! R2Dreamer used channel-last (H,W,C): `shape = (sum(x[-1]...),) + some_shape[:-1]`
            some_shape = list(self.cnn_shapes.values())[0]  #!
            shape = (sum(x[0] for x in self.cnn_shapes.values()),) + some_shape[1:]  #!
            self._cnn = ConvDecoder(
                config.cnn,
                deter,
                flat_stoch,
                shape,
            )
            self._image_dist = partial(
                getattr(dists, str(config.cnn_dist.name)), **config.cnn_dist
            )
        if self.mlp_shapes:
            shape = (sum(sum(x) for x in self.mlp_shapes.values()),)
            config.mlp.shape = shape
            self._mlp = MLPHead(config.mlp, deter + flat_stoch)
            self._mlp_dist = partial(
                getattr(dists, str(config.mlp_dist.name)), **config.mlp_dist
            )

    def forward(self, stoch, deter):
        """Decode latent states into observation distributions."""
        # (B, T, S, K), (B, T, D)
        dists = {}
        if self.cnn_shapes:
            split_sizes = [
                v[0] for v in self.cnn_shapes.values()
            ]  #! R2Dreamer used v[-1] (channel-last C); channel-first C is dim 0
            # (B, T, C_sum, H, W) — channel-first  #! was (B, T, H, W, C_sum) in R2Dreamer
            outputs = self._cnn(stoch, deter)
            outputs = torch.split(
                outputs, split_sizes, -3
            )  #! was dim=-1 (channel-last); channel-first C is dim -3 for (B,T,C,H,W)
            dists.update(
                {
                    key: self._image_dist(output)
                    for key, output in zip(self.cnn_shapes.keys(), outputs)
                }
            )
        if self.mlp_shapes:
            split_sizes = [v[0] for v in self.mlp_shapes.values()]
            # (B, T, S*K + D)
            feat = torch.cat([stoch.reshape(*deter.shape[:-1], -1), deter], -1)
            outputs = self._mlp(feat)
            outputs = torch.split(outputs, split_sizes, -1)
            dists.update(
                {
                    key: self._mlp_dist(output)
                    for key, output in zip(self.mlp_shapes.keys(), outputs)
                }
            )
        return dists


class ConvEncoder(nn.Module):
    def __init__(self, config, input_shape):
        super().__init__()
        act = getattr(torch.nn, config.act)
        h, w, input_ch = input_shape
        self.depths = tuple(
            int(config.depth) * int(mult) for mult in list(config.mults)
        )
        self.kernel_size = int(config.kernel_size)
        in_dim = input_ch
        layers = []
        for i, depth in enumerate(self.depths):
            layers.append(
                Conv2dSamePad(
                    in_channels=in_dim,
                    out_channels=depth,
                    kernel_size=self.kernel_size,
                    stride=1,
                    bias=True,
                )
            )
            layers.append(nn.MaxPool2d(2, 2))
            if config.norm:
                layers.append(RMSNorm2D(depth, eps=1e-04, dtype=torch.bfloat16))
            layers.append(act())
            in_dim = depth
            h, w = h // 2, w // 2

        self.out_dim = self.depths[-1] * h * w
        self.layers = nn.Sequential(*layers)

    def forward(self, obs):
        """Encode image-like observations with a CNN."""
        # (B, T, C, H, W) — TorchRL channel-first format  #! R2Dreamer received (B,T,H,W,C) channel-last and permuted here
        obs = obs - 0.5
        # (B*T, C, H, W)
        x = obs.reshape(-1, *obs.shape[-3:])
        # already channel-first; no permute needed  #! R2Dreamer had `x = x.permute(0, 3, 1, 2)` here
        # (B*T, C_feat, H_feat, W_feat)
        x = self.layers(x)
        # (B*T, C_feat*H_feat*W_feat)
        x = x.reshape(x.shape[0], -1)
        # (B, T, C_feat*H_feat*W_feat)
        return x.reshape(*obs.shape[:-3], x.shape[-1])


class ConvDecoder(nn.Module):
    def __init__(self, config, deter, flat_stoch, shape=(3, 64, 64)):
        super().__init__()
        act = getattr(torch.nn, config.act)
        self._shape = shape
        self.depths = tuple(
            int(config.depth) * int(mult) for mult in list(config.mults)
        )
        factor = 2 ** (len(self.depths))
        minres = [int(x // factor) for x in shape[1:]]
        self.min_shape = (*minres, self.depths[-1])
        self.bspace = int(config.bspace)
        self.kernel_size = int(config.kernel_size)
        self.units = int(config.units)
        u, g = math.prod(self.min_shape), self.bspace
        self.sp0 = BlockLinear(deter, u, g)
        self.sp1 = nn.Sequential(
            nn.Linear(flat_stoch, 2 * self.units),
            nn.RMSNorm(2 * self.units, eps=1e-04, dtype=torch.bfloat16),
            act(),
        )
        self.sp2 = nn.Linear(2 * self.units, math.prod(self.min_shape))
        self.sp_norm = nn.Sequential(
            nn.RMSNorm(self.depths[-1], eps=1e-04, dtype=torch.bfloat16), act()
        )
        layers = []
        in_dim = self.depths[-1]
        for depth in reversed(self.depths[:-1]):
            layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
            layers.append(
                Conv2dSamePad(in_dim, depth, self.kernel_size, stride=1, bias=True)
            )
            layers.append(RMSNorm2D(depth, eps=1e-04, dtype=torch.bfloat16))
            layers.append(act())
            in_dim = depth
        layers.append(nn.Upsample(scale_factor=2, mode="nearest"))
        layers.append(
            Conv2dSamePad(in_dim, self._shape[0], self.kernel_size, stride=1, bias=True)
        )
        self.layers = nn.Sequential(*layers)
        self.apply(weight_init_)

    def forward(self, stoch, deter):
        """Decode latent states into images.

        Notes
        -----
        The decoder first constructs a low-resolution spatial feature map from
        the deterministic state (block-linear projection) and from the stochastic
        state (MLP projection), concats them, then upsamples back to the target
        resolution.
        """
        # (B, T, S, K), (B, T, D)
        B_T = deter.shape[:-1]
        # (B*T, D), (B*T, S*K)
        x0, x1 = deter.reshape(B_T.numel(), deter.shape[-1]), stoch.reshape(
            B_T.numel(), -1
        )

        # Spatial features from deterministic state
        # (H_feat, W_feat, C_feat)
        H_feat, W_feat, C_feat = self.min_shape
        # (B*T, H_feat*W_feat*C_feat)
        x0 = self.sp0(x0)
        # (B*T, G, H_feat, W_feat, C_feat/G)
        x0 = x0.reshape(-1, self.bspace, H_feat, W_feat, C_feat // self.bspace)
        # (B*T, H_feat, W_feat, C_feat)
        x0 = x0.permute(0, 2, 3, 1, 4).reshape(-1, H_feat, W_feat, C_feat)

        # Spatial features from stochastic state
        # (B*T, 2*U)
        x1 = self.sp1(x1)
        # (B*T, H_feat, W_feat, C_feat)
        x1 = self.sp2(x1).reshape(-1, H_feat, W_feat, C_feat)

        # Combine and upsample
        # (B*T, H_feat, W_feat, C_feat)
        x = self.sp_norm(x0 + x1)
        # (B*T, C_feat, H_feat, W_feat)
        x = x.permute(0, 3, 1, 2)
        x = self.layers(x)  # Upsamples to original H, W; output is (B*T, C, H, W)
        x = torch.sigmoid(
            x
        )  #! R2Dreamer had `x = x.permute(0, 2, 3, 1)` here to go back to channel-last
        # (B, T, C, H, W) — keep channel-first to match TorchRL format  #!
        return x.reshape(*B_T, *x.shape[1:])


class MLP(nn.Module):
    def __init__(
        self,
        config,
        inp_dim,
    ):
        super().__init__()
        act = getattr(torch.nn, config.act)
        self._symlog_inputs = bool(config.symlog_inputs)
        self._device = torch.device(config.device)
        self.layers = nn.Sequential()
        for i in range(config.layers):
            self.layers.add_module(
                f"{config.name}_linear{i}", nn.Linear(inp_dim, config.units, bias=True)
            )
            self.layers.add_module(
                f"{config.name}_norm{i}",
                nn.RMSNorm(config.units, eps=1e-04, dtype=torch.bfloat16),
            )
            self.layers.add_module(f"{config.name}_act{i}", act())
            inp_dim = config.units
        self.out_dim = config.units

    def forward(self, x):
        # (B, T, I)
        if self._symlog_inputs:
            x = dists.symlog(x)
        # (B, T, U)
        return self.layers(x)


class MLPHead(nn.Module):
    def __init__(self, config, inp_dim):
        super().__init__()
        self.mlp = MLP(config, inp_dim)
        self._dist_name = str(config.dist.name)
        self._outscale = float(config.outscale)
        self._dist = getattr(dists, str(config.dist.name))

        if self._dist_name == "bounded_normal":
            self.last = nn.Linear(self.mlp.out_dim, config.shape[0] * 2, bias=True)
            kwargs = {
                "min_std": float(config.dist.min_std),
                "max_std": float(config.dist.max_std),
            }
        elif self._dist_name == "onehot":
            self.last = nn.Linear(self.mlp.out_dim, config.shape[0], bias=True)
            kwargs = {"unimix_ratio": float(config.dist.unimix_ratio)}
        elif self._dist_name == "multi_onehot":
            self.last = nn.Linear(self.mlp.out_dim, sum(config.shape), bias=True)
            kwargs = {
                "unimix_ratio": float(config.dist.unimix_ratio),
                "shape": tuple(config.shape),
            }
        elif self._dist_name == "symexp_twohot":
            self.last = nn.Linear(self.mlp.out_dim, config.shape[0], bias=True)
            kwargs = {
                "device": torch.device(config.device),
                "bin_num": int(config.dist.bin_num),
            }
        elif self._dist_name in ("binary", "identity"):
            self.last = nn.Linear(self.mlp.out_dim, config.shape[0], bias=True)
            kwargs = {}
        else:
            raise NotImplementedError

        self._dist = partial(self._dist, **kwargs)

        self.mlp.apply(weight_init_)
        self.last.apply(weight_init_)
        # apply explicit output scaling.
        if self._outscale != 1.0:
            with torch.no_grad():
                self.last.weight.mul_(self._outscale)

    def forward(self, x):
        """Produce a distribution head."""
        # (B, T, F)
        return self._dist(self.last(self.mlp(x)))


class Projector(nn.Module):
    def __init__(self, in_ch1, in_ch2):
        super().__init__()
        self.w = nn.Linear(in_ch1, in_ch2, bias=False)
        self.apply(weight_init_)

    def forward(self, x):
        return self.w(x)


class ReturnEMA(nn.Module):
    """running mean and std"""

    def __init__(self, device, alpha=1e-2):
        super().__init__()
        self.device = device
        self.alpha = alpha
        self.range = torch.tensor([0.05, 0.95], device=device)
        self.register_buffer(
            "ema_vals", torch.zeros(2, dtype=torch.float32, device=self.device)
        )

    def __call__(self, x):
        x_quantile = torch.quantile(torch.flatten(x.detach()), self.range)
        # Using out-of-place update for torch.compile compatibility
        self.ema_vals.copy_(
            self.alpha * x_quantile.detach() + (1 - self.alpha) * self.ema_vals
        )
        scale = torch.clip(self.ema_vals[1] - self.ema_vals[0], min=1.0)
        offset = self.ema_vals[0]
        return offset.detach(), scale.detach()
