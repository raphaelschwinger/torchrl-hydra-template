import copy
import math

import torch
import torch.nn.functional as F
from torch import nn
from torch.amp import autocast

from src.components.ema import polyak_update

from .dreamerv3 import DreamerV3


class _DreamerProMixin:
    """Prototype-based auxiliary loss methods (DreamerPro).

    Mixed in alongside DreamerV3 so the prototype machinery lives in its own
    file without duplicating any base-class logic.
    """

    @torch.no_grad()
    def augment_data(self, data):
        # Double the batch so we get two augmented views: (B, T, ...) -> (2B, T, ...)
        data_aug = {k: torch.cat([v, v], dim=0) for k, v in data.items()}
        # Images are channel-first (B, T, C, H, W) in TorchRL — no permute needed
        data_aug["image"] = self.random_translate(
            data_aug["image"],
            self.aug_max_delta,
            same_across_time=self.aug_same_across_time,
            bilinear=self.aug_bilinear,
        )
        return data_aug

    @torch.no_grad()
    def ema_proj(self, data):
        """Project observations through the EMA (stop-gradient) encoder and obs_proj."""
        embed = self._ema_encoder(data)
        proj = self._ema_obs_proj(embed)
        return F.normalize(proj, p=2, dim=-1)

    @torch.no_grad()
    def ema_update(self):
        """Polyak-average the online encoder/obs_proj into their EMA counterparts.

        Also re-normalises prototype rows to unit length each call so that dot
        products with projected embeddings stay within [-1, 1].
        """
        # Re-normalise prototype rows to unit length
        prototypes = F.normalize(self._prototypes, p=2, dim=-1)
        self._prototypes.data.copy_(prototypes)
        if self._ema_updates % self.ema_update_every == 0:
            mix = self.ema_update_fraction if self._ema_updates > 0 else 1.0
            polyak_update(self.encoder.parameters(), self._ema_encoder.parameters(), mix)
            polyak_update(self.obs_proj.parameters(), self._ema_obs_proj.parameters(), mix)
        self._ema_updates += 1

    def sinkhorn(self, scores):
        """Sinkhorn-Knopp normalisation in log space.

        Notes
        -----
        Given a score matrix of shape (K, B*T), we iteratively normalise rows
        and columns in log space so that the resulting soft-assignment matrix Q
        is approximately doubly stochastic (each prototype and each sample
        receives equal total probability mass).  Working in log space avoids
        numerical underflow compared to dividing by row/column sums directly.
        """
        shape = scores.shape
        K = shape[0]
        log_Q = F.log_softmax(scores.reshape(-1) / self.sinkhorn_eps, dim=0).reshape(K, -1)
        N = log_Q.shape[1]
        for _ in range(self.sinkhorn_iters):
            # Normalise rows (each prototype gets mass 1/K)
            log_Q = log_Q - torch.logsumexp(log_Q, dim=1, keepdim=True) - math.log(K)
            # Normalise columns (each sample gets mass 1/N)
            log_Q = log_Q - torch.logsumexp(log_Q, dim=0, keepdim=True) - math.log(N)
        return torch.exp(log_Q + math.log(N)).reshape(shape)

    def proto_loss(self, post_stoch, post_deter, embed, ema_proj_out):
        """SwAV-style prototype loss with Sinkhorn-Knopp assignment.

        Computes:
          swav  — cross-view SwAV loss: obs projections assigned to EMA targets
          temp  — temporal prototype loss: feat projections assigned to EMA targets
          norm  — projection norm regularisation (pushes norms toward 1)

        post_stoch / post_deter are from the augmented (2B) rollout so each
        batch position i and i+B form a pair of views.
        """
        prototypes = F.normalize(self._prototypes, p=2, dim=-1)

        obs_proj = self.obs_proj(embed)
        obs_norm = torch.norm(obs_proj, dim=-1)
        obs_proj = F.normalize(obs_proj, p=2, dim=-1)

        B, T = obs_proj.shape[:2]
        # (B, T, P) -> (B*T, P) -> scores (B*T, K) -> (K, B, T)
        obs_scores = torch.matmul(obs_proj.reshape(B * T, -1), prototypes.T)
        obs_scores = obs_scores.reshape(B, T, -1).permute(2, 0, 1)[:, :, self.warm_up:]
        obs_logits = F.log_softmax(obs_scores / self.temperature, dim=0)
        obs_logits_1, obs_logits_2 = torch.chunk(obs_logits, 2, dim=1)

        ema_scores = torch.matmul(ema_proj_out.reshape(B * T, -1), prototypes.T)
        ema_scores = ema_scores.reshape(B, T, -1).permute(2, 0, 1)[:, :, self.warm_up:]
        ema_scores_1, ema_scores_2 = torch.chunk(ema_scores, 2, dim=1)

        with torch.no_grad():
            ema_targets_1 = self.sinkhorn(ema_scores_1)
            ema_targets_2 = self.sinkhorn(ema_scores_2)
        ema_targets = torch.cat([ema_targets_1, ema_targets_2], dim=1)

        feat = self.rssm.get_feat(post_stoch, post_deter)
        feat_proj = self.feat_proj(feat)
        feat_norm = torch.norm(feat_proj, dim=-1)
        feat_proj = F.normalize(feat_proj, p=2, dim=-1)
        # (B, T, P) -> (B*T, P) -> scores (B*T, K) -> (K, B, T)
        feat_scores = torch.matmul(feat_proj.reshape(B * T, -1), prototypes.T)
        feat_scores = feat_scores.reshape(B, T, -1).permute(2, 0, 1)[:, :, self.warm_up:]
        feat_logits = F.log_softmax(feat_scores / self.temperature, dim=0)

        swav_loss = -0.5 * torch.mean(
            torch.sum(ema_targets_2 * obs_logits_1, dim=0)
        ) - 0.5 * torch.mean(torch.sum(ema_targets_1 * obs_logits_2, dim=0))
        temp_loss = -torch.mean(torch.sum(ema_targets * feat_logits, dim=0))
        norm_loss = torch.mean((obs_norm - 1).square()) + torch.mean((feat_norm - 1).square())
        return {"swav": swav_loss, "temp": temp_loss, "norm": norm_loss}

    @torch.no_grad()
    def random_translate(self, x, max_delta, same_across_time=False, bilinear=False):
        """Random integer crop-and-paste augmentation via grid_sample.

        Pads each frame by `max_delta` pixels with replicate padding, then
        samples a random integer pixel shift in [0, 2*pad] per (B, T) entry.
        With same_across_time=True the same shift applies to every timestep in
        a sequence, preserving temporal consistency.
        """
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        pad = int(max_delta)
        # Pad with border replication so grid_sample doesn't see zeros at edges
        x_padded = F.pad(x_flat, (pad, pad, pad, pad), "replicate")
        h_padded, w_padded = H + 2 * pad, W + 2 * pad

        # Build a normalised sampling grid covering the original (H, W) window
        eps_h, eps_w = 1.0 / h_padded, 1.0 / w_padded
        arange_h = torch.linspace(-1 + eps_h, 1 - eps_h, h_padded, device=x.device, dtype=x.dtype)[:H]
        arange_w = torch.linspace(-1 + eps_w, 1 - eps_w, w_padded, device=x.device, dtype=x.dtype)[:W]
        base_grid = torch.cat(
            [arange_w.unsqueeze(0).expand(H, -1).unsqueeze(2),
             arange_h.unsqueeze(1).expand(-1, W).unsqueeze(2)],
            dim=2,
        ).unsqueeze(0).expand(B * T, -1, -1, -1)

        # Integer pixel shift converted to normalised [-1, 1] coordinates
        if same_across_time:
            shift = torch.randint(0, 2 * pad + 1, (B, 1, 1, 1, 2), device=x.device, dtype=x.dtype)
            shift = shift.expand(-1, T, -1, -1, -1).reshape(B * T, 1, 1, 2)
        else:
            shift = torch.randint(0, 2 * pad + 1, (B * T, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift = shift * 2.0 / torch.tensor([w_padded, h_padded], device=x.device, dtype=x.dtype)

        grid = base_grid + shift
        out = F.grid_sample(
            x_padded, grid,
            mode="bilinear" if bilinear else "nearest",
            padding_mode="zeros",
            align_corners=False,
        )
        return out.reshape(B, T, C, H, W)


class DreamerPro(_DreamerProMixin, DreamerV3):
    """DreamerPro: DreamerV3 with augmentation + EMA targets + Sinkhorn prototypes.

    Reconstruction is disabled by default (loss_scales.recon: 0).
    swav, temp, and norm losses are independently toggleable via loss_scales.
    """

    def _init_extra(self, config, shapes):
        super()._init_extra(config, shapes)  # builds decoder if recon > 0
        if (
            self._loss_scales.get("swav", 0) > 0
            or self._loss_scales.get("temp", 0) > 0
        ):
            dpc = config.dreamer_pro
            self.warm_up = int(dpc.warm_up)
            self.num_prototypes = int(dpc.num_prototypes)
            self.proto_dim = int(dpc.proto_dim)
            self.temperature = float(dpc.temperature)
            self.sinkhorn_eps = float(dpc.sinkhorn_eps)
            self.sinkhorn_iters = int(dpc.sinkhorn_iters)
            self.ema_update_every = int(dpc.ema_update_every)
            self.ema_update_fraction = float(dpc.ema_update_fraction)
            self.freeze_prototypes_iters = int(dpc.freeze_prototypes_iters)
            self.aug_max_delta = float(dpc.aug.max_delta)
            self.aug_same_across_time = bool(dpc.aug.same_across_time)
            self.aug_bilinear = bool(dpc.aug.bilinear)

            self._prototypes = nn.Parameter(
                torch.randn(self.num_prototypes, self.proto_dim)
            )
            self.obs_proj = nn.Linear(self.embed_size, self.proto_dim)
            self.feat_proj = nn.Linear(self.rssm.feat_size, self.proto_dim)
            self._ema_encoder = copy.deepcopy(self.encoder)
            self._ema_obs_proj = copy.deepcopy(self.obs_proj)
            for p in self._ema_encoder.parameters():
                p.requires_grad = False
            for p in self._ema_obs_proj.parameters():
                p.requires_grad = False
            self._ema_updates = 0

    def _build_module_dict(self) -> dict:
        mods = super()._build_module_dict()
        if hasattr(self, "_prototypes"):
            mods["prototypes"] = self._prototypes
            mods["obs_proj"] = self.obs_proj
            mods["feat_proj"] = self.feat_proj
            mods["ema_encoder"] = self._ema_encoder
            mods["ema_obs_proj"] = self._ema_obs_proj
        return mods

    def _compute_rep_losses(
        self, post_stoch, post_deter, feat, embed, data, initial, B, T
    ) -> dict:
        losses = super()._compute_rep_losses(
            post_stoch, post_deter, feat, embed, data, initial, B, T
        )
        if not hasattr(self, "_prototypes"):
            return losses

        # DreamerPro: augmentation + EMA targets + Sinkhorn assignment.
        with torch.no_grad():
            data_aug = self.augment_data(data)
            initial_aug = (
                # (B, ...) -> (2B, ...)
                torch.cat([initial[0], initial[0]], dim=0),
                torch.cat([initial[1], initial[1]], dim=0),
            )
            ema_proj_out = self.ema_proj(data_aug)

        embed_aug = self.encoder(data_aug)
        post_stoch_aug, post_deter_aug, _ = self.rssm.observe(
            embed_aug, data_aug["action"], initial_aug, data_aug["is_first"]
        )
        proto_losses = self.proto_loss(post_stoch_aug, post_deter_aug, embed_aug, ema_proj_out)

        # Only include losses whose scale is > 0
        for key in ("swav", "temp", "norm"):
            if self._loss_scales.get(key, 0) > 0:
                losses[key] = proto_losses[key]
        return losses

    def update(self, data, initial):
        """EMA-update prototypes and encoder before the gradient step."""
        if hasattr(self, "_prototypes"):
            self.ema_update()
        return super().update(data, initial)

    def _post_grad_hook(self):
        """Freeze prototype gradients during warm-up phase."""
        if hasattr(self, "_prototypes") and self._ema_updates < self.freeze_prototypes_iters:
            if self._prototypes.grad is not None:
                self._prototypes.grad.zero_()
