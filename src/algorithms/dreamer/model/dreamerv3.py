import copy
from collections import OrderedDict

import torch
from tensordict import TensorDict
from torch import nn
from torch.amp import autocast
from torch.optim.lr_scheduler import LambdaLR

import src.algorithms.dreamer.networks as networks
import src.algorithms.dreamer.rssm as rssm
import src.algorithms.dreamer.tools as tools
from src.components.optim import LaProp, clip_grad_agc_
from src.algorithms.dreamer.tools import to_f32


class DreamerV3(nn.Module):
    """DreamerV3 world model + actor-critic.

    Subclasses extend via four hooks called from __init__:
      _init_extra(config, shapes)           — build variant-specific modules
      _build_module_dict() -> dict          — expose them to the shared optimiser
      _compute_rep_losses(...) -> dict      — return variant-specific losses
      _post_grad_hook()                     — runs between backward and optim.step()
    """

    def __init__(self, config, obs_space, act_space):
        super().__init__()
        self.device = torch.device(config.device)
        self.act_entropy = float(config.act_entropy)
        self.kl_free = float(config.kl_free)
        self.imag_horizon = int(config.imag_horizon)
        self.horizon = int(config.horizon)
        self.lamb = float(config.lamb)
        self.return_ema = networks.ReturnEMA(device=self.device)
        self.act_dim = act_space.n if hasattr(act_space, "n") else sum(act_space.shape)
        self._loss_scales = dict(config.loss_scales)
        self._log_grads = bool(config.log_grads)

        if hasattr(obs_space, "spaces"):
            shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        else:
            shapes = {k: tuple(v.shape) for k, v in obs_space.items()}

        # === Core world model ===
        self.encoder = networks.MultiEncoder(config.encoder, shapes)
        self.embed_size = self.encoder.out_dim
        self.rssm = rssm.RSSM(config.rssm, self.embed_size, self.act_dim)
        self.reward = networks.MLPHead(config.reward, self.rssm.feat_size)
        self.cont = networks.MLPHead(config.cont, self.rssm.feat_size)

        config.actor.shape = (
            (act_space.n,) if hasattr(act_space, "n") else tuple(map(int, act_space.shape))
        )
        self.act_discrete = False
        if hasattr(act_space, "multi_discrete"):
            config.actor.dist = config.actor.dist.multi_disc
            self.act_discrete = True
            print("Using multi-discrete action space", flush=True)
        elif hasattr(act_space, "n"):
            config.actor.dist = config.actor.dist.disc
            self.act_discrete = True
            print("Using discrete action space", flush=True)
        else:
            config.actor.dist = config.actor.dist.cont
            print("Using continuous action space", flush=True)

        self.actor = networks.MLPHead(config.actor, self.rssm.feat_size)
        self.value = networks.MLPHead(config.critic, self.rssm.feat_size)
        self.slow_target_update = int(config.slow_target_update)
        self.slow_target_fraction = float(config.slow_target_fraction)
        self._slow_value = copy.deepcopy(self.value)
        for param in self._slow_value.parameters():
            param.requires_grad = False
        self._slow_value_updates = 0

        # Phase 1: subclass builds its extra components (decoder / projector / prototypes)
        self._init_extra(config, shapes)

        # Phase 2: collect everything into one shared optimiser
        modules = self._build_module_dict()
        for key, module in modules.items():
            if isinstance(module, nn.Parameter):
                print(f"{module.numel():>14,}: {key}", flush=True)
            else:
                print(f"{sum(p.numel() for p in module.parameters()):>14,}: {key}", flush=True)

        self._named_params = OrderedDict()
        for name, module in modules.items():
            if isinstance(module, nn.Parameter):
                self._named_params[name] = module
            else:
                for pname, param in module.named_parameters():
                    self._named_params[f"{name}.{pname}"] = param
        print(
            f"Optimizer has: {sum(p.numel() for p in self._named_params.values())} parameters.",
            flush=True,
        )

        def _agc(params):
            clip_grad_agc_(params, float(config.agc), float(config.pmin), foreach=True)

        self._agc = _agc
        self._optimizer = LaProp(
            self._named_params.values(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )

        def lr_lambda(step):
            if config.warmup:
                return min(1.0, (step + 1) / config.warmup)
            return 1.0

        self._scheduler = LambdaLR(self._optimizer, lr_lambda=lr_lambda)
        self.train()
        self.clone_and_freeze()
        # All shapes are static, so cuDNN conv-algorithm autotuning is safe.
        # TF32 speeds up the float32 ops that autocast keeps out of bfloat16.
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True
            torch.set_float32_matmul_precision("high")
        if config.compile:
            print("Compiling update function with torch.compile...", flush=True)
            self._cal_grad = torch.compile(self._cal_grad, mode="reduce-overhead")

    # ------------------------------------------------------------------
    # Subclass hooks
    # ------------------------------------------------------------------

    def _init_extra(self, config, shapes):
        """Build decoder when loss_scales.recon > 0."""
        if self._loss_scales.get("recon", 0) > 0:
            self.decoder = networks.MultiDecoder(
                config.decoder, self.rssm._deter, self.rssm.flat_stoch, shapes,
            )
            recon = self._loss_scales.pop("recon")
            self._loss_scales.update({k: recon for k in self.decoder.all_keys})

    def _build_module_dict(self) -> dict:
        """Return modules the shared optimiser should train."""
        mods = {
            "rssm": self.rssm,
            "actor": self.actor,
            "value": self.value,
            "reward": self.reward,
            "cont": self.cont,
            "encoder": self.encoder,
        }
        if hasattr(self, "decoder"):
            mods["decoder"] = self.decoder
        return mods

    def _compute_rep_losses(
        self, post_stoch, post_deter, feat, embed, data, initial, B, T
    ) -> dict:
        """Reconstruction loss when decoder is present."""
        if not hasattr(self, "decoder"):
            return {}
        return {
            key: torch.mean(-dist.log_prob(data[key]))
            for key, dist in self.decoder(post_stoch, post_deter).items()
        }

    def _post_grad_hook(self):
        """Called after backward(), before optimizer.step(). Override in subclasses."""
        pass

    # ------------------------------------------------------------------
    # Slow-target + frozen-copy management
    # ------------------------------------------------------------------

    def _update_slow_target(self):
        """Update slow-moving value target network."""
        if self._slow_value_updates % self.slow_target_update == 0:
            with torch.no_grad():
                mix = self.slow_target_fraction
                for v, s in zip(self.value.parameters(), self._slow_value.parameters()):
                    s.data.copy_(mix * v.data + (1 - mix) * s.data)
        self._slow_value_updates += 1

    def train(self, mode=True):
        super().train(mode)
        # slow_value should always be in eval mode
        self._slow_value.train(False)
        return self

    def clone_and_freeze(self):
        """Create no-grad inference copies of the live modules.

        NOTE: ``requires_grad`` affects whether a parameter is updated by the
        optimiser — it does *not* prevent gradients from flowing through the
        operations that use those parameters in the forward pass.  The frozen
        copies are used purely as constant targets inside ``_cal_grad`` so we
        can evaluate the old value / slow-target without accumulating gradients
        onto those weights.
        """
        for attr in ("encoder", "rssm", "reward", "cont", "actor", "value", "_slow_value"):
            src = getattr(self, attr)
            frozen = copy.deepcopy(src)
            for (_, p_src), (_, p_frz) in zip(
                src.named_parameters(), frozen.named_parameters()
            ):
                # Share underlying storage so the frozen copy tracks the live
                # weights for free — no extra VRAM, no explicit sync needed.
                p_frz.data = p_src.data
                p_frz.requires_grad_(False)
            # _slow_value → _frozen_slow_value
            dest_name = f"_frozen_{attr.lstrip('_')}"
            setattr(self, dest_name, frozen)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Re-establish shared memory after moving the model to a new device
        self.clone_and_freeze()
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def act(self, obs, state, eval=False):
        """Policy inference step.

        obs:   TensorDict of (B, *), keys include "image", "is_first"
        state: TensorDict with stoch (B, S, K), deter (B, D), prev_action (B, A)
        """
        p_obs = self.preprocess(obs)
        # (B, E)
        embed = self._frozen_encoder(p_obs)
        prev_stoch, prev_deter, prev_action = (
            state["stoch"], state["deter"], state["prev_action"],
        )
        # (B, S, K), (B, D)
        stoch, deter, _ = self._frozen_rssm.obs_step(
            prev_stoch, prev_deter, prev_action, embed, obs["is_first"]
        )
        # (B, F)
        feat = self._frozen_rssm.get_feat(stoch, deter)
        action_dist = self._frozen_actor(feat)
        # (B, A)
        action = action_dist.mode if eval else action_dist.rsample()
        return action, TensorDict(
            {"stoch": stoch, "deter": deter, "prev_action": action},
            batch_size=state.batch_size,
        )

    @torch.no_grad()
    def get_initial_state(self, B):
        stoch, deter = self.rssm.initial(B)
        action = torch.zeros(B, self.act_dim, dtype=torch.float32, device=self.device)
        return TensorDict(
            {"stoch": stoch, "deter": deter, "prev_action": action}, batch_size=(B,)
        )

    @torch.no_grad()
    def video_pred(self, data, initial):
        """Return a (B, T, C, H*3, W) video tile: truth / reconstruction / open-loop."""
        if not hasattr(self, "decoder"):
            raise NotImplementedError("video_pred requires loss_scales.recon > 0.")
        p_data = self.preprocess(data)
        B = min(p_data["action"].shape[0], 6)
        # (B, T, E)
        embed = self.encoder(p_data)
        post_stoch, post_deter, _ = self.rssm.observe(
            embed[:B, :5],
            p_data["action"][:B, :5],
            tuple(val[:B] for val in initial),
            p_data["is_first"][:B, :5],
        )
        recon = self.decoder(post_stoch, post_deter)["image"].mode()[:B]
        prior_stoch, prior_deter = self.rssm.imagine_with_action(
            post_stoch[:, -1], post_deter[:, -1], p_data["action"][:B, 5:],
        )
        openl = self.decoder(prior_stoch, prior_deter)["image"].mode()
        model = torch.cat([recon[:, :5], openl], 1)
        truth = p_data["image"][:B]
        error = (model - truth + 1.0) / 2.0
        return torch.cat([truth, model, error], 3)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def update(self, data: TensorDict, initial: tuple[torch.Tensor, torch.Tensor]):
        p_data = self.preprocess(data)
        self._update_slow_target()
        torch.compiler.cudagraph_mark_step_begin()
        with autocast(device_type=self.device.type, dtype=torch.bfloat16):
            (stoch, deter), mets = self._cal_grad(p_data, initial)
        self._post_grad_hook()
        if self._log_grads:
            old_params = [p.data.clone().detach() for p in self._named_params.values()]
            grads = [p.grad for p in self._named_params.values() if p.grad is not None]
            mets["opt/grad_norm"] = tools.compute_global_norm(grads)
            mets["opt/grad_rms"] = tools.compute_rms(grads)
        self._agc(self._named_params.values())
        self._optimizer.step()
        self._scheduler.step()
        self._optimizer.zero_grad(set_to_none=True)
        mets["opt/lr"] = self._scheduler.get_last_lr()[0]
        if self._log_grads:
            updates = [
                new - old
                for new, old in zip(self._named_params.values(), old_params)
            ]
            mets["opt/param_rms"] = tools.compute_rms(list(self._named_params.values()))
            mets["opt/update_rms"] = tools.compute_rms(updates)
        return (stoch.detach(), deter.detach()), mets

    def _cal_grad(self, data, initial):
        """Compute all losses and call .backward() for one batch.

        Notes
        -----
        1. World model: posterior RSSM rollout + KL (dyn + rep).
        2. Variant representation losses dispatched via _compute_rep_losses()
           (decoder recon, Barlow Twins, InfoNCE, SwAV prototypes …).
        3. Imagination rollout for actor-critic updates (policy + value losses).
        4. Replay-based value learning that keeps gradients flowing through the
           world model (repval loss).
        """
        # data: TensorDict (B, T, *), initial: (stoch (B, S, K), deter (B, D))
        losses = {}
        metrics = {}
        B, T = data.shape

        # === World model: posterior rollout + KL ===
        # (B, T, E)
        embed = self.encoder(data)
        # (B, T, S, K), (B, T, D), (B, T, S, K)
        post_stoch, post_deter, post_logit = self.rssm.observe(
            embed, data["action"], initial, data["is_first"]
        )
        # (B, T, S, K)
        _, prior_logit = self.rssm.prior(post_deter)
        dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
        losses["dyn"] = torch.mean(dyn_loss)
        losses["rep"] = torch.mean(rep_loss)

        # === Variant representation losses (decoder / BT / prototypes …) ===
        # (B, T, F)
        feat = self.rssm.get_feat(post_stoch, post_deter)
        losses.update(
            self._compute_rep_losses(post_stoch, post_deter, feat, embed, data, initial, B, T)
        )

        # === Reward + continuation ===
        losses["rew"] = torch.mean(
            -self.reward(feat).log_prob(to_f32(data["next", "reward"]))
        )
        cont = 1.0 - to_f32(data["next", "terminated"])
        losses["con"] = torch.mean(-self.cont(feat).log_prob(cont))

        metrics["dyn_entropy"] = torch.mean(self.rssm.get_dist(prior_logit).entropy())
        metrics["rep_entropy"] = torch.mean(self.rssm.get_dist(post_logit).entropy())

        # === Imagination rollout for actor-critic ===
        # (B*T, S, K), (B*T, D)
        start = (
            post_stoch.reshape(-1, *post_stoch.shape[2:]).detach(),
            post_deter.reshape(-1, *post_deter.shape[2:]).detach(),
        )
        # (B*T, T_imag, F), (B*T, T_imag, A)
        imag_feat, imag_action = self._imagine(start, self.imag_horizon + 1)
        imag_feat, imag_action = imag_feat.detach(), imag_action.detach()

        # (B*T, T_imag, 1)
        imag_reward = self._frozen_reward(imag_feat).mode()
        # (B*T, T_imag, 1)  probability of continuation
        imag_cont = self._frozen_cont(imag_feat).mean
        # (B*T, T_imag, 1)
        imag_value = self._frozen_value(imag_feat).mode()
        imag_slow_value = self._frozen_slow_value(imag_feat).mode()
        disc = 1 - 1 / self.horizon
        # (B*T, T_imag, 1)
        weight = torch.cumprod(imag_cont * disc, dim=1)
        # (B*T, T_imag-1, 1)
        ret = self._lambda_return(
            torch.zeros_like(imag_cont), 1 - imag_cont,
            imag_reward, imag_value, imag_value, disc, self.lamb,
        )
        ret_offset, ret_scale = self.return_ema(ret)
        # (B*T, T_imag-1, 1)
        adv = (ret - imag_value[:, :-1]) / ret_scale

        policy = self.actor(imag_feat)
        # (B*T, T_imag-1, 1)
        logpi = policy.log_prob(imag_action)[:, :-1].unsqueeze(-1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        losses["policy"] = torch.mean(
            weight[:, :-1].detach() * -(logpi * adv.detach() + self.act_entropy * entropy)
        )
        tar_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
        # One value forward reused for both log_probs (the dist just wraps logits).
        imag_value_dist = self.value(imag_feat)
        losses["value"] = torch.mean(
            weight[:, :-1].detach()
            * (
                -imag_value_dist.log_prob(tar_padded.detach())
                - imag_value_dist.log_prob(imag_slow_value.detach())
            )[:, :-1].unsqueeze(-1)
        )

        ret_normed = (ret - ret_offset) / ret_scale
        metrics["ret"] = torch.mean(ret_normed)
        metrics["ret_005"] = self.return_ema.ema_vals[0]
        metrics["ret_095"] = self.return_ema.ema_vals[1]
        metrics["adv"] = torch.mean(adv)
        metrics["adv_std"] = torch.std(adv)
        metrics["con"] = torch.mean(imag_cont)
        metrics["rew"] = torch.mean(imag_reward)
        metrics["val"] = torch.mean(imag_value)
        metrics["tar"] = torch.mean(ret)
        metrics["slowval"] = torch.mean(imag_slow_value)
        metrics["weight"] = torch.mean(weight)
        metrics["action_entropy"] = torch.mean(entropy)
        metrics.update(tools.tensorstats(imag_action, "action"))

        # === Replay-based value learning (gradients flow through world model) ===
        last = to_f32(data["next", "done"])
        term = to_f32(data["next", "terminated"])
        reward = to_f32(data["next", "reward"])
        feat = self.rssm.get_feat(post_stoch, post_deter)
        boot = ret[:, 0].reshape(B, T, 1)
        value = self._frozen_value(feat).mode()
        slow_value = self._frozen_slow_value(feat).mode()
        ret = self._lambda_return(last, term, reward, value, boot, disc, self.lamb)
        ret_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
        rep_value_dist = self.value(feat)
        losses["repval"] = torch.mean(
            (1.0 - last)[:, :-1]
            * (
                -rep_value_dist.log_prob(ret_padded.detach())
                - rep_value_dist.log_prob(slow_value.detach())
            )[:, :-1].unsqueeze(-1)
        )
        metrics.update(tools.tensorstats(ret, "ret_replay"))
        metrics.update(tools.tensorstats(value, "value_replay"))
        metrics.update(tools.tensorstats(slow_value, "slow_value_replay"))

        total_loss = sum(v * self._loss_scales[k] for k, v in losses.items())
        total_loss.backward()
        metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
        metrics["opt/loss"] = total_loss
        return (post_stoch, post_deter), metrics

    @torch.no_grad()
    def _imagine(self, start, imag_horizon):
        """Roll out the policy in latent space for imag_horizon steps.

        Returns (B*T, T_imag, F) features and (B*T, T_imag, A) actions.
        """
        # (B, S, K), (B, D)
        feats, actions = [], []
        stoch, deter = start
        for _ in range(imag_horizon):
            # (B, F)
            feat = self._frozen_rssm.get_feat(stoch, deter)
            # (B, A)
            action = self._frozen_actor(feat).rsample()
            feats.append(feat)
            actions.append(action)
            stoch, deter = self._frozen_rssm.img_step(stoch, deter, action)
        # (B, T_imag, F), (B, T_imag, A)
        return torch.stack(feats, dim=1), torch.stack(actions, dim=1)

    @torch.no_grad()
    def _lambda_return(self, last, term, reward, value, boot, disc, lamb):
        """Compute lambda-return targets.

        lamb=1 → discounted Monte-Carlo return.
        lamb=0 → fixed 1-step TD return.
        """
        assert last.shape == term.shape == reward.shape == value.shape == boot.shape
        live = (1 - to_f32(term))[:, 1:] * disc
        cont = (1 - to_f32(last))[:, 1:] * lamb
        interm = reward[:, 1:] + (1 - cont) * live * boot[:, 1:]
        out = [boot[:, -1]]
        for i in reversed(range(live.shape[1])):
            out.append(interm[:, i] + live[:, i] * cont[:, i] * out[-1])
        return torch.stack(list(reversed(out))[:-1], 1)

    @torch.no_grad()
    def preprocess(self, data):
        if "image" in data:
            data["image"] = to_f32(data["image"])
        return data
