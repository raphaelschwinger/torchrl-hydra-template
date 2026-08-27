# DreamerV3

**Paper:** Hafner et al. (2025), [*Mastering Diverse Control Tasks through World Models*](https://www.nature.com/articles/s41586-025-08744-2).

Model-based RL that learns a **world model** in a compact latent space and derives both
actor and critic entirely from **imagined** rollouts. Scales across continuous and discrete control with a single hyperparameter set.

The core implementation is based on the [NM512/r2dreamer](https://github.com/NM512/r2dreamer)
codebase, adapted to the TorchRL + Hydra structure of this template.

This implementation also supports two decoder-free variants:
Morihira et al. (2026), [*R2-Dreamer: Redundancy-Reduced World Models without Decoders or Augmentation*](https://arxiv.org/abs/2603.18202) (Barlow Twins auxiliary loss) and
Deng et al. (2021), [*DreamerPro: Reconstruction-Free Model-Based Reinforcement Learning with Prototypical Representations*](https://arxiv.org/abs/2110.14565) (prototypical assignment via Sinkhorn-Knopp OT).
Select the variant via `algorithm=r2dreamer` or `algorithm=dreamerpro`.

## Key ideas

- **Recurrent State Space Model (RSSM).** A GRU-based world model maintains a hybrid
  latent state $(h_t, z_t)$: $h_t$ is a deterministic recurrent belief, $z_t$ is a
  discrete stochastic categorical variable. The prior predicts $z_t$ from $h_t$; the
  posterior refines $z_t$ using the current observation.
- **KL balancing.** The KL loss is split into a dynamics term (prior ↔ sg-posterior)
  and a representation term (posterior ↔ sg-prior), each weighted separately with a
  `kl_free` free-nats floor to avoid penalising early exploration.
- **Symlog + two-hot regression.** Reward and value targets are mapped through
  $\text{symlog}(x) = \text{sign}(x)\ln(|x|+1)$ and predicted as a categorical
  distribution over a fixed bucket grid, making critics robust to large reward scales
  without any normalisation.
- **λ-returns in imagination.** Actor-critic targets blend Monte-Carlo and TD via
  $\lambda$: $\lambda=1$ recovers full returns, $\lambda=0$ recovers 1-step TD.
- **Entropy-regularised actor.** An `act_entropy` bonus discourages premature
  commitment during imagination rollouts.

**Decoder-free variants** replace pixel reconstruction (`loss_scales.recon`) with
auxiliary self-supervised losses on the encoder latents:
**R2-Dreamer** uses a **Barlow Twins** cross-correlation loss between projected RSSM
features and encoder embeddings (no decoder, no data augmentation needed).
**DreamerPro** uses **SwAV-style prototypical assignment** with Sinkhorn-Knopp OT and
an EMA target encoder trained on randomly-translated observations.

## Pseudocode

1. Initialise RSSM, encoder, decoder/Barlow/prototype head, reward head,
   continuation head, actor, critic.
2. Initialise replay buffer $\mathcal{D}$ (ring buffer, stores raw transitions).

**For each environment step:**

3. Observe $o_t$; encode → posterior $z_t$; update recurrent state $h_t$.
4. Sample action $a_t \sim \pi(a \mid h_t, z_t)$ (actor with entropy bonus).
5. Store $(o_t, a_t, r_t, \text{done}_t)$ in $\mathcal{D}$.

**Every** $\lfloor B \cdot T / \rho \rfloor$ **environment steps** (train ratio $\rho=128$):

6. Sample a sequence batch $(B=16, T=64)$ from $\mathcal{D}$.
7. **World model update:** encode sequences → RSSM forward pass →
   compute KL + representation loss (reconstruction / Barlow / prototypical) + reward +
   continuation losses → update encoder + RSSM + heads.
8. **Actor-critic update:** unroll `imag_horizon=15` imagination steps from posterior
   states → compute λ-returns → update actor (entropy-regularised policy gradient) +
   critic (two-hot symexp regression).
9. Write posterior latents $(h_t, z_t)$ back to the replay buffer for future sampling.

## Implementation in this template

| Resource | Path |
|----------|------|
| Algorithm wrapper + policy | [`dreamer.py`](dreamer.py) |
| Base DreamerV3 model | [`model/dreamerv3.py`](model/dreamerv3.py) |
| R2Dreamer extension | [`model/r2dreamer.py`](model/r2dreamer.py) |
| DreamerPro extension | [`model/dreamerpro.py`](model/dreamerpro.py) |
| RSSM | [`rssm.py`](rssm.py) |
| Network modules | [`networks.py`](networks.py) |
| Sequence replay buffer | [`buffer.py`](buffer.py) |
| Utility functions | [`tools.py`](tools.py) |
| Algorithm config (DreamerV3) | [`configs/algorithm/dreamer.yaml`](../../../configs/algorithm/dreamer.yaml) |
| Algorithm config (R2-Dreamer) | [`configs/algorithm/r2dreamer.yaml`](../../../configs/algorithm/r2dreamer.yaml) |
| Algorithm config (DreamerPro) | [`configs/algorithm/dreamerpro.yaml`](../../../configs/algorithm/dreamerpro.yaml) |
| Model size presets | [`configs/algorithm/dreamer/`](../../../configs/algorithm/dreamer/) |
| Atari environment | [`configs/environment/atari100k.yaml`](../../../configs/environment/atari100k.yaml) |
| Experiment (DreamerV3 preprocessing lives here) | [`configs/experiment/dreamer/atari100k.yaml`](../../../configs/experiment/dreamer/atari100k.yaml) |

```shell
# Standard DreamerV3 (pixel reconstruction)
python src/train.py experiment=dreamer/atari100k

# Another game
python src/train.py experiment=dreamer/atari100k environment.task=Breakout

# R2-Dreamer (Barlow Twins, decoder-free)
python src/train.py experiment=dreamer/atari100k algorithm=r2dreamer

# DreamerPro (prototypical assignment, decoder-free)
python src/train.py experiment=dreamer/atari100k algorithm=dreamerpro
```

### Model size presets

Model width is configured via a separate Hydra config group that sets `model.*`
interpolation variables consumed by the RSSM, encoder, decoder, actor, and critic.

| Preset | `deter` | `hidden` / `units` | `depth` | `discrete` | ~Params |
|--------|---------|--------------------|---------|------------|---------|
| `12m`  | 2048    | 256                | 16      | 16         | ~12M    |
| `25m`  | 3072    | 512                | 32      | 32         | ~25M    |
| `50m`  | 4096    | 640                | 48      | 48         | ~50M    |
| `100m` | 6144    | 768                | 64      | 64         | ~100M   |
| `200m` | 8192    | 1024               | 64      | 64         | ~200M   |
| `400m` | 12288   | 1536               | 64      | 64         | ~400M   |

Experiments default to `200m`. Override with e.g. `algorithm/dreamer=50m`.

### Mapping pseudocode → code

| Pseudocode step | Where in code |
|-----------------|---------------|
| RSSM prior + posterior | [`rssm.py`](rssm.py) → `RSSM.forward()` |
| World model losses (DreamerV3) | [`model/dreamerv3.py`](model/dreamerv3.py) → `DreamerV3._cal_grad()` |
| Representation loss (R2Dreamer) | [`model/r2dreamer.py`](model/r2dreamer.py) → `R2Dreamer._compute_rep_losses()` |
| Representation loss (DreamerPro) | [`model/dreamerpro.py`](model/dreamerpro.py) → `DreamerPro._compute_rep_losses()` |
| Imagination rollout | [`model/dreamerv3.py`](model/dreamerv3.py) → `DreamerV3._imagine()` |
| λ-return computation | [`model/dreamerv3.py`](model/dreamerv3.py) → `DreamerV3._lambda_return()` |
| Latent write-back | [`buffer.py`](buffer.py) → `Buffer.update()` |
| Sequence sampling | [`buffer.py`](buffer.py) → `Buffer.sample()` |
| RSSM state across steps | [`dreamer.py`](dreamer.py) → `DreamerPolicy.forward()` |
| Proportional update cadence | [`dreamer.py`](dreamer.py) → `DreamerAlgorithm.step()` |

### TorchRL integration notes

Three design decisions required custom adaptation due to TorchRL conventions:

1. **Proportional collection cadence.** `SyncDataCollector` collects
   `frames_per_batch = ⌊B·T / ρ⌋` transitions per iteration (default: 8 with
   $B=16, T=64, \rho=128$) and feeds them to `Buffer.add_transition()`. Update
   frequency is governed by `train_ratio` inside `DreamerAlgorithm.step()`, not by
   the collector batch size — the algorithm fires one gradient update per batch
   regardless of how many frames were collected.

2. **`is_first` key.** TorchRL's `InitTracker` emits `is_init`; the RSSM needs
   `is_first` to zero its hidden state at episode boundaries. A `RenameTransform`
   in the environment pipeline bridges the gap.

3. **Cross-episode sequence sampling.** Dreamer trains on fixed-length sequences
   $(T=64)$ that intentionally cross episode boundaries. Masking TorchRL's `truncated`
   and `done` keys and injecting a per-environment `episode` key allows
   `SliceSampler(traj_key="episode", end_key=None)` to sample contiguous blocks
   without respect for episode ends, matching the original ring-buffer semantics.

4. **Online sampling** (`buffer_config.online`, default on; official DreamerV3
   `replay.online: True`, absent from R2Dreamer). Each fresh, non-overlapping
   $(T{+}1)$-step segment of experience is queued and served at the front of the
   next batch before uniform sampling fills the rest, so every collected
   transition is trained on exactly once as soon as it exists. Uniform
   `SliceSampler` slices near the write cursor cannot cover the newest steps,
   so without this queue fresh data is systematically under-sampled.

### Performance flags

`dreamer_config.perf` (see `configs/algorithm/dreamer.yaml`) gates speed
optimisations behind independent flags; the booleans all default `true` and
`amp` defaults to `bf16`:

```bash
python src/train.py experiment=dreamer/atari100k algorithm.dreamer_config.perf.amp=fp16
```

| flag | file | what it toggles | numerics | reusable for DQN/DDPG/A2C? |
|---|---|---|---|---|
| `static_pad` | `networks.py` `Conv2dSamePad` | native `padding=k//2` conv vs. r2dreamer's runtime `F.pad` + copy | identical | No|
| `dedup_value` | `model/dreamerv3.py` `_cal_grad` | the *target* value forward only: reuses the trainable forward's mode vs. r2dreamer's separate (redundant, same weights) `_frozen_value` forward. The value/repval losses always reuse a single trainable forward for both `log_prob` terms — r2dreamer never duplicated that call, so this part isn't flag-gated. | identical | No |
| `cudnn_benchmark` | `model/dreamerv3.py` | `torch.backends.cudnn.benchmark` (r2dreamer leaves it commented out at `train.py:17`, so `false` = its effective behaviour: the PyTorch default) | identical | Yes — a global PyTorch backend flag; would help any conv-heavy algorithm with static shapes |
| `tf32` | `model/dreamerv3.py` | `float32_matmul_precision="high"` for f32 ops outside the autocast region. **Not a change over r2dreamer** — it sets the same thing unconditionally at `train.py:18`, so `true` is parity and `false` runs *below* upstream | identical | Yes — same story as `cudnn_benchmark`|
| `amp` | `model/dreamerv3.py` `update` | mixed-precision scheme, not a boolean: `bf16` = bfloat16 autocast with no gradient scaling (official DreamerV3); `fp16` = float16 autocast + `GradScaler` (r2dreamer parity — fp16's 5-bit exponent underflows, so the scaling is required); `off` = full f32. A plain switch only because `RMSNormF32` keeps f32 norm weights and upcasts internally; norms constructed in bf16 would couple to it. Quote `"off"` in YAML — bare `off` is a YAML boolean; `perf.amp=off` on the CLI is fine. | differs | Partially — autocast/`GradScaler` is standard PyTorch AMP and would work for any algorithm's forward/backward |
| `foreach_laprop` | `src/components/optim/laprop.py` | batched `torch._foreach_*` optimiser step vs. r2dreamer's original per-parameter loop | identical on f32 params (verified by `tests/test_laprop.py`) | No|

One more speed knob lives outside `perf`, on the buffer it configures:
`buffer_config.pin_memory` (`buffer.py`) stages a sampled batch in page-locked
host memory so the CPU→GPU copy is async. Numerics-identical; `false` measures
what the transfer costs without it.

(Two places run f32 inside an otherwise bf16 model, both following r2dreamer
rather than official DreamerV: act() and RSSM Carry)

**Additionally one big speed improvement is using `max-autotune` for the torch compile type, which can be set in the dreamer config file.**

## Experimental results

**Evaluation protocol.** The Atari-100k experiment uses
`evaluation: atari100k_native` — the standard Atari-100k protocol (eval every
10k agent steps on 10 episodes, 100 episodes at the end, `canonical_source:
eval`) run on a fresh instance of this experiment's *own* env stack, since the
shared grayscale, frame-stacked `atari100k_eval` cannot serve DreamerV3's 64x64
RGB `image` observations. That stack sets `terminal_on_life_loss: false` with no
reward clipping, so its returns are true game scores either way; measuring from
rollouts is what makes `charts/episodic_return` mean the same thing here as it
does for BBF and Rainbow.

Both experiments set `evaluation.policy: explore`, so rollouts use the sampled
actor. Official DreamerV3 has no argmax path at all — it acts from the sampled
actor everywhere, and that is what its published scores measure. The argmax
policy remains reachable as `evaluation.policy: eval`, untested and prone to
looping in the deterministic ALE.

Note for recurrent policies generally: eval rollouts advance with
`env.step_mdp`, which preserves the RSSM state (`stoch` / `deter` /
`prev_action`) that `DreamerPolicy` keeps at the tensordict root. Advancing with
`td["next"]` instead drops it and the policy silently acts from a fresh latent
at every step — that bug scored 0.0 across 100 Jamesbond episodes against a
~250 training stream before it was fixed.

**Benchmarks.** The paper reports dm_control on two suites, and both are
implemented: `experiment=dreamer/atari100k` (pixels, 200M preset) and
`experiment=dreamer/dmc` (DMC **Proprio** — state observations, 12M preset,
`encoder/decoder mlp_keys: observation`, `cnn_keys: '$^'`). The proprio stack is
shared with `ppo/dmc` and `tdmpc2/dmc`, so the three are directly comparable on
a task. DMC Vision would additionally need `from_pixels` plumbed through
`_make_dmc_env` and a working MUJOCO_GL renderer.

**Video diagnostics.** `video/world_model` (truth / reconstruction / open-loop
tile) and `video/agent` (gameplay) are logged on their own `video/frame` axis at
`algorithm.world_model_video_log_every` / `agent_video_log_every` environment
frames; `0` disables either. Both are automatically skipped on stacks with no
image observation — on DMC Proprio the decoder has no CNN head, so there is
nothing to reconstruct and nothing to record.

The official DreamerV3 code trains 10 % past the Atari100k budget
(`run.steps: 1.1e5`) and relies on `summary_max_step` to cut the headline
number back to 100k agent steps. This template's evaluation stream is
already the canonical source (`atari100k_native`, inherited
`canonical_source: eval`), so the Atari100k experiment trains for exactly
100k agent steps (400k game frames) and reports `eval/final_return_mean`
straight off the last `evaluation.summary_window` (100) eval episodes — no
`summary_max_step` trimming needed.

Older runs in the table below predate this protocol (110k agent steps,
`eval/score_mean_last10pct`); `scripts/update_algo_results.py` still reads
that key but labels it `legacy` in the Notes column. Replace them with new
runs.

**Live W&B table (canonical):** [LatentLab/torchrl-hydra-template — Table](https://wandb.ai/LatentLab/torchrl-hydra-template/table)

| Run | Environment | Config | Seed | Frames | Eval return | Notes |
|-----|-------------|--------|------|--------|-------------|-------|
| [dreamer_Jamesbond_atari100k_200m_2026-08-03_12-04-31](https://wandb.ai/LatentLab/torchrl-hydra-template/runs/wwt2hl3y) | ALE/Jamesbond-v5 | `experiment=dreamer/atari100k` | 1 | 110,000 | 222.0 | — |
| [dreamer_Jamesbond_atari100k_200m_2026-08-03_13-40-08](https://wandb.ai/LatentLab/torchrl-hydra-template/runs/9o2v5b67) | ALE/Jamesbond-v5 | `experiment=dreamer/atari100k` | 2 | 110,000 | 195.0 | — |
| [dreamer_Jamesbond_atari100k_200m_2026-08-03_13-41-35](https://wandb.ai/LatentLab/torchrl-hydra-template/runs/omlhkgde) | ALE/Jamesbond-v5 | `experiment=dreamer/atari100k` | 3 | 110,000 | 168.5 | — |
| **Mean ± Std** | ALE/Jamesbond-v5 | `experiment=dreamer/atari100k` | — | — | **195.2 ± 21.8** | n=3 seeds |
| [dreamer_cheetah-run_2026-08-03_20-44-09](https://wandb.ai/LatentLab/torchrl-hydra-template/runs/7a2tmns6) | cheetah-run | `experiment=dreamer/dmc` | 1 | 1,000,000 | 687.0 | — |
| [dreamer_cheetah-run_2026-08-03_23-05-17](https://wandb.ai/LatentLab/torchrl-hydra-template/runs/qw6dmfpk) | cheetah-run | `experiment=dreamer/dmc` | 2 | 1,000,000 | 798.6 | — |
| [dreamer_cheetah-run_2026-08-03_23-49-29](https://wandb.ai/LatentLab/torchrl-hydra-template/runs/o556nwid) | cheetah-run | `experiment=dreamer/dmc` | 3 | 1,000,000 | 841.2 | — |
| **Mean ± Std** | cheetah-run | `experiment=dreamer/dmc` | — | — | **775.6 ± 65.0** | n=3 seeds |
