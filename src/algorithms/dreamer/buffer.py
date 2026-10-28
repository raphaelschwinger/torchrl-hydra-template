"""
This file is from the R2Dreamer Repository: https://github.com/NM512/r2dreamer
It is modified to integrate with our Hydra Pipeline and TorchRL
Changes are marked with Comments #!
"""

from collections import deque

import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler

#! Transition outputs under TorchRL's `next`; the other `next` keys are the
#! following observation.
_NEXT_KEYS = ("reward", "terminated", "truncated", "done")


class Buffer:
    def __init__(self, config):
        self.device = torch.device(config.device)
        self.storage_device = torch.device(config.storage_device)
        self.batch_size = int(config.batch_size)
        self.batch_length = int(config.batch_length)
        self.num_envs = int(getattr(config, "num_envs", 1))
        #! Terminal rows are inserted into the stream, which only keeps one
        #! contiguous stream per env with a single env.
        if self.num_envs != 1:
            raise NotImplementedError("Dreamer's replay buffer supports num_envs=1 only")
        #! Reward of the last stored TorchRL transition: it arrives with the
        #! first observation of the next collector batch.
        self._pending_reward = None
        #! `max_size` counts agent steps. _to_dreamer_rows adds one terminal row
        #! per episode on top, so storage gets 5% headroom for them: a run of
        #! max_size steps then fits without overwriting its oldest data.
        self.max_size = int(float(config.max_size) * 1.05)
        #! Online sampling (official DreamerV3 `replay.online: True`, absent from
        #! R2Dreamer): every fresh, non-overlapping (batch_length+1)-step segment
        #! is queued and served before uniform samples, so all new experience is
        #! trained on immediately at least once.
        self.online = bool(getattr(config, "online", True))
        #! Pinned (page-locked) staging for the CPU->GPU copy of a sampled batch.
        #! On by default (the optimisation ablation's combined setting), as
        #! r2dreamer pins whenever replay lives on the CPU.
        self.pin_memory = bool(getattr(config, "pin_memory", True))
        self._seq_len = self.batch_length + 1
        self._online_queue: deque[tuple[int, int]] = deque()  # (per-env start step, env)
        self._steps_per_env = 0        # per-env transitions added so far
        self._next_online_start = 0    # per-env step where the next online segment begins
        self._buffer = ReplayBuffer(
            #! Flat (ndim=1) storage: every transition occupies one slot in a 1-D
            # sequence.  The previous ndim=2 design stored each transition as a
            # 1-timestep trajectory (shape [N, 1, ...]), which made SliceSampler
            storage=LazyTensorStorage(
                max_size=self.max_size, device=self.storage_device
            ),
            #! traj_key="episode" groups all of env-i's steps into one long stream
            sampler=SliceSampler(
                num_slices=self.batch_size,
                traj_key="episode",
                end_key=None,
                truncated_key=None,
                strict_length=True,
            ),
            prefetch=0,
            batch_size=self.batch_size * (self.batch_length + 1),  # +1 for context
        )

    def _to_dreamer_rows(self, data):
        """TorchRL rows -> R2Dreamer rows.

        TorchRL row t: o_t, the action a_t taken from it, and under `next` the
        reward, terminated and done that a_t caused. R2Dreamer row t: o_t with
        the reward / is_terminal / is_last that arrived *with* o_t (0 / False on
        reset), and the action taken from it. sample() then shifts only the
        action, so every training step pairs a_{t-1} with its own outcome.
        The terminal observation, `("next", obs)` of a done row, becomes its own
        row, as R2Dreamer stores it.
        """
        nxt = data.get("next")
        reward = nxt["reward"].float()
        done = nxt["done"].bool()
        first = data["is_first"].bool().reshape(reward.shape)
        pending = (
            torch.zeros_like(reward[:1])
            if self._pending_reward is None
            else self._pending_reward.to(reward.device)
        )
        arrived = torch.cat([pending, reward[:-1]], 0)
        rows = data.exclude("next")
        rows["reward"] = torch.where(first, torch.zeros_like(arrived), arrived)
        rows["is_terminal"] = torch.zeros_like(first)
        rows["is_last"] = torch.zeros_like(first)
        self._pending_reward = reward[-1:].clone()

        done_idx = done.reshape(-1).nonzero().reshape(-1).tolist()
        if not done_idx:
            return rows
        term = rows[done_idx].clone()
        for key in nxt.keys():
            if key not in _NEXT_KEYS and key in term.keys():
                term[key] = nxt[key][done_idx]
        term["action"] = torch.zeros_like(term["action"])
        term["is_first"] = torch.zeros_like(term["is_first"])
        term["reward"] = reward[done_idx]
        term["is_terminal"] = nxt["terminated"].bool()[done_idx].reshape(term["is_last"].shape)
        term["is_last"] = torch.ones_like(term["is_last"])
        # Interleave: each terminal row directly after the row that ended.
        pieces, start = [], 0
        for j, k in enumerate(done_idx):
            pieces += [rows[start : k + 1], term[j : j + 1]]
            start = k + 1
        pieces.append(rows[start:])
        return torch.cat(pieces, 0)

    def add_transition(self, data):
        #! R2Dreamer received single frames (B=num_envs); now frames_per_batch may be > num_envs
        data = self._to_dreamer_rows(data.copy())  #! copy to avoid mutating the collector's TensorDict
        n_rows = data.batch_size[0]
        #! Store image as uint8 to save 4x memory
        #! Converted back to float32 in sample() before the model sees it.
        if "image" in data.keys():
            data["image"] = (data["image"] * 255).byte()
        data.set("episode", torch.zeros(n_rows, dtype=torch.int32))
        self._buffer.extend(data)
        #! Queue every complete fresh segment for online sampling. Segments tile
        #! each env's stream without overlap, like the official replay's queue.
        self._steps_per_env += n_rows
        if self.online:
            while self._steps_per_env - self._next_online_start >= self._seq_len:
                for env in range(self.num_envs):
                    self._online_queue.append((self._next_online_start, env))
                self._next_online_start += self._seq_len

    def sample(self):
        sample_td, info = self._buffer.sample(return_info=True)
        # SliceSampler returns B*(T+1) contiguous steps in a flat TensorDict.
        # Reshape to (B, T+1) so each row is one training sequence.
        sample_td = sample_td.view(-1, self.batch_length + 1)
        #! TorchRL wraps storage indices in a tuple; unwrap to a flat 1-D tensor.
        raw_index = info["index"]
        index = raw_index[0] if isinstance(raw_index, tuple) else raw_index
        #! Online sampling: overwrite leading rows with the freshest queued
        #! segments (fetched from storage by index) before uniform rows.
        n_online = min(len(self._online_queue), self.batch_size) if self.online else 0
        if n_online:
            rows = []
            for _ in range(n_online):
                start, env = self._online_queue.popleft()
                steps = torch.arange(start, start + self._seq_len, dtype=torch.long)
                rows.append((steps * self.num_envs + env) % self.max_size)
            online_idx = torch.stack(rows)  # (n_online, T+1)
            online_td = self._buffer[online_idx.reshape(-1).to(self.storage_device)]
            sample_td = torch.cat(
                [online_td.view(n_online, self._seq_len), sample_td[n_online:]], 0
            )
            index = index.view(-1, self._seq_len).clone()
            index[:n_online] = online_idx.to(dtype=index.dtype, device=index.device)
            index = index.reshape(-1)
        src_dev = sample_td.device
        if src_dev.type == "cpu" and self.device.type == "cuda" and self.pin_memory:
            sample_td = sample_td.pin_memory().to(self.device, non_blocking=True)
        elif src_dev != self.device:
            sample_td = sample_td.to(self.device, non_blocking=self.pin_memory)
        if "image" in sample_td.keys():  #! restore float32 for the model
            sample_td["image"] = sample_td["image"].float() / 255.0
        #! First timestep of each sequence is used only to warm-start RSSM state.
        initial = (sample_td["stoch"][:, 0], sample_td["deter"][:, 0])
        data = sample_td[:, 1:]
        #! clone(): `data` is a view of sample_td, so an in-place copy from the
        #! overlapping slice is undefined (R2Dreamer's version corrupted ~1 in 9
        #! steps on CUDA and nearly all on CPU).
        data.set_("action", sample_td["action"][:, :-1].clone())  # action is 1 step back
        return data, index, initial

    def update(
        self, index, stoch, deter
    ):  #! R2Dreamer used 2-D storage so index was a list of tensors and wrote `self._buffer[index[1], index[0]]`
        # index: flat 1-D tensor (B*(T+1),) — includes the initial context step.  #!
        # stoch/deter: (B, T, ...) — posterior latents for the T training steps only.  #!
        B, T = stoch.shape[0], stoch.shape[1]  #!
        training_index = index.view(B, T + 1)[:, 1:].reshape(-1)  # (B*T,)  #!
        self._buffer[training_index] = TensorDict(  #!
            {  #!
                "stoch": stoch.reshape(-1, *stoch.shape[2:]),  #!
                "deter": deter.reshape(-1, *deter.shape[2:]),  #!
            },  #!
            batch_size=(B * T,),  #!
        )  #!

    def count(self):
        if self._buffer.storage.shape is None:
            return 0
        return self._buffer.storage.shape.numel()
