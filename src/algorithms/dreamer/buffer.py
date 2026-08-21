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


class Buffer:
    def __init__(self, config):
        self.device = torch.device(config.device)
        self.storage_device = torch.device(config.storage_device)
        self.batch_size = int(config.batch_size)
        self.batch_length = int(config.batch_length)
        self.num_envs = int(getattr(config, "num_envs", 1))  #! R2Dreamer had `self.num_eps = 0` (unused episode counter); replaced with num_envs for episode-ID stamping
        self.max_size = int(config.max_size)
        #! Online sampling (official DreamerV3 `replay.online: True`, absent from
        #! R2Dreamer): every fresh, non-overlapping (batch_length+1)-step segment
        #! is queued and served before uniform samples, so all new experience is
        #! trained on immediately at least once.
        self.online = bool(getattr(config, "online", True))
        #! Pinned (page-locked) staging for the CPU->GPU copy of a sampled batch.
        #! On by default (the published configuration); set false to measure what
        #! the transfer costs without it.
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
                max_size=int(config.max_size), device=self.storage_device
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

    def add_transition(self, data):
        n_frames = data.batch_size[0] if data.batch_size else 1  #! R2Dreamer received single frames (B=num_envs); now frames_per_batch may be > num_envs
        data = data.copy()  #! copy to avoid mutating the collector's TensorDict
        #! Store image as uint8 to save 4x memory
        #! Converted back to float32 in sample() before the model sees it.
        if "image" in data.keys():
            data["image"] = (data["image"] * 255).byte()
        #! frames_per_batch may exceed num_envs (multiple steps per env per call).
        #! Tile env indices so each step is stamped with the correct stream id.
        episode_ids = torch.arange(self.num_envs, dtype=torch.int32).repeat(n_frames // self.num_envs)
        data.set("episode", episode_ids)
        self._buffer.extend(data)
        #! Queue every complete fresh segment for online sampling. Segments tile
        #! each env's stream without overlap, like the official replay's queue.
        self._steps_per_env += n_frames // self.num_envs
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
        data.set_("action", sample_td["action"][:, :-1])  # action is 1 step back
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
