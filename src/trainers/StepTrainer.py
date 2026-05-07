"""Step-based trainer using ``SyncDataCollector``.

Each iteration:  collector yields one batch of transitions
                 ->  ``algorithm.step(batch)``
                 ->  fire callbacks if it's a logging step.

The trainer owns the loop, the collector and the callbacks; everything that
affects learning lives in the algorithm.
"""
from __future__ import annotations

from src.algorithms.utils import last_episode_return
from src.trainers.BaseTrainer import BaseTrainer, TrainerEvent, fire_callbacks


class StepTrainer(BaseTrainer):
    def setup(self) -> None:
        super().setup()
        self._create_collector()

    def _create_collector(self) -> None:
        from torchrl.collectors import Collector

        cc = self.algorithm.get_collector_config()
        self.collector = Collector(
            create_env_fn=self.train_env,
            policy=self.algorithm.get_explore_policy(),
            frames_per_batch=cc.frames_per_batch,
            total_frames=int(self.trainer_cfg.total_frames),
            init_random_frames=cc.init_random_frames,
            max_frames_per_traj=cc.max_frames_per_traj,
            device=self.device,
            storing_device=self.device,
        )

    def _training_loop(self) -> dict[str, float]:
        log_every = int(self.trainer_cfg.log_every_n_steps)
        metrics: dict[str, float] = {}

        for batch in self.collector:
            batch_frames = batch.numel()
            self._step += batch_frames

            metrics = self.algorithm.step(batch)

            if self._should_log(log_every, batch_frames):
                metrics["reward/last"] = last_episode_return(batch)
                fire_callbacks(
                    TrainerEvent.ON_STEP_END,
                    self.callbacks,
                    metrics=metrics,
                    step=self._step,
                )

        return metrics
