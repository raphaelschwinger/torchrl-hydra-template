"""Step-based trainer for algorithms like DQN and PPO."""
from __future__ import annotations

from src.algorithms.utils import last_episode_return
from src.trainers.BaseTrainer import BaseTrainer, TrainerEvent, fire_callbacks


class StepTrainer(BaseTrainer):
    """Trainer for step-based algorithms using ``SyncDataCollector``.

    Each iteration: collector yields a batch → ``algorithm.step(batch)``.
    Used by DQN, PPO.
    """

    def setup(self) -> None:
        """Create env, set up algorithm, then create the collector."""
        super().setup()
        self._create_collector()

    def _create_collector(self) -> None:
        from torchrl.collectors import Collector 

        collector_cfg = self.algorithm.get_collector_config()

        self.collector = Collector(
            create_env_fn=self.train_env,
            policy=self.algorithm.get_explore_policy(),
            frames_per_batch=collector_cfg.frames_per_batch,
            total_frames=self.trainer_cfg.total_frames,
            device=self.device,
            storing_device=self.device,
            max_frames_per_traj=collector_cfg.max_frames_per_traj,
        )

    def _training_loop(self) -> dict[str, float]:
        log_every = int(self.trainer_cfg.log_every_n_steps)
        metrics: dict[str, float] = {}

        collected_frames = 0
        batch_size = self.trainer_cfg.batch_size
        test_interval = self.trainer_cfg.test_interval



        c_iter = iter(self.collector)
        total_iter = len(self.collector)
        for i in range(total_iter):
            batch = next(c_iter)

            batch_frames = batch.numel()
            collected_frames += batch_frames

            metrics = self.algorithm.step(batch_frames)

            self.algorithm.on_step_complete(self._step)

            if self._should_log(log_every, batch_frames):
                metrics["reward/last"] = last_episode_return(batch)
                fire_callbacks(
                    TrainerEvent.ON_STEP_END,
                    self.callbacks,
                    metrics=metrics,
                    step=self._step,
                )

        return metrics
