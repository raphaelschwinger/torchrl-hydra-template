from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.trainers import BaseTrainer


class CheckpointCallback:
    """Saves full training state at fixed step intervals and optionally at the end.

    Args:
        save_dir: directory where checkpoint files are written
        save_every_n_steps: save a checkpoint every this many environment steps
        save_last: if True, save "last.pt" when training finishes
    """

    def __init__(
        self,
        save_dir: str | Path,
        save_every_n_steps: int,
        save_last: bool = True,
    ) -> None:
        self.save_dir = Path(save_dir)
        self.save_every_n_steps = save_every_n_steps
        self.save_last = save_last
        self._trainer: BaseTrainer | None = None
        self._last_saved_step: int = 0

    def set_trainer(self, trainer: BaseTrainer) -> None:
        """Inject the trainer instance (called by build_callbacks)."""
        self._trainer = trainer

    def on_train_start(self, state: dict[str, Any]) -> None:
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def on_step_end(self, metrics: dict[str, float], step: int) -> None:
        if self._trainer is None:
            return
        # Check if we've crossed a save boundary since the last save
        if step // self.save_every_n_steps > self._last_saved_step // self.save_every_n_steps:
            path = self.save_dir / f"step_{step:010d}.pt"
            self._trainer.save_checkpoint(path)
            self._last_saved_step = step

    def on_train_end(self, state: dict[str, Any]) -> None:
        if self._trainer is None:
            return
        if self.save_last:
            self._trainer.save_checkpoint(self.save_dir / "last.pt")
