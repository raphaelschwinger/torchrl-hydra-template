from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class WandBLogger:
    """Logs metrics to Weights & Biases in an openrlbenchmark-readable layout.

    Two things here are load-bearing for openrlbenchmark compatibility:

    1. ``global_step`` is logged as a **data column**, not as ``wandb.log(step=)``.
       ``rlops`` reads history with
       ``run.history(keys=[xaxis, "_runtime", metric]).dropna()``, so the metric
       and the x-axis must land in the same history row. Passing ``step=``
       writes W&B's internal ``_step`` instead and yields an empty join.
    2. ``env_id`` / ``exp_name`` / ``seed`` must sit at the **top level** of the
       run config. They are top-level keys in ``configs/train.yaml``, so dumping
       the resolved config places them correctly; nesting them would force
       fragile ``ceik=environment.value.task`` style filters.

    Args:
        project: W&B project name
        entity: W&B entity (team/user). None uses the default from wandb login.
        name: run name. None lets W&B generate one.
        tags: list of tags to attach to the run
        mode: "online", "offline", or "disabled"
        save_dir: directory for W&B's local run files
        run_id: explicit W&B run id; None generates one (or reads ``run_id_file``
            when resuming)
        resume: W&B resume mode ("allow" / "must" / None). With ``run_id_file``
            this is how ``src/eval.py`` appends evaluation results to the run
            that produced the checkpoint instead of creating a second run.
        job_type: W&B job type, e.g. "train" or "eval"
        run_id_file: JSON sidecar holding this run's id. Written on train start,
            read back when resuming. A sidecar rather than a checkpoint field
            because checkpoints unpickle a dataclass, and adding a field to it
            would break every checkpoint already on disk.
    """

    def __init__(
        self,
        project: str = "torchrl-hydra-template",
        entity: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        mode: str = "online",
        save_dir: str | None = None,
        run_id: str | None = None,
        resume: str | None = None,
        job_type: str | None = None,
        run_id_file: str | None = None,
    ) -> None:
        self.project = project
        self.entity = entity
        self.name = name
        self.tags = tags or []
        self.mode = mode
        self.save_dir = save_dir
        self.run_id = run_id
        self.resume = resume
        self.job_type = job_type
        self.run_id_file = run_id_file
        self._run = None

    def _read_run_id(self) -> str | None:
        if self.run_id is not None:
            return self.run_id
        if not self.run_id_file:
            return None
        path = Path(self.run_id_file)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text()).get("id")
        except (json.JSONDecodeError, OSError):
            return None

    def _write_run_id(self) -> None:
        if not self.run_id_file or self._run is None:
            return
        path = Path(self.run_id_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "id": self._run.id,
                    "project": self.project,
                    "entity": self.entity,
                }
            )
        )

    def on_train_start(self, state: dict[str, Any]) -> None:
        import wandb
        from omegaconf import OmegaConf

        if self.save_dir is not None:
            Path(self.save_dir).mkdir(parents=True, exist_ok=True)

        run_id = self._read_run_id()
        if self.resume is not None and run_id is None:
            raise RuntimeError(
                f"resume={self.resume!r} was requested but no W&B run id was found "
                f"(run_id_file={self.run_id_file!r}). Without an id, wandb would "
                "silently start a *new* run instead of appending. Pass "
                "logger.0.run_id=<id> explicitly, or point checkpoint.resume_from "
                "at a checkpoint whose directory contains wandb_run.json."
            )

        cfg = state.get("cfg")
        config_dict = OmegaConf.to_container(cfg, resolve=True) if cfg is not None else {}
        self._run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.name,
            tags=self.tags,
            mode=self.mode,
            config=config_dict,
            dir=self.save_dir,
            id=run_id,
            resume=self.resume,
            job_type=self.job_type,
        )
        # Make global_step the x-axis for every metric, so panels and
        # openrlbenchmark agree on the axis without a per-run `xaxis=` override.
        wandb.define_metric("global_step")
        wandb.define_metric("*", step_metric="global_step")
        if self.resume is None:
            self._write_run_id()

    def on_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self._run is not None:
            import wandb
            # No step= on purpose; see the class docstring.
            wandb.log(metrics)

    def on_train_end(self, state: dict[str, Any]) -> None:
        if self._run is None:
            return
        import wandb

        summary = state.get("summary") or {}
        for key, value in summary.items():
            self._run.summary[key] = value
        wandb.finish()
        self._run = None


class TensorBoardLogger:
    """Logs metrics to TensorBoard.

    Args:
        log_dir: directory where TensorBoard event files are written
    """

    def __init__(self, log_dir: str) -> None:
        self.log_dir = log_dir
        self._writer = None

    def on_train_start(self, state: dict[str, Any]) -> None:
        from torch.utils.tensorboard import SummaryWriter
        self._writer = SummaryWriter(log_dir=self.log_dir)

    def on_metrics(self, metrics: dict[str, float], step: int) -> None:
        if self._writer is None:
            return
        global_step = int(metrics.get("global_step", step))
        for key, value in metrics.items():
            if key == "global_step":
                continue
            if isinstance(value, (int, float)):
                self._writer.add_scalar(key, value, global_step=global_step)

    def on_train_end(self, state: dict[str, Any]) -> None:
        if self._writer is not None:
            for key, value in (state.get("summary") or {}).items():
                self._writer.add_scalar(key, value, global_step=0)
            self._writer.flush()
            self._writer.close()
            self._writer = None
