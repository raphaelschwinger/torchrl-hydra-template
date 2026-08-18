"""Evaluation entry point.

Runs the evaluation protocol from ``configs/evaluation/`` against a checkpoint
and logs the results — one row per episode plus the ``eval/*`` aggregate — with
the same W&B layout as training.

Usage:
    python src/eval.py experiment=dqn/gym checkpoint.resume_from=logs/.../last.pt

    # append the results to the training run instead of creating a new one
    python src/eval.py experiment=bbf/atari100k \
        checkpoint.resume_from=logs/.../checkpoints/last.pt logger.0.resume=must
"""
from __future__ import annotations

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="../configs", config_name="eval", version_base="1.3")
def evaluate(cfg: DictConfig) -> None:
    results = _evaluate(cfg)
    print("\nEvaluation results:")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")


def _configure_eval_logging(cfg: DictConfig) -> None:
    """Normalise W&B logger settings for the evaluation entry point.

    Done in code rather than in a config group because ``experiment=`` configs
    carry their own ``- override /logger: wandb``, which beats anything
    ``configs/eval.yaml`` selects. Without this, an eval run would silently
    inherit the training logger's ``job_type`` and — worse — a ``run_id_file``
    pointing at the *eval* run's own output dir, so ``resume`` would quietly
    start a fresh run instead of appending to the training run.
    """
    from pathlib import Path

    resume_from = cfg.checkpoint.get("resume_from")
    sidecar = (
        str(Path(resume_from).parent / "wandb_run.json")
        if resume_from is not None
        else None
    )

    for logger_cfg in cfg.get("logger") or []:
        if "WandBLogger" not in str(logger_cfg.get("_target_", "")):
            continue
        logger_cfg.job_type = "eval"
        tags = list(logger_cfg.get("tags") or [])
        if "eval" not in tags:
            tags.append("eval")
        logger_cfg.tags = tags
        if logger_cfg.get("resume"):
            # Always the checkpoint's sidecar: resuming means "the run that
            # produced this checkpoint", never whatever the training config
            # happened to interpolate.
            logger_cfg.run_id_file = sidecar
        else:
            # A standalone eval run must not overwrite the training sidecar.
            logger_cfg.run_id_file = None


def _evaluate(cfg: DictConfig) -> dict[str, float]:
    from src.utils.instantiate import build_trainer
    from src.utils.seeding import seed_everything

    seed_everything(int(cfg.trainer.seed))
    _configure_eval_logging(cfg)

    if not bool(cfg.get("train", False)) and cfg.evaluation.get("summary_max_step"):
        # `summary_max_step` trims *training-stream* episodes completed past a
        # benchmark budget. Without training, every canonical episode is logged
        # at the checkpoint's step — for a run that trains past the budget on
        # purpose (dreamer/atari100k: 110k steps, cutoff 100k) that is the whole
        # set, so summary() would filter all of them out and silently write no
        # `eval/final_return_*` at all.
        cfg.evaluation.summary_max_step = None

    trainer = build_trainer(cfg)

    trainer.setup()
    trainer.load_checkpoint(cfg.checkpoint.resume_from)

    # `train: false` in configs/eval.yaml; the evaluation stage runs the
    # `evaluation.final_num_episodes` protocol and logs it.
    return trainer.run(
        train=bool(cfg.get("train", False)),
        evaluate=bool(cfg.get("eval", True)),
    )


if __name__ == "__main__":
    evaluate()
