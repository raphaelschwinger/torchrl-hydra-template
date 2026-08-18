"""Guards on the openrlbenchmark logging contract.

These assertions look pedantic, but each one corresponds to a way a run can be
silently dropped from an openrlbenchmark comparison rather than fail loudly:

- ``rlops`` joins with ``run.history(keys=[xaxis, "_runtime", metric]).dropna()``,
  so a metric logged without ``global_step`` *in the same row* contributes
  nothing and only prints "Skipping run".
- its Atari human-normalised-score table is keyed ``"Pong-v5"``; an ``env_id``
  of ``"ALE/Pong-v5"`` raises ``KeyError``.
- runs are grouped by ``(exp_name, env_id)``, so two variants sharing both
  silently merge into one hypothesis.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tests.conftest import CONFIGS_DIR, load_experiment_cfg


EXPERIMENTS = sorted(
    f"{p.parent.name}/{p.stem}"
    for p in (Path(CONFIGS_DIR) / "experiment").glob("*/*.yaml")
)


def _compose_resolved(experiment: str):
    """Compose with the hydra node available so ``${hydra:...}`` resolves."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.core.hydra_config import HydraConfig

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIGS_DIR, version_base="1.3"):
        cfg = compose(
            config_name="train",
            overrides=[f"experiment={experiment}"],
            return_hydra_config=True,
        )
        # exp_name interpolates ${hydra:runtime.choices.algorithm}, which needs
        # the HydraConfig singleton populated as it is under @hydra.main.
        HydraConfig.instance().set_config(cfg)
        return cfg


def test_experiments_discovered():
    assert len(EXPERIMENTS) >= 11, EXPERIMENTS


@pytest.mark.parametrize("experiment", EXPERIMENTS)
def test_openrlbenchmark_config_keys(experiment: str):
    cfg = _compose_resolved(experiment)

    env_id = cfg.env_id
    assert env_id and env_id != "unknown", f"{experiment} has no env_id"
    assert not env_id.startswith("ALE/"), (
        f"{experiment}: env_id={env_id!r} — openrlbenchmark's HNS table is keyed "
        "without the ALE/ namespace"
    )
    assert cfg.exp_name, f"{experiment} has no exp_name"
    assert isinstance(cfg.seed, int), f"{experiment}: seed must be an int for ?seed= filters"
    assert int(cfg.environment.action_repeat) >= 1


def test_exp_name_env_id_pairs_are_unique():
    """Two experiments sharing (exp_name, env_id) would merge into one curve."""
    seen: dict[tuple[str, str], str] = {}
    for experiment in EXPERIMENTS:
        cfg = _compose_resolved(experiment)
        key = (cfg.exp_name, cfg.env_id)
        assert key not in seen, (
            f"{experiment} and {seen[key]} both log exp_name={key[0]!r} "
            f"env_id={key[1]!r}; set a distinct `exp_name` in one of them"
        )
        seen[key] = experiment


class _Recorder:
    """Captures every metric row the trainer emits."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def on_metrics(self, metrics: dict, step: int) -> None:
        self.rows.append(dict(metrics))


def _tiny_dqn_trainer(extra_overrides: list[str]):
    from src.utils.instantiate import build_trainer

    cfg = load_experiment_cfg(
        "dqn/gym",
        [
            "logger=[]",
            "trainer.accelerator=cpu",
            "trainer.devices=[0]",
            "trainer.total_frames=600",
            "trainer.log_every_n_steps=100",
            "algorithm.frames_per_batch=100",
            "algorithm.init_random_frames=100",
            "algorithm.batch_size=8",
            "algorithm.num_updates=2",
            "algorithm.annealing_frames=600",
            "checkpoint.enabled=false",
            "hydra.run.dir=/tmp/hydra_smoke_tests",
            *extra_overrides,
        ],
    )
    trainer = build_trainer(cfg)
    recorder = _Recorder()
    trainer.callbacks.append(recorder)
    trainer.setup()
    return trainer, recorder


def test_every_row_carries_global_step_and_frames():
    trainer, recorder = _tiny_dqn_trainer(["evaluation.every_n_steps=200"])
    trainer.run(train=True, evaluate=True)

    assert recorder.rows
    for row in recorder.rows:
        assert "global_step" in row, row
        assert "frames" in row, row
        assert row["frames"] == row["global_step"] * trainer.action_repeat


def test_canonical_metric_co_occurs_with_global_step():
    """charts/episodic_return and global_step must land in the SAME row."""
    trainer, recorder = _tiny_dqn_trainer([])
    trainer.run(train=True, evaluate=False)

    canonical = [r for r in recorder.rows if "charts/episodic_return" in r]
    assert canonical, "no canonical episodic-return rows were emitted"
    for row in canonical:
        assert "global_step" in row
    # One row per episode, not one pre-aggregated point per log boundary.
    assert len(canonical) > 1


def test_periodic_evaluation_emits_one_row_per_episode():
    trainer, recorder = _tiny_dqn_trainer(
        ["evaluation.every_n_steps=200", "evaluation.num_episodes=3"]
    )
    trainer.run(train=True, evaluate=False)

    eval_rows = [r for r in recorder.rows if "charts/eval_episodic_return" in r]
    aggregates = [r for r in recorder.rows if "eval/return_mean" in r]
    assert aggregates, "periodic evaluation did not run"
    assert len(eval_rows) == 3 * len(aggregates)
    for row in aggregates:
        # std of a single episode is NaN; with >1 episode it must be finite.
        assert row["eval/return_std"] == row["eval/return_std"]
        assert row["eval/episodes"] == 3


def test_canonical_source_eval_mirrors_eval_rollouts():
    trainer, recorder = _tiny_dqn_trainer(
        [
            "evaluation.every_n_steps=200",
            "evaluation.num_episodes=2",
            "evaluation.canonical_source=eval",
        ]
    )
    trainer.run(train=True, evaluate=False)

    for row in recorder.rows:
        # With canonical_source=eval, training episodes must NOT feed the
        # canonical key, and eval episodes must.
        if "charts/train_episodic_return" in row:
            assert "charts/episodic_return" not in row
        if "charts/eval_episodic_return" in row:
            assert row["charts/episodic_return"] == row["charts/eval_episodic_return"]


def test_evaluation_preserves_recurrent_policy_state():
    """Eval rollouts must carry the policy's own root keys across steps.

    A recurrent policy (DreamerPolicy's RSSM: `stoch` / `deter` / `prev_action`)
    keeps its state at the root of the tensordict, exactly where the collector
    leaves it. Advancing the rollout with ``td["next"]`` keeps only env-written
    keys, which silently resets that state on every step — the policy still runs,
    it just acts from a fresh latent and scores near zero.
    """
    trainer, _ = _tiny_dqn_trainer([])

    original = trainer.algorithm.get_policy()
    seen: list[int] = []

    def recurrent_policy(td):
        # Read the state this policy left behind on the previous step.
        carried = int(td["policy_state"].item()) if "policy_state" in td.keys() else 0
        seen.append(carried)
        td = original(td)
        td.set("policy_state", torch.full((1,), carried + 1, dtype=torch.long))
        return td

    trainer.algorithm.get_policy = lambda: recurrent_policy
    trainer.evaluate(num_episodes=1)

    assert len(seen) > 1, "evaluation did not step the policy"
    # 0 on reset, then one increment per step: a dropped state would be all 0s.
    assert seen == list(range(len(seen))), seen


def test_evaluation_restores_module_training_flags():
    """Rainbow's get_policy() toggles .training on shared modules; periodic
    evaluation must not leak that into the training loop."""
    trainer, _ = _tiny_dqn_trainer([])
    modules = trainer._algorithm_modules()
    assert modules

    original = trainer.algorithm.get_policy
    def flipping_get_policy():
        policy = original()
        for module in modules:
            module.train(not module.training)
        return policy

    trainer.algorithm.get_policy = flipping_get_policy
    for module in modules:
        module.train(True)

    trainer.run_evaluation(num_episodes=1, step=0)

    assert all(m.training for m in modules), "evaluation leaked module mode changes"


class _FakeRun:
    id = "abc123"

    def __init__(self) -> None:
        self.summary: dict = {}


class _FakeWandb:
    def __init__(self) -> None:
        self.run = None
        self.logged: list[tuple[dict, dict]] = []
        self.defined: list[tuple[tuple, dict]] = []
        self.init_kwargs: dict = {}
        self.finished = False

    def init(self, **kwargs):
        self.init_kwargs = kwargs
        self.run = _FakeRun()
        return self.run

    def define_metric(self, *args, **kwargs):
        self.defined.append((args, kwargs))

    def log(self, metrics, **kwargs):
        self.logged.append((metrics, kwargs))

    def finish(self):
        self.finished = True


@pytest.fixture
def fake_wandb(monkeypatch):
    import sys

    fake = _FakeWandb()
    monkeypatch.setitem(sys.modules, "wandb", fake)
    return fake


def test_wandb_logger_writes_global_step_as_a_column(fake_wandb, tmp_path):
    from src.callbacks.logger import WandBLogger

    sidecar = tmp_path / "checkpoints" / "wandb_run.json"
    logger = WandBLogger(mode="offline", run_id_file=str(sidecar))
    logger.on_train_start({"cfg": None})
    logger.on_metrics({"charts/episodic_return": 1.0, "global_step": 42, "frames": 168}, 42)
    logger.on_train_end({"summary": {"eval/final_return_mean": 7.0}})

    metrics, kwargs = fake_wandb.logged[0]
    # step= would write W&B's internal _step and break rlops' history join.
    assert "step" not in kwargs
    assert metrics["global_step"] == 42
    assert ("global_step",) in [a for a, _ in fake_wandb.defined]
    assert ("*",) in [a for a, _ in fake_wandb.defined]
    assert {"step_metric": "global_step"} in [k for _, k in fake_wandb.defined]

    import json

    assert json.loads(sidecar.read_text())["id"] == "abc123"
    assert fake_wandb.finished


def test_wandb_logger_resumes_from_sidecar(fake_wandb, tmp_path):
    import json

    from src.callbacks.logger import WandBLogger

    sidecar = tmp_path / "wandb_run.json"
    sidecar.write_text(json.dumps({"id": "prior-run", "project": "p", "entity": None}))

    logger = WandBLogger(mode="offline", run_id_file=str(sidecar), resume="must")
    logger.on_train_start({"cfg": None})

    assert fake_wandb.init_kwargs["id"] == "prior-run"
    assert fake_wandb.init_kwargs["resume"] == "must"


def test_wandb_logger_refuses_to_resume_without_an_id(fake_wandb, tmp_path):
    from src.callbacks.logger import WandBLogger

    logger = WandBLogger(
        mode="offline", run_id_file=str(tmp_path / "missing.json"), resume="must"
    )
    with pytest.raises(RuntimeError, match="no W&B run id was found"):
        logger.on_train_start({"cfg": None})


def test_eval_entry_point_normalises_logger_config():
    """`experiment=` configs force `override /logger: wandb`, which beats
    anything configs/eval.yaml selects — so eval.py must fix up the logger
    itself, or `resume` silently forks a new run."""
    from omegaconf import OmegaConf

    from src.eval import _configure_eval_logging

    cfg = OmegaConf.create(
        {
            "checkpoint": {"resume_from": "/runs/a/checkpoints/last.pt"},
            "logger": [
                {
                    "_target_": "src.callbacks.logger.WandBLogger",
                    "job_type": "train",
                    "tags": [],
                    "resume": "must",
                    "run_id_file": "/some/other/eval-run/wandb_run.json",
                }
            ],
        }
    )
    _configure_eval_logging(cfg)

    logger_cfg = cfg.logger[0]
    assert logger_cfg.job_type == "eval"
    assert "eval" in logger_cfg.tags
    assert logger_cfg.run_id_file == "/runs/a/checkpoints/wandb_run.json"


def test_eval_entry_point_does_not_clobber_training_sidecar():
    """A standalone eval run must not overwrite the training run's sidecar."""
    from omegaconf import OmegaConf

    from src.eval import _configure_eval_logging

    cfg = OmegaConf.create(
        {
            "checkpoint": {"resume_from": "/runs/a/checkpoints/last.pt"},
            "logger": [
                {
                    "_target_": "src.callbacks.logger.WandBLogger",
                    "job_type": "train",
                    "tags": [],
                    "resume": None,
                    "run_id_file": "/runs/a/checkpoints/wandb_run.json",
                }
            ],
        }
    )
    _configure_eval_logging(cfg)

    assert cfg.logger[0].run_id_file is None


def test_summary_respects_window_and_cutoff():
    trainer, _ = _tiny_dqn_trainer([])
    trainer._canonical_episodes = [(100, 1.0), (200, 2.0), (300, 3.0), (400, 4.0)]

    trainer.eval_cfg = {"summary_window": 2, "summary_max_step": None}
    assert trainer.summary()["eval/final_return_mean"] == pytest.approx(3.5)

    trainer.eval_cfg = {"summary_window": 2, "summary_max_step": 200}
    assert trainer.summary()["eval/final_return_mean"] == pytest.approx(1.5)

    trainer.eval_cfg = {"summary_window": 100, "summary_max_step": None}
    summary = trainer.summary()
    assert summary["eval/final_return_episodes"] == 4
