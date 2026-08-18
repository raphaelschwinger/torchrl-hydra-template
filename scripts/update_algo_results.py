#!/usr/bin/env python3
"""Populate algorithm README experimental-results tables from W&B runs.

Fetches finished runs tagged ``template`` from the torchrl-hydra-template W&B
project and rewrites the markdown table under ``## Experimental results`` in
each algorithm README.

Usage::

    # Requires ``wandb login`` (or WANDB_API_KEY).
    python scripts/update_algo_results.py

    # Preview without writing files.
    python scripts/update_algo_results.py --dry-run

    # Override project / entity / tag.
    python scripts/update_algo_results.py --entity LatentLab --tag template
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
def algo_readme_path(algo: str) -> Path:
    return REPO_ROOT / "src" / "algorithms" / algo / "README.md"
CONFIG_DIR = REPO_ROOT / "configs"
EXPERIMENT_DIR = CONFIG_DIR / "experiment"
WANDB_TABLE_URL = "https://wandb.ai/LatentLab/torchrl-hydra-template/table"
CANONICAL_ENTITY = "LatentLab"
DEFAULT_PROJECT = "torchrl-hydra-template"

TABLE_HEADER = (
    "| Run | Environment | Config | Seed | Frames | Eval return | Notes |"
)
TABLE_SEPARATOR = (
    "|-----|-------------|--------|------|--------|-------------|-------|"
)

# Maps ``src.algorithms.<pkg>.<Class>`` prefix to README package directory.
ALGO_TARGET_PREFIXES: dict[str, str] = {
    "src.algorithms.dqn.": "dqn",
    "src.algorithms.ddpg.": "ddpg",
    "src.algorithms.a2c.": "a2c",
    "src.algorithms.ppo.": "ppo",
    "src.algorithms.tdmpc2.": "tdmpc2",
    "src.algorithms.dreamer.": "dreamer",
    "src.algorithms.rainbow.": "rainbow",
    "src.algorithms.bbf.": "bbf",
}


@dataclass(frozen=True)
class ExperimentSpec:
    """One composed experiment under ``configs/experiment/``.

    Built by actually composing the Hydra config rather than regex-scraping the
    YAML, so this reads the same source of truth as the W&B run config it is
    matched against.
    """

    path: str  # e.g. ``dqn/gym``
    algorithm_choice: str | None  # set when it differs from the experiment default
    algo_identity: tuple  # (target, obs_key, encoder_type, world-model target)
    env_family: str  # ``gym`` | ``dm_control`` | ``ale``
    env_task: str | None  # this experiment's default task
    algo_scalars: tuple  # every scalar algorithm kwarg, for tie-breaking


@dataclass(frozen=True)
class ResultRow:
    """One row in an algorithm README results table."""

    run_name: str
    run_url: str
    environment: str
    config: str
    seed: int | None
    frames: int | None
    eval_return: str
    notes: str


def _compose(overrides: list[str]):
    """Compose the training config outside a Hydra runtime.

    Only ``algorithm`` and ``environment`` are ever resolved by callers, so the
    ``${hydra:...}`` interpolations in ``paths`` / ``run_name`` are never hit.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.3"):
        return compose(config_name="train", overrides=overrides)


def _to_dict(node) -> dict:
    from omegaconf import OmegaConf

    if node is None:
        return {}
    return OmegaConf.to_container(node, resolve=True)


def algo_scalars(algo_cfg: dict) -> tuple:
    """Every scalar algorithm kwarg, as a sorted ``(key, value)`` tuple.

    Used only to break ties between experiments that share an identity and a
    benchmark — e.g. BBF's RR2 and RR8 variants, which differ solely by
    ``replay_ratio``. Comparing all scalars avoids having to hand-register a
    new discriminator every time such a pair appears.
    """
    return tuple(
        sorted(
            (k, v)
            for k, v in algo_cfg.items()
            if isinstance(v, (int, float, bool, str)) or v is None
        )
    )


def _class_identity(target: str | None) -> tuple | None:
    """``(package, ClassName)`` — stable when a module path is refactored.

    Historical runs logged e.g. ``src.algorithms.dreamer.dreamer.DreamerAlgorithm``
    before the package re-export shortened it; both must resolve to the same
    algorithm.
    """
    if not target:
        return None
    return (algo_package_from_target(target), target.rsplit(".", 1)[-1])


def algo_identity(algo_cfg: dict) -> tuple:
    """Identity of an algorithm setup, shared by specs and W&B run configs.

    ``obs_key`` separates the pixel and state variants of one algorithm class,
    ``encoder_type`` separates Rainbow from its data-efficient preset, and the
    world-model class separates DreamerV3 / R2Dreamer / DreamerPro.
    """
    return (
        _class_identity(algo_cfg.get("_target_")),
        algo_cfg.get("obs_key") or "observation",
        algo_cfg.get("encoder_type"),
        _class_identity((algo_cfg.get("dreamer_config") or {}).get("_target_")),
    )


def env_family(env_cfg: dict) -> str:
    """Coarse benchmark identity — stable across a change of task.

    Deliberately coarse for Atari: the preprocessing stack has changed shape
    over time (max-and-skip moved into ``gymnasium_wrappers``), and pinning the
    family to it would orphan older runs of an experiment that still exists.
    Which ALE protocol a run used is already carried by the algorithm identity
    (``obs_key``, ``encoder_type``).
    """
    if env_cfg.get("backend") == "dm_control":
        return "dm_control"
    if str(env_cfg.get("name") or "").startswith("ALE/"):
        return "ale"
    return "gym"


def env_task(env_cfg: dict) -> str | None:
    """The task within a benchmark, normalised across old and new config shapes."""
    task = env_cfg.get("task")
    name = env_cfg.get("name")
    if env_cfg.get("backend") == "dm_control":
        if task and "-" in task:
            return task                      # new form: cheetah-run
        if name and task:
            return f"{name}-{task}"          # old form: name: cheetah + task: run
        return (name or task or "").replace("/", "-") or None
    if name:
        match = re.fullmatch(r"ALE/(.+)-v\d+", name)
        return match.group(1) if match else name
    return task


def load_experiment_registry(root: Path = EXPERIMENT_DIR) -> list[ExperimentSpec]:
    """Compose every experiment (and every algorithm variant of it) into a spec.

    Algorithm options sharing an experiment's algorithm class are enumerated so
    runs launched as e.g. ``experiment=dreamer/atari100k algorithm=r2dreamer``
    still resolve to a reproducible command.
    """
    specs: list[ExperimentSpec] = []
    algo_options = _algorithm_options()

    for path in sorted(root.rglob("*.yaml")):
        exp_path = path.relative_to(root).with_suffix("").as_posix()
        try:
            cfg = _compose([f"experiment={exp_path}"])
        except Exception:
            continue
        base_algo = _to_dict(cfg.algorithm)
        env_cfg = _to_dict(cfg.environment)
        family = env_family(env_cfg)
        task = env_task(env_cfg)

        specs.append(
            ExperimentSpec(
                path=exp_path,
                algorithm_choice=None,
                algo_identity=algo_identity(base_algo),
                env_family=family,
                env_task=task,
                algo_scalars=algo_scalars(base_algo),
            )
        )

        # Sibling algorithm options of the same class (dreamerpro, r2dreamer...).
        for option, target in algo_options.items():
            if target != base_algo.get("_target_"):
                continue
            try:
                variant = _compose([f"experiment={exp_path}", f"algorithm={option}"])
            except Exception:
                continue
            variant_algo = _to_dict(variant.algorithm)
            identity = algo_identity(variant_algo)
            if identity == specs[-1].algo_identity:
                continue
            specs.append(
                ExperimentSpec(
                    path=exp_path,
                    algorithm_choice=option,
                    algo_identity=identity,
                    env_family=family,
                    env_task=task,
                    algo_scalars=algo_scalars(variant_algo),
                )
            )
    return specs


def _algorithm_options() -> dict[str, str]:
    """``{option name: _target_}`` for every top-level algorithm config.

    Composed rather than regex-scraped: variants like ``r2dreamer`` inherit
    ``_target_`` through their ``defaults:`` list and have no literal key of
    their own. Only the ``_target_`` leaf is read, so configs with unresolvable
    interpolations (dreamer's ``${model.*}``) do not need a size preset here.
    """
    options: dict[str, str] = {}
    for path in sorted((CONFIG_DIR / "algorithm").glob("*.yaml")):
        try:
            cfg = _compose([f"algorithm={path.stem}", "environment=gym"])
            target = cfg.algorithm._target_
        except Exception:
            continue
        if target:
            options[path.stem] = str(target)
    return options


def algo_package_from_target(target: str | None) -> str | None:
    if not target:
        return None
    for prefix, package in ALGO_TARGET_PREFIXES.items():
        if target.startswith(prefix):
            return package
    return None


def infer_experiment_config(
    config: dict,
    registry: list[ExperimentSpec],
) -> str:
    """Return the CLI command that reproduces a W&B run.

    Matching is on algorithm identity plus benchmark family, so a run of a game
    or task the experiment does not default to still resolves — the differing
    task comes back as an explicit ``environment.task=`` override.
    """
    explicit = config.get("experiment")
    if explicit not in (None, "", "null"):
        return f"experiment={explicit}"

    env_cfg = config.get("environment") or {}
    run_algo = config.get("algorithm") or {}
    identity = algo_identity(run_algo)
    family = env_family(env_cfg)
    task = env_task(env_cfg)

    candidates = [
        spec
        for spec in registry
        if spec.algo_identity == identity and spec.env_family == family
    ]
    if not candidates:
        return "—"

    # Several experiments can share an identity and a benchmark and differ only
    # in scalar hyperparameters (BBF RR2 vs RR8). Pick the one whose scalars the
    # run actually agrees with, rather than whichever sorts first.
    if len(candidates) > 1:
        run_scalars = dict(algo_scalars(run_algo))
        candidates.sort(
            key=lambda s: sum(
                1 for k, v in s.algo_scalars if k in run_scalars and run_scalars[k] == v
            ),
            reverse=True,
        )

    spec = candidates[0]
    parts = [f"experiment={spec.path}"]
    if spec.algorithm_choice:
        parts.append(f"algorithm={spec.algorithm_choice}")
    if task and task != spec.env_task:
        parts.append(f"environment.task={task}")
    return " ".join(parts)


def _read_yaml_scalar(path: Path, key: str) -> str | None:
    for line in path.read_text(encoding="utf-8").splitlines():
        match = re.match(rf"^{re.escape(key)}:\s*(.+?)\s*$", line)
        if match:
            return match.group(1).strip().strip("'\"")
    return None


def format_int(n: int | None) -> str:
    if n is None:
        return "—"
    return f"{n:,}"


def format_return(value: float | None) -> str:
    if value is None:
        return "—"
    if abs(value) >= 100:
        return f"{value:,.1f}"
    if abs(value) >= 10:
        return f"{value:.1f}"
    return f"{value:.2f}"


# openrlbenchmark's reporting rule: the mean of the last N logged points of the
# metric (its `metric_last_n_average_window`, default 100).
SUMMARY_WINDOW = 100

# Metric keys written by runs predating the unified evaluation protocol. Kept so
# already-finished runs still populate the tables; each carries a note because
# the three are not comparable with each other or with the canonical metric.
LEGACY_KEYS = (
    ("eval/score_mean_last10pct", "legacy eval/score_mean_last10pct"),
    ("eval/return_mean", "legacy eval/return_mean"),
)


def _finite(value) -> float | None:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return None
    return val if val == val else None  # drop NaN


def get_eval_return(run) -> tuple[float | None, str]:
    """The run's headline return, using one definition for every algorithm.

    Before the unified evaluation protocol this column was filled from three
    mutually incomparable sources (Dreamer's last-10%-of-frames summary, an
    eval-env mean, or the best training episode). Now it is always openrlbenchmark's
    own rule — the mean of the last 100 logged ``charts/episodic_return`` points
    — which is exactly what `rlops` puts in its own comparison tables.

    ``eval/final_return_mean`` is the trainer's precomputed version of the same
    quantity and is preferred when present; the history scan is the fallback for
    runs interrupted before the summary was written. Legacy keys come last and
    are labelled in the Notes column.
    """
    summary = run.summary

    val = _finite(summary.get("eval/final_return_mean"))
    if val is not None:
        return val, ""

    try:
        history = run.history(keys=["charts/episodic_return"], pandas=False)
        values = [
            v
            for v in (_finite(row.get("charts/episodic_return")) for row in history)
            if v is not None
        ]
        if values:
            recent = values[-SUMMARY_WINDOW:]
            return sum(recent) / len(recent), ""
    except Exception:
        pass

    for key, note in LEGACY_KEYS:
        val = _finite(summary.get(key))
        if val is not None:
            return val, note

    try:
        history = run.history(keys=["train/episode_reward"], pandas=False)
        values = [
            v
            for v in (_finite(row.get("train/episode_reward")) for row in history)
            if v is not None
        ]
        if values:
            return max(values), "legacy best train/episode_reward"
    except Exception:
        pass

    return None, ""


def parse_run(run, registry: list[ExperimentSpec]) -> ResultRow | None:
    config = dict(run.config)
    algo_cfg = config.get("algorithm") or {}
    package = algo_package_from_target(algo_cfg.get("_target_"))
    if package is None:
        return None

    trainer_cfg = config.get("trainer") or {}
    env_cfg = config.get("environment") or {}

    eval_return, metric_note = get_eval_return(run)
    notes = (run.notes or "").strip()
    if metric_note:
        notes = f"{notes}; {metric_note}".strip("; ")

    entity = run.entity
    project = run.project
    run_url = f"https://wandb.ai/{entity}/{project}/runs/{run.id}"

    return ResultRow(
        run_name=run.name,
        run_url=run_url,
        # dm_control configs carry no `name` — fall back to the task id.
        environment=env_cfg.get("name") or env_cfg.get("task") or "—",
        config=infer_experiment_config(config, registry),
        seed=trainer_cfg.get("seed"),
        frames=trainer_cfg.get("total_frames"),
        eval_return=format_return(eval_return),
        notes=notes or "—",
    )


def row_to_markdown(row: ResultRow) -> str:
    run_cell = f"[{row.run_name}]({row.run_url})"
    seed = str(row.seed) if row.seed is not None else "—"
    return (
        f"| {run_cell} | {row.environment} | `{row.config}` | {seed} | "
        f"{format_int(row.frames)} | {row.eval_return} | {row.notes} |"
    )


def build_table(rows: list[ResultRow]) -> str:
    if not rows:
        return (
            f"{TABLE_HEADER}\n"
            f"{TABLE_SEPARATOR}\n"
            f"| — | — | — | — | — | — | "
            f"No finished runs tagged ``template`` yet — see "
            f"[W&B table]({WANDB_TABLE_URL}) |"
        )

    sorted_rows = sorted(rows, key=lambda r: (r.environment, r.config, r.run_name))
    body = "\n".join(row_to_markdown(r) for r in sorted_rows)
    return f"{TABLE_HEADER}\n{TABLE_SEPARATOR}\n{body}"


def replace_results_table(content: str, new_table: str) -> str:
    pattern = re.compile(
        r"(\| Run \| Environment \| Config \| Seed \| Frames \| Eval return \| Notes \|\n"
        r"\|[-| ]+\|\n)"
        r"(?:\|[^\n]+\|\n)*",
        re.MULTILINE,
    )
    if not pattern.search(content):
        raise ValueError("Could not find experimental results table in README.")
    return pattern.sub(new_table + "\n", content, count=1)


def resolve_entity(api, entity: str | None) -> str:
    """Match ``configs/logger/wandb.yaml``: explicit entity, else env, else login default."""
    if entity:
        return entity
    env_entity = os.environ.get("WANDB_ENTITY")
    if env_entity:
        return env_entity
    return api.default_entity


def list_entity_projects(api, entity: str) -> list[str]:
    try:
        return sorted(project.name for project in api.projects(entity))
    except Exception:
        return []


def format_project_not_found_error(
    *,
    entity: str,
    project: str,
    available_projects: list[str],
) -> str:
    lines = [
        f"W&B project not found: {entity}/{project}",
        "",
        "The script defaults to your logged-in entity (or WANDB_ENTITY), not the "
        f"canonical team project {CANONICAL_ENTITY}/{DEFAULT_PROJECT}.",
    ]
    if available_projects:
        lines.append(
            f"Projects visible under '{entity}': {', '.join(available_projects)}"
        )
    else:
        lines.append(f"No projects visible under entity '{entity}'.")
    lines.extend(
        [
            "",
            "Options:",
            f"  • Shared template benchmarks:  --entity {CANONICAL_ENTITY}  "
            f"(requires team access)",
            f"  • Your own runs:               --entity {entity} --project <name>",
            "  • Set default entity:          export WANDB_ENTITY=your-team",
        ]
    )
    return "\n".join(lines)


def fetch_template_runs(
    *,
    entity: str | None,
    project: str,
    tag: str,
    states: tuple[str, ...] = ("finished",),
):
    import wandb

    api = wandb.Api()
    resolved_entity = resolve_entity(api, entity)
    path = f"{resolved_entity}/{project}"
    filters: dict = {"tags": {"$in": [tag]}}
    if len(states) == 1:
        filters["state"] = states[0]
    elif states:
        filters["state"] = {"$in": list(states)}

    try:
        runs = list(api.runs(path, filters=filters, order="-created_at"))
    except Exception as exc:
        message = str(exc)
        if "Could not find project" in message or "404" in message:
            available = list_entity_projects(api, resolved_entity)
            raise RuntimeError(
                format_project_not_found_error(
                    entity=resolved_entity,
                    project=project,
                    available_projects=available,
                )
            ) from exc
        raise

    return runs, resolved_entity


def group_rows_by_algo(
    runs,
    registry: list[ExperimentSpec],
) -> dict[str, list[ResultRow]]:
    grouped: dict[str, list[ResultRow]] = {pkg: [] for pkg in set(ALGO_TARGET_PREFIXES.values())}
    for run in runs:
        row = parse_run(run, registry)
        if row is None:
            continue
        config = dict(run.config)
        algo_cfg = config.get("algorithm") or {}
        package = algo_package_from_target(algo_cfg.get("_target_"))
        if package is not None:
            grouped[package].append(row)
    return grouped


def update_readme(path: Path, rows: list[ResultRow], *, dry_run: bool) -> bool:
    content = path.read_text(encoding="utf-8")
    new_table = build_table(rows)
    updated = replace_results_table(content, new_table)
    if updated == content:
        return False
    if not dry_run:
        path.write_text(updated, encoding="utf-8")
    return True


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Populate algorithm README experimental-results tables from W&B.",
    )
    parser.add_argument(
        "--entity",
        default=None,
        help=(
            "W&B entity (team/user). Defaults to WANDB_ENTITY or the logged-in "
            f"user's default entity. Canonical shared benchmarks live under "
            f"{CANONICAL_ENTITY}."
        ),
    )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help="W&B project name.",
    )
    parser.add_argument(
        "--tag",
        default="template",
        help="Only include runs with this W&B tag (default: template).",
    )
    parser.add_argument(
        "--algo",
        choices=sorted(set(ALGO_TARGET_PREFIXES.values())),
        nargs="*",
        help="Limit updates to these algorithm packages (default: all).",
    )
    parser.add_argument(
        "--include-running",
        action="store_true",
        help="Also include runs that are still running (default: finished only).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print tables without modifying README files.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    registry = load_experiment_registry()

    states: tuple[str, ...]
    if args.include_running:
        states = ("finished", "running")
    else:
        states = ("finished",)

    try:
        runs, resolved_entity = fetch_template_runs(
            entity=args.entity,
            project=args.project,
            tag=args.tag,
            states=states,
        )
    except Exception as exc:
        print(f"Failed to fetch W&B runs:\n{exc}", file=sys.stderr)
        if "W&B project not found" not in str(exc):
            print(
                "Authenticate with `wandb login` or set WANDB_API_KEY, then retry.",
                file=sys.stderr,
            )
        return 1

    grouped = group_rows_by_algo(runs, registry)
    targets = args.algo or sorted(grouped.keys())

    print(
        f"Fetched {len(runs)} run(s) tagged '{args.tag}' "
        f"from {resolved_entity}/{args.project}."
    )

    changed = 0
    for algo in targets:
        readme = algo_readme_path(algo)
        if not readme.exists():
            print(f"Skipping missing README: {readme}", file=sys.stderr)
            continue
        rows = grouped.get(algo, [])
        print(f"\n## {algo} ({len(rows)} run(s))")
        table = build_table(rows)
        print(table)
        if update_readme(readme, rows, dry_run=args.dry_run):
            changed += 1
            action = "Would update" if args.dry_run else "Updated"
            print(f"{action}: {readme.relative_to(REPO_ROOT)}")

    if args.dry_run:
        print(f"\nDry run complete — {changed} README(s) would change.")
    else:
        print(f"\nDone — {changed} README(s) updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
