from __future__ import annotations

import os

from .base import RunData


def _algo_label(config: dict) -> str:
    """Derive a short algorithm label from a W&B run config.

    For Dreamer variants the inner model class distinguishes them even though
    all share the same outer DreamerAlgorithm _target_.
    """
    algo_cfg = config.get("algorithm") or {}
    dreamer_cfg = algo_cfg.get("dreamer_config") or {}
    inner = dreamer_cfg.get("_target_", "")
    if inner:
        return inner.rsplit(".", 1)[-1]
    outer = algo_cfg.get("_target_", "")
    return outer.rsplit(".", 1)[-1] if outer else "unknown"


class WandbDataSource:
    def __init__(self, *, tag: str, entity: str | None, project: str) -> None:
        self.tag = tag
        self.entity = entity
        self.project = project

    def fetch_runs(
        self,
        *,
        algos: list[str] | None,
        envs: list[str] | None,
        metric: str,
        fetch_history: bool = False,
        history_metric: str = "episode/score",
    ) -> list[RunData]:
        import wandb

        api = wandb.Api()
        path = f"{self.entity}/{self.project}"

        filters: dict = {"state": "finished"}
        if self.tag:
            filters["tags"] = {"$in": [self.tag]}

        raw_runs = list(
            api.runs(path, filters=filters, order="-created_at")
        )
        tag_info = f"tagged '{self.tag}' " if self.tag else ""
        print(f"Fetched {len(raw_runs)} finished run(s) {tag_info}from {path}.")

        env_set = set(envs) if envs else None
        algo_set = set(algos) if algos else None
        results: list[RunData] = []
        n_skipped = {"metric": 0, "env": 0, "algo": 0}

        for run in raw_runs:
            config = dict(run.config)
            env_name = (config.get("environment") or {}).get("name")

            if env_set and env_name not in env_set:
                n_skipped["env"] += 1
                continue

            label = _algo_label(config)
            if algo_set and label not in algo_set:
                n_skipped["algo"] += 1
                continue

            score = run.summary.get(metric)
            if score is None:
                n_skipped["metric"] += 1
                continue

            seed = (config.get("trainer") or {}).get("seed")
            history: list[tuple[int, float]] = []

            if fetch_history:
                history = self._fetch_history(run, history_metric)

            results.append(
                RunData(
                    run_id=run.id,
                    algo_label=label,
                    env_name=env_name or "unknown",
                    seed=int(seed) if seed is not None else None,
                    summary_score=float(score),
                    history=history,
                )
            )

        for reason, count in n_skipped.items():
            if count:
                print(f"  Skipped {count} run(s): {reason}.")

        return results

    def _fetch_history(self, run, metric: str) -> list[tuple[int, float]]:
        try:
            rows = run.history(keys=[metric], pandas=False, samples=10_000)
            return [
                (int(row["_step"]), float(row[metric]))
                for row in rows
                if row.get(metric) is not None
            ]
        except Exception as exc:
            print(f"  Warning: could not fetch history for {run.id}: {exc}")
            return []
