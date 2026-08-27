"""Repository path helpers."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Absolute path of the template repository root."""
    here = Path(__file__).resolve()
    for candidate in (here.parent.parent, *here.parents):
        if (candidate / "src" / "train.py").is_file() and (candidate / "configs").is_dir():
            return candidate
    raise RuntimeError("could not locate repository root")


_STUDY_DIRS = {
    "profiling": "scripts/profiling",
    "envthroughput": "scripts/envbench",
}


def study_dir(name: str) -> Path:
    """Directory holding configs, scripts and README for one measured study."""
    rel = _STUDY_DIRS.get(name, f"scripts/{name}")
    path = repo_root() / rel
    if not path.is_dir():
        raise FileNotFoundError(f"no study directory at {path}")
    return path


def results_dir(study: str) -> Path:
    """Aggregated sweep output under ``logs/<study>/results/`` (gitignored)."""
    path = repo_root() / "logs" / study / "results"
    path.mkdir(parents=True, exist_ok=True)
    return path
