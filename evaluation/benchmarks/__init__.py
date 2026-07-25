from .atari100k import ATARI100K
from .base import BenchmarkSpec

_REGISTRY: dict[str, BenchmarkSpec] = {
    "atari100k": ATARI100K,
}


def get_benchmark(name: str) -> BenchmarkSpec:
    if name not in _REGISTRY:
        raise ValueError(f"Unknown benchmark '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def register_benchmark(spec: BenchmarkSpec) -> None:
    _REGISTRY[spec.name] = spec


__all__ = ["BenchmarkSpec", "get_benchmark", "register_benchmark", "ATARI100K"]
