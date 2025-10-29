"""Benchmark orchestration primitives."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Callable, Iterable, List, Mapping


MetricMapping = Mapping[str, float]


@dataclass(frozen=True)
class BenchmarkResult:
    """Container for a single benchmark execution."""

    scenario: str
    metrics: MetricMapping


@dataclass
class BenchmarkScenario:
    """Callable benchmark scenario with optional warm-up logic."""

    name: str
    execute: Callable[[], MetricMapping]
    warmup: Callable[[], None] | None = None

    def run(self) -> BenchmarkResult:
        if self.warmup is not None:
            self.warmup()
        return BenchmarkResult(scenario=self.name, metrics=self.execute())


class BenchmarkSuite:
    """Execute a collection of :class:`BenchmarkScenario` objects."""

    def __init__(self, scenarios: Iterable[BenchmarkScenario] | None = None):
        self._scenarios: list[BenchmarkScenario] = list(scenarios or [])

    def add(self, scenario: BenchmarkScenario) -> None:
        self._scenarios.append(scenario)

    def run(self, repeats: int = 1) -> list[BenchmarkResult]:
        results: list[BenchmarkResult] = []
        for scenario in self._scenarios:
            for _ in range(repeats):
                results.append(scenario.run())
        return results

    @staticmethod
    def aggregate(results: Iterable[BenchmarkResult]) -> dict[str, float]:
        """Aggregate metrics using arithmetic mean."""

        grouped: dict[str, list[float]] = {}
        for result in results:
            for key, value in result.metrics.items():
                grouped.setdefault(key, []).append(value)
        return {key: mean(values) for key, values in grouped.items()}
