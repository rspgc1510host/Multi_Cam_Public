import json
from pathlib import Path

import numpy as np
import pytest

from lunar_rover.benchmarking import BenchmarkScenario, BenchmarkSuite
from lunar_rover.pathfinding import GridPlanner
from lunar_rover.terrain import load_heightmap
from lunar_rover.utils import ConfigLoader, merge_dicts
from lunar_rover.vision import VisionPipeline


def test_vision_pipeline_executes() -> None:
    frame = np.zeros((4, 4), dtype=np.float32)

    pipeline = VisionPipeline(
        preprocessors=(lambda f: f + 1,),
        inference=lambda f: (f.mean(axis=0),),
        postprocessors=(lambda detections: tuple(d * 2 for d in detections),),
    )

    output = pipeline.run(frame)
    assert len(output) == 1
    assert pytest.approx(float(output[0][0])) == 2.0


def test_load_heightmap_numpy(tmp_path: Path) -> None:
    heightmap = np.arange(9, dtype=np.float32).reshape(3, 3)
    file_path = tmp_path / "terrain.npy"
    np.save(file_path, heightmap)

    terrain = load_heightmap(file_path, resolution=0.5)
    assert terrain.shape == (3, 3)
    assert pytest.approx(float(terrain.elevation[1, 1])) == 4.0


def test_grid_planner_finds_path() -> None:
    traversal_cost = np.ones((3, 3), dtype=float)
    planner = GridPlanner(traversal_cost)
    path = planner.plan((0, 0), (2, 2))

    assert path.waypoints[0] == (0, 0)
    assert path.waypoints[-1] == (2, 2)
    assert path.cost > 0


def test_benchmark_suite_runs() -> None:
    counter = {"calls": 0}

    def execute() -> dict[str, float]:
        counter["calls"] += 1
        return {"time_ms": 1.0}

    scenario = BenchmarkScenario(name="mock", execute=execute)
    suite = BenchmarkSuite([scenario])
    results = suite.run(repeats=2)

    assert len(results) == 2
    aggregated = BenchmarkSuite.aggregate(results)
    assert aggregated["time_ms"] == pytest.approx(1.0)
    assert counter["calls"] == 2


def test_config_loader_and_merge(tmp_path: Path) -> None:
    loader = ConfigLoader(base_path=tmp_path)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"a": 1, "shared": {"threshold": 0.5}}))

    loaded = loader.load("config.json")
    merged = merge_dicts(loaded, {"shared": {"threshold": 1.0, "mode": "test"}})

    assert merged["a"] == 1
    assert merged["shared"]["threshold"] == 1.0
    assert merged["shared"]["mode"] == "test"
