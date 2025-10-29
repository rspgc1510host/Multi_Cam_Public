# Lunar Rover Autonomy Toolkit

This repository hosts the scaffolding for a Python-based experimentation
environment focused on lunar rover autonomy. The project is organised into
modular subsystems that cover perception, terrain understanding, pathfinding,
and benchmarking so new algorithms can be added predictably as the platform
matures.

## Repository Layout

```
lunar_rover/
├── vision/         # Sensor ingestion and perception pipelines
├── terrain/        # Heightmap parsing and terrain feature extraction
├── pathfinding/    # Route planning algorithms and utilities
├── benchmarking/   # Scenario runners and performance tracking helpers
└── utils/          # Cross-cutting helpers such as configuration loaders
```

Additional folders:

- `tests/` contains pytest suites that exercise the scaffolding and provide
  working examples for new contributors.
- `Makefile` defines developer tasks for installing dependencies, running the
  linters, and executing the test suite.

## Getting Started

### Requirements

- Python 3.11 or newer
- A C++ toolchain for compiling native wheels (required by `opencv-python`)

### Environment setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e ".[dev]"
```

The last command installs the package in editable mode alongside the optional
`dev` dependencies declared in `pyproject.toml` (black, flake8, isort, pytest,
etc.).

Alternatively, the same steps can be executed via the provided make targets:

```bash
make install
make lint
make test
```

## Development Workflow

- **Vision (`lunar_rover.vision`)** hosts reusable pipelines that chain
  pre-processing, inference, and post-processing stages for on-board cameras.
- **Terrain (`lunar_rover.terrain`)** provides helpers for parsing height maps
  and computing slope or traversability metrics.
- **Pathfinding (`lunar_rover.pathfinding`)** implements grid-based planners and
  defines the base interfaces future planners should adopt.
- **Benchmarking (`lunar_rover.benchmarking`)** supplies light-weight scenario
  orchestration to compare competing algorithms.
- **Utils (`lunar_rover.utils`)** centralises configuration loading and other
  shared helpers.

## Linting & Formatting

The project ships with configuration for:

- **black** for code formatting
- **isort** for import ordering
- **flake8** for lightweight static analysis

Run them together using `make check`, or individually through their respective
make targets.

## Testing

Pytest is the default test runner. To execute all tests:

```bash
make test
```

Test discovery is configured in `pyproject.toml` to target the `tests/`
directory.

## Next Steps

This skeleton is intentionally lightweight. Suggested roadmap items include:

1. Integrate real sensor simulation datasets and expand the terrain loaders.
2. Add probabilistic planners (e.g., RRT*, D* Lite) and benchmarking harnesses.
3. Provide dataset download scripts and continuous integration workflows.
4. Expand utilities for logging, configuration validation, and telemetry.
