"""Lunar Rover exploration toolkit.

This package provides modular building blocks for computer vision, terrain
understanding, pathfinding, and benchmarking workflows used to prototype lunar
rover autonomy.
"""

from importlib import metadata

try:  # pragma: no cover - best effort metadata lookup
    __version__ = metadata.version("lunar-rover")
except metadata.PackageNotFoundError:  # pragma: no cover - local development
    __version__ = "0.0.0"

__all__ = ["__version__"]
