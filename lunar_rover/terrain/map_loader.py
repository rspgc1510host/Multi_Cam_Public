"""Terrain data structures and loaders."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass(slots=True)
class TerrainMap:
    """Simple container for elevation data.

    The map stores a 2D height-field along with metadata describing the spatial
    resolution of each cell in metres. Utility methods provide common
    pre-computations used by navigation algorithms, such as slope estimation and
    traversability masks.
    """

    elevation: np.ndarray
    resolution: float
    frame: str = "lunar_local_level"

    def __post_init__(self) -> None:
        if self.elevation.ndim != 2:
            raise ValueError("TerrainMap expects a 2D elevation grid.")
        if self.resolution <= 0:
            raise ValueError("Resolution must be a positive float.")

    @property
    def shape(self) -> tuple[int, int]:
        return self.elevation.shape

    def gradient(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute terrain gradients along the x and y axes."""

        gx, gy = np.gradient(self.elevation, self.resolution)
        return gx, gy

    def slope_magnitude(self) -> np.ndarray:
        """Return the per-cell slope magnitude in radians."""

        gx, gy = self.gradient()
        return np.hypot(gx, gy)

    def traversability_mask(self, max_slope_radians: float) -> np.ndarray:
        """Binary mask indicating terrains with slope below ``max_slope_radians``."""

        return self.slope_magnitude() <= max_slope_radians


def load_heightmap(
    path: str | Path,
    *,
    resolution: float,
    loader: str | None = None,
    dtype: type[np.floating] = np.float32,
) -> TerrainMap:
    """Load elevation data from disk and wrap it in a :class:`TerrainMap`.

    Args:
        path: Location of the height map. ``.npy`` files are loaded with
            :func:`numpy.load` while other text-based formats fall back to
            :func:`numpy.loadtxt`.
        resolution: Spatial resolution of each cell in metres.
        loader: Optional override for the loading strategy (``"npy"`` or
            ``"text"``).
        dtype: Target floating point data type.
    """

    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(file_path)

    strategy = loader or ("npy" if file_path.suffix == ".npy" else "text")
    if strategy == "npy":
        elevation = np.load(file_path)
    elif strategy == "text":
        elevation = np.loadtxt(file_path)
    else:
        raise ValueError(
            "Unknown loader strategy. Expected one of {'npy', 'text'} or None."
        )

    elevation = np.asarray(elevation, dtype=dtype)
    return TerrainMap(elevation=elevation, resolution=resolution)
