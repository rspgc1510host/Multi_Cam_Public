"""Path planning utilities for grid-based environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from heapq import heappop, heappush
from typing import Iterable, Iterator, Sequence, Tuple

import numpy as np

Coordinate = Tuple[int, int]


@dataclass(frozen=True)
class GridPath:
    """Container for a discovered path."""

    waypoints: Tuple[Coordinate, ...]
    cost: float

    def __iter__(self) -> Iterator[Coordinate]:
        return iter(self.waypoints)


class PathPlanner(ABC):
    """Base interface for all path planners."""

    @abstractmethod
    def plan(self, start: Coordinate, goal: Coordinate) -> GridPath:
        """Return a path between ``start`` and ``goal`` coordinates."""


class GridPlanner(PathPlanner):
    """A* planner operating on 4-connected grids.

    The planner treats large traversal costs as soft obstacles. Infinite values
    are considered impassable cells.
    """

    def __init__(self, traversal_cost: np.ndarray):
        if traversal_cost.ndim != 2:
            raise ValueError("Traversal cost map must be 2D.")
        self.traversal_cost = traversal_cost.astype(float)
        self.height, self.width = traversal_cost.shape

    def plan(self, start: Coordinate, goal: Coordinate) -> GridPath:
        self._validate_coordinate(start)
        self._validate_coordinate(goal)

        open_set: list[tuple[float, Coordinate]] = []
        heappush(open_set, (0.0, start))

        came_from: dict[Coordinate, Coordinate] = {}
        g_score: dict[Coordinate, float] = {start: 0.0}

        while open_set:
            _, current = heappop(open_set)
            if current == goal:
                return self._reconstruct_path(came_from, current)

            for neighbor in self._neighbors(current):
                if np.isinf(self.traversal_cost[neighbor]):
                    continue

                tentative_g = g_score[current] + self.traversal_cost[neighbor]
                if tentative_g < g_score.get(neighbor, np.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self._heuristic(neighbor, goal)
                    heappush(open_set, (f, neighbor))

        raise RuntimeError("No path found between start and goal coordinates.")

    def _heuristic(self, start: Coordinate, goal: Coordinate) -> float:
        return float(abs(start[0] - goal[0]) + abs(start[1] - goal[1]))

    def _neighbors(self, coord: Coordinate) -> Iterable[Coordinate]:
        row, col = coord
        candidates = ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
        for nr, nc in candidates:
            if 0 <= nr < self.height and 0 <= nc < self.width:
                yield nr, nc

    def _reconstruct_path(self, came_from: dict[Coordinate, Coordinate], current: Coordinate) -> GridPath:
        waypoints = [current]
        total_cost = self.traversal_cost[current]
        while current in came_from:
            current = came_from[current]
            waypoints.append(current)
            total_cost += self.traversal_cost[current]
        waypoints.reverse()
        return GridPath(waypoints=tuple(waypoints), cost=float(total_cost))

    def _validate_coordinate(self, coord: Coordinate) -> None:
        row, col = coord
        if not (0 <= row < self.height and 0 <= col < self.width):
            raise ValueError(f"Coordinate {coord} falls outside the traversal grid.")
