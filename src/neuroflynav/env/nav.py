"""A tiny discrete-grid navigation-to-target environment (development spike, M0).

Standard library only. The API mirrors Gymnasium (``reset`` / ``step`` returning an
observation, reward, terminated, truncated and info) so it can be wrapped with
Gymnasium later, but it deliberately depends on nothing external. The task is
intentionally trivial: it exists to close the agent/environment loop, not to be a
scientific navigation benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from random import Random


class NavError(Exception):
    """Raised when the environment is used incorrectly."""


class Action(IntEnum):
    """The four grid moves."""

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


_MOVES: dict[Action, tuple[int, int]] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


@dataclass(frozen=True)
class GridNavConfig:
    """Configuration for the grid navigation environment."""

    size: int = 5
    max_steps: int = 50

    def __post_init__(self) -> None:
        if self.size < 2:
            raise ValueError("Grid size must be at least 2")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")


@dataclass(frozen=True)
class GridObservation:
    """What the agent sees: its cell, the goal cell and their signed offset."""

    agent: tuple[int, int]
    goal: tuple[int, int]
    goal_delta: tuple[int, int]


class GridNavEnv:
    """A deterministic-per-seed grid where an agent walks toward a goal cell."""

    def __init__(self, config: GridNavConfig | None = None) -> None:
        self._config = config or GridNavConfig()
        self._rng = Random()
        self._agent: tuple[int, int] = (0, 0)
        self._goal: tuple[int, int] = (0, 0)
        self._steps = 0
        self._started = False

    @property
    def config(self) -> GridNavConfig:
        return self._config

    def reset(self, seed: int) -> tuple[GridObservation, dict[str, object]]:
        """Place the agent and a distinct goal; return the first observation."""
        self._rng = Random(seed)
        self._agent = self._random_cell()
        goal = self._random_cell()
        while goal == self._agent:
            goal = self._random_cell()
        self._goal = goal
        self._steps = 0
        self._started = True
        return self._observation(), {"steps": self._steps}

    def step(
        self, action: Action | int
    ) -> tuple[GridObservation, float, bool, bool, dict[str, object]]:
        """Apply one move and return (observation, reward, terminated, truncated, info)."""
        if not self._started:
            raise NavError("reset() must be called before step()")
        move = _MOVES[Action(action)]
        previous_distance = self._manhattan()
        self._agent = self._clamp((self._agent[0] + move[0], self._agent[1] + move[1]))
        self._steps += 1
        distance = self._manhattan()
        reached = self._agent == self._goal
        truncated = (not reached) and self._steps >= self._config.max_steps
        reward = self._reward(previous_distance, distance, reached=reached)
        info: dict[str, object] = {"steps": self._steps, "distance": distance}
        return self._observation(), reward, reached, truncated, info

    def _random_cell(self) -> tuple[int, int]:
        size = self._config.size
        return (self._rng.randrange(size), self._rng.randrange(size))

    def _clamp(self, cell: tuple[int, int]) -> tuple[int, int]:
        size = self._config.size
        row = min(max(cell[0], 0), size - 1)
        column = min(max(cell[1], 0), size - 1)
        return (row, column)

    def _manhattan(self) -> int:
        return abs(self._agent[0] - self._goal[0]) + abs(self._agent[1] - self._goal[1])

    def _reward(self, previous_distance: int, distance: int, *, reached: bool) -> float:
        if reached:
            return 1.0
        return 0.1 * float(previous_distance - distance) - 0.01

    def _observation(self) -> GridObservation:
        delta = (
            self._goal[0] - self._agent[0],
            self._goal[1] - self._agent[1],
        )
        return GridObservation(agent=self._agent, goal=self._goal, goal_delta=delta)
