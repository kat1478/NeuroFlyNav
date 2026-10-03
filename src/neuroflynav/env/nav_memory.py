"""A discrete-grid navigation task with a *hidden goal cue* (development spike, M0+).

The goal direction is shown only for the first ``cue_steps`` steps, then hidden. A
reactive policy cannot solve it after the cue disappears; a policy with memory
(a recurrent substrate) can. This makes the task able to *discriminate* between
topologies, which the trivial M0 task could not. It is still a toy, non-scientific
spike task; nothing here is a frozen scientific benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from random import Random

import numpy as np
from numpy.typing import NDArray

from neuroflynav.env.nav import Action, NavError

FloatArray = NDArray[np.float64]

#: Length of the feature vector produced by :func:`encode_observation`.
FEATURE_DIM = 6

_MOVES: dict[Action, tuple[int, int]] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}


@dataclass(frozen=True)
class MemoryGridNavConfig:
    """Configuration for the hidden-cue grid navigation task."""

    size: int = 7
    max_steps: int = 60
    cue_steps: int = 3

    def __post_init__(self) -> None:
        if self.size < 2:
            raise ValueError("size must be at least 2")
        if self.max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if self.cue_steps < 0:
            raise ValueError("cue_steps must not be negative")


@dataclass(frozen=True)
class MemoryGridObservation:
    """Agent cell, the goal offset *while the cue is visible*, and the cue flag."""

    agent: tuple[int, int]
    cue_delta: tuple[int, int]
    cue_visible: bool


class MemoryGridNavEnv:
    """A grid where the goal direction is only visible for the first few steps."""

    def __init__(self, config: MemoryGridNavConfig | None = None) -> None:
        self._config = config or MemoryGridNavConfig()
        self._rng = Random()
        self._agent: tuple[int, int] = (0, 0)
        self._goal: tuple[int, int] = (0, 0)
        self._steps = 0
        self._started = False

    @property
    def config(self) -> MemoryGridNavConfig:
        return self._config

    def reset(self, seed: int) -> tuple[MemoryGridObservation, dict[str, object]]:
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
    ) -> tuple[MemoryGridObservation, float, bool, bool, dict[str, object]]:
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

    def optimal_action(self) -> Action:
        """Greedy privileged move toward the true goal (used only to teach)."""
        if not self._started:
            raise NavError("reset() must be called before optimal_action()")
        row_delta = self._goal[0] - self._agent[0]
        column_delta = self._goal[1] - self._agent[1]
        if row_delta < 0:
            return Action.UP
        if row_delta > 0:
            return Action.DOWN
        if column_delta < 0:
            return Action.LEFT
        return Action.RIGHT

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

    def _observation(self) -> MemoryGridObservation:
        visible = self._steps < self._config.cue_steps
        if visible:
            delta = (
                self._goal[0] - self._agent[0],
                self._goal[1] - self._agent[1],
            )
        else:
            delta = (0, 0)
        return MemoryGridObservation(
            agent=self._agent, cue_delta=delta, cue_visible=visible
        )


def encode_observation(observation: MemoryGridObservation, size: int) -> FloatArray:
    """Encode an observation as a fixed-length feature vector (length ``FEATURE_DIM``)."""
    denominator = float(max(size - 1, 1))
    cue_row = observation.cue_delta[0] / denominator if observation.cue_visible else 0.0
    cue_col = observation.cue_delta[1] / denominator if observation.cue_visible else 0.0
    return np.array(
        [
            1.0,
            observation.agent[0] / denominator,
            observation.agent[1] / denominator,
            cue_row,
            cue_col,
            1.0 if observation.cue_visible else 0.0,
        ],
        dtype=np.float64,
    )
