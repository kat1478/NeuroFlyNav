"""A random navigation policy (development spike, M0).

This is the M0 baseline that closes the agent/environment loop. It ignores the
observation and makes no scientific claim.
"""

from __future__ import annotations

from random import Random

from neuroflynav.env.nav import Action, GridObservation


class RandomAgent:
    """Selects a uniformly random action, ignoring the observation."""

    def __init__(self, seed: int) -> None:
        self._rng = Random(seed)
        self._actions: tuple[Action, ...] = tuple(Action)

    def act(self, observation: GridObservation) -> Action:
        del observation
        return self._rng.choice(self._actions)
