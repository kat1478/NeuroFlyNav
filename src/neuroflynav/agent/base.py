"""The policy protocol shared by navigation agents (development spike, M0)."""

from __future__ import annotations

from typing import Protocol

from neuroflynav.env.nav import Action, GridObservation


class Policy(Protocol):
    """Anything that maps an observation to an action.

    RandomAgent, FlyConnectome and ShuffledFly implement this protocol in M0.
    """

    def act(self, observation: GridObservation) -> Action: ...
