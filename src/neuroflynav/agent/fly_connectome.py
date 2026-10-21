"""Graph-driven placeholder policy for the synthetic M0 development spike.

This is a computational routing rule over a synthetic graph, not a biological
model. Sensory/motor ordering and shortest-path scoring are arbitrary M0 choices;
weights, synaptic signs and neural dynamics remain unresolved scientific decisions.
"""

from __future__ import annotations

from collections import deque

from neuroflynav.agent.base import Policy
from neuroflynav.env.nav import Action, GridObservation
from neuroflynav.graph.interface import Graph, NodeRole

_DIRECTION_ORDER = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)


class FlyConnectome(Policy):
    """Choose an action by the shortest directed route in a provided graph.

    The first four sensory-role nodes and first four motor-role nodes, in ascending
    node-index order, stand for UP, DOWN, LEFT and RIGHT. For each direction that
    reduces the observed goal offset, the policy scores the shortest directed path
    from its corresponding sensory node to motor node as ``1 / (1 + hops)``.
    Unreachable routes score zero; ties use the fixed direction order. If no
    goal-directed route is reachable, the first goal-directed direction is used
    as a simple fallback.

    The convention is deliberately arbitrary and exists only to exercise graph
    consumption through ``Policy.act`` on the synthetic M0 fixture.
    """

    def __init__(self, graph: Graph) -> None:
        self._graph = graph
        sensory = graph.nodes_with_role(NodeRole.SENSORY)
        motor = graph.nodes_with_role(NodeRole.MOTOR)
        if len(sensory) < len(_DIRECTION_ORDER) or len(motor) < len(_DIRECTION_ORDER):
            raise ValueError(
                "FlyConnectome requires at least four sensory and motor nodes"
            )
        self._sensory = sensory[: len(_DIRECTION_ORDER)]
        self._motor = motor[: len(_DIRECTION_ORDER)]
        self._route_scores = tuple(
            self._shortest_path_score(source, target)
            for source, target in zip(self._sensory, self._motor, strict=True)
        )

    def act(self, observation: GridObservation) -> Action:
        """Return the best graph-routed move that points toward the goal."""
        row_delta, column_delta = observation.goal_delta
        desired = (
            row_delta < 0,
            row_delta > 0,
            column_delta < 0,
            column_delta > 0,
        )
        scores = tuple(
            route_score if is_desired else 0.0
            for route_score, is_desired in zip(self._route_scores, desired, strict=True)
        )
        candidates = [index for index, is_desired in enumerate(desired) if is_desired]
        if not candidates:
            return _DIRECTION_ORDER[0]
        best_index = max(candidates, key=scores.__getitem__)
        if scores[best_index] == 0.0:
            best_index = candidates[0]
        return _DIRECTION_ORDER[best_index]

    def _shortest_path_score(self, source: int, target: int) -> float:
        if source == target:
            return 1.0
        distances = {source: 0}
        queue = deque([source])
        while queue:
            node = queue.popleft()
            for neighbor, _weight in self._graph.out_neighbors(node):
                if neighbor in distances:
                    continue
                distance = distances[node] + 1
                if neighbor == target:
                    return 1.0 / (1.0 + distance)
                distances[neighbor] = distance
                queue.append(neighbor)
        return 0.0
