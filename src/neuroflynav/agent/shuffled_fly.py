"""Degree-preserving shuffled-graph control for the synthetic M0 spike."""

from __future__ import annotations

from neuroflynav.agent.base import Policy
from neuroflynav.agent.fly_connectome import FlyConnectome
from neuroflynav.env.nav import Action, GridObservation
from neuroflynav.graph.interface import Edge, Graph
from neuroflynav.graph.shuffle import shuffle_graph


class ShuffledFly(Policy):
    """Apply the FlyConnectome policy to a seeded rewiring of a supplied graph.

    The directed double-edge swaps preserve each node's in-degree and out-degree,
    graph size, node roles, and the multiset of edge weights. They do not preserve
    higher-order motifs, path lengths, role-to-role connectivity or biological
    properties. Those preserved/unpreserved features define this synthetic M0
    computational control only; this is not a frozen scientific control design.
    """

    def __init__(self, graph: Graph, seed: int) -> None:
        self.graph: Graph = shuffle_graph(graph, seed=seed)
        self.seed = seed
        self._policy = FlyConnectome(self.graph)

    @property
    def edges(self) -> tuple[Edge, ...]:
        """Return the edges of the graph consumed by this policy."""
        return self.graph.edges

    def act(self, observation: GridObservation) -> Action:
        """Choose an action using the shuffled topology and shared policy rule."""
        return self._policy.act(observation)
