"""Source-agnostic directed graph types for NeuroFlyNav (development spike, M0).

This module defines a minimal directed, weighted graph and a loader protocol so a
synthetic graph (M0) and a real connectome subgraph (M1 onward) can be used
interchangeably. Nothing here encodes a biological claim: edge weights, synaptic
signs and dynamics are deliberately out of scope for the spike and remain open
scientific decisions (see ops DECISIONS D-026).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Protocol


class GraphError(Exception):
    """Raised when a graph is structurally invalid."""


class NodeRole(Enum):
    """Coarse functional role of a node, used only to wire inputs and outputs."""

    SENSORY = "sensory"
    INTERNAL = "internal"
    MOTOR = "motor"


@dataclass(frozen=True)
class Edge:
    """A directed, weighted edge between two node indices."""

    source: int
    target: int
    weight: float


@dataclass(frozen=True)
class Graph:
    """A directed, weighted graph with one coarse role per node."""

    num_nodes: int
    edges: tuple[Edge, ...]
    roles: tuple[NodeRole, ...]

    def __post_init__(self) -> None:
        if self.num_nodes <= 0:
            raise GraphError("A graph needs at least one node")
        if len(self.roles) != self.num_nodes:
            raise GraphError("The number of roles must match the number of nodes")
        for edge in self.edges:
            if not 0 <= edge.source < self.num_nodes:
                raise GraphError(f"Edge source out of range: {edge.source}")
            if not 0 <= edge.target < self.num_nodes:
                raise GraphError(f"Edge target out of range: {edge.target}")
            if not isfinite(edge.weight):
                raise GraphError("Edge weights must be finite")

    def nodes_with_role(self, role: NodeRole) -> tuple[int, ...]:
        """Return node indices with the given role, in ascending order."""
        return tuple(
            index for index, node_role in enumerate(self.roles) if node_role == role
        )

    def out_neighbors(self, node: int) -> tuple[tuple[int, float], ...]:
        """Return (target, weight) pairs leaving a node, in edge order."""
        if not 0 <= node < self.num_nodes:
            raise GraphError(f"Node out of range: {node}")
        return tuple(
            (edge.target, edge.weight) for edge in self.edges if edge.source == node
        )


class GraphLoader(Protocol):
    """Anything that can produce a Graph.

    The synthetic loader implements this for M0; a real connectome loader will
    implement the same protocol at M1, so the rest of the code does not change.
    """

    def load(self) -> Graph: ...
