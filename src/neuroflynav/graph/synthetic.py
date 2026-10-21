"""Deterministic synthetic CX-like graph generator (development spike, M0).

The structure here is a placeholder: a ring of locally connected nodes plus a few
sparse long-range edges, with a handful of sensory and motor nodes. It is NOT a
model of the real central complex and carries no biological claim. Its only jobs
are to be deterministic for a seed and to exercise the graph interface so a real
connectome loader can replace it at M1 behind the same `GraphLoader` protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from random import Random

from neuroflynav.graph.interface import Edge, Graph, NodeRole


@dataclass(frozen=True)
class SyntheticCXConfig:
    """Parameters for the synthetic CX-like graph."""

    num_nodes: int = 64
    num_sensory: int = 4
    num_motor: int = 4
    long_range_edges: int = 32
    ring_weight: float = 1.0
    long_range_weight: float = 0.5

    def __post_init__(self) -> None:
        if self.num_nodes < self.num_sensory + self.num_motor + 1:
            raise ValueError("num_nodes is too small for the requested roles")
        if self.num_sensory < 0 or self.num_motor < 0:
            raise ValueError("role counts must not be negative")
        if self.long_range_edges < 0:
            raise ValueError("long_range_edges must not be negative")


def _build_roles(config: SyntheticCXConfig) -> tuple[NodeRole, ...]:
    roles = [NodeRole.INTERNAL] * config.num_nodes
    for index in range(config.num_sensory):
        roles[index] = NodeRole.SENSORY
    for index in range(config.num_motor):
        roles[config.num_nodes - 1 - index] = NodeRole.MOTOR
    return tuple(roles)


def generate_synthetic_cx_graph(
    seed: int, config: SyntheticCXConfig | None = None
) -> Graph:
    """Return a deterministic synthetic CX-like graph for the given seed."""
    settings = config or SyntheticCXConfig()
    rng = Random(seed)
    edges: list[Edge] = []
    node_count = settings.num_nodes
    for node in range(node_count):
        neighbor = (node + 1) % node_count
        edges.append(Edge(node, neighbor, settings.ring_weight))
        edges.append(Edge(neighbor, node, settings.ring_weight))
    for _ in range(settings.long_range_edges):
        source = rng.randrange(node_count)
        target = rng.randrange(node_count)
        if source != target:
            edges.append(Edge(source, target, settings.long_range_weight))
    return Graph(
        num_nodes=node_count,
        edges=tuple(edges),
        roles=_build_roles(settings),
    )


@dataclass(frozen=True)
class SyntheticCXGraphLoader:
    """A `GraphLoader` producing a deterministic synthetic CX-like graph."""

    seed: int = 0
    config: SyntheticCXConfig = field(default_factory=SyntheticCXConfig)

    def load(self) -> Graph:
        return generate_synthetic_cx_graph(self.seed, self.config)
