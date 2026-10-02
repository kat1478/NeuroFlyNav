"""Tests for the synthetic graph and the source-agnostic graph interface (M0 spike)."""

from __future__ import annotations

import pytest

from neuroflynav.graph.interface import Edge, Graph, GraphError, NodeRole
from neuroflynav.graph.synthetic import (
    SyntheticCXConfig,
    SyntheticCXGraphLoader,
    generate_synthetic_cx_graph,
)


def test_synthetic_graph_is_deterministic_for_a_seed() -> None:
    assert generate_synthetic_cx_graph(seed=7) == generate_synthetic_cx_graph(seed=7)


def test_different_seeds_change_long_range_edges() -> None:
    assert generate_synthetic_cx_graph(seed=1) != generate_synthetic_cx_graph(seed=2)


def test_roles_cover_every_node_and_include_inputs_and_outputs() -> None:
    graph = generate_synthetic_cx_graph(seed=0)
    assert len(graph.roles) == graph.num_nodes
    assert graph.nodes_with_role(NodeRole.SENSORY)
    assert graph.nodes_with_role(NodeRole.MOTOR)


def test_loader_matches_direct_generation() -> None:
    config = SyntheticCXConfig(num_nodes=16, num_sensory=2, num_motor=2)
    loader = SyntheticCXGraphLoader(seed=3, config=config)
    assert loader.load() == generate_synthetic_cx_graph(seed=3, config=config)


def test_out_neighbors_reports_ring_connection() -> None:
    graph = generate_synthetic_cx_graph(seed=0)
    targets = {target for target, _ in graph.out_neighbors(0)}
    assert 1 in targets


def test_graph_validates_edge_indices() -> None:
    with pytest.raises(GraphError):
        Graph(
            num_nodes=2,
            edges=(Edge(0, 5, 1.0),),
            roles=(NodeRole.INTERNAL, NodeRole.MOTOR),
        )


def test_graph_validates_role_count() -> None:
    with pytest.raises(GraphError):
        Graph(num_nodes=2, edges=(), roles=(NodeRole.INTERNAL,))


def test_config_rejects_too_many_roles() -> None:
    with pytest.raises(ValueError, match="too small"):
        SyntheticCXConfig(num_nodes=3, num_sensory=2, num_motor=2)
