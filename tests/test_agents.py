"""Focused tests for the graph-driven M0 policy and topology control."""

from __future__ import annotations

from collections import Counter

import pytest

from neuroflynav.agent.fly_connectome import FlyConnectome
from neuroflynav.agent.shuffled_fly import ShuffledFly
from neuroflynav.env.nav import Action, GridObservation
from neuroflynav.graph.interface import Edge, Graph, NodeRole
from neuroflynav.graph.shuffle import shuffle_graph
from neuroflynav.graph.synthetic import generate_synthetic_cx_graph


def _diagonal_graph(*, direct_up: bool) -> Graph:
    roles = (
        NodeRole.SENSORY,
        NodeRole.SENSORY,
        NodeRole.SENSORY,
        NodeRole.SENSORY,
        NodeRole.MOTOR,
        NodeRole.MOTOR,
        NodeRole.MOTOR,
        NodeRole.MOTOR,
        NodeRole.INTERNAL,
    )
    if direct_up:
        edges = (Edge(0, 4, 1.0), Edge(2, 8, 1.0), Edge(8, 6, 1.0))
    else:
        edges = (Edge(0, 8, 1.0), Edge(8, 4, 1.0), Edge(2, 6, 1.0))
    return Graph(num_nodes=len(roles), edges=edges, roles=roles)


def test_fly_connectome_uses_graph_routes_to_rank_diagonal_actions() -> None:
    observation = GridObservation(agent=(1, 1), goal=(0, 0), goal_delta=(-1, -1))
    assert FlyConnectome(_diagonal_graph(direct_up=True)).act(observation) is Action.UP
    assert (
        FlyConnectome(_diagonal_graph(direct_up=False)).act(observation) is Action.LEFT
    )


def test_fly_connectome_requires_four_ordered_inputs_and_outputs() -> None:
    graph = Graph(
        num_nodes=2,
        edges=(),
        roles=(NodeRole.SENSORY, NodeRole.MOTOR),
    )
    with pytest.raises(ValueError, match="four sensory and motor"):
        FlyConnectome(graph)


def test_shuffle_is_seeded_and_preserves_documented_properties() -> None:
    graph = generate_synthetic_cx_graph(seed=7)
    shuffled = shuffle_graph(graph, seed=31)
    assert shuffled == shuffle_graph(graph, seed=31)
    assert shuffled != graph
    assert shuffled.roles == graph.roles
    assert shuffled.num_nodes == graph.num_nodes
    assert len(shuffled.edges) == len(graph.edges)
    assert Counter(edge.source for edge in shuffled.edges) == Counter(
        edge.source for edge in graph.edges
    )
    assert Counter(edge.target for edge in shuffled.edges) == Counter(
        edge.target for edge in graph.edges
    )
    assert Counter(edge.weight for edge in shuffled.edges) == Counter(
        edge.weight for edge in graph.edges
    )
    assert all(edge.source != edge.target for edge in shuffled.edges)


def test_shuffle_rejects_negative_swap_budget() -> None:
    with pytest.raises(ValueError, match="must not be negative"):
        shuffle_graph(generate_synthetic_cx_graph(seed=0), seed=1, swaps_per_edge=-1)


def test_shuffled_fly_implements_policy_interface_and_uses_seeded_graph() -> None:
    graph = generate_synthetic_cx_graph(seed=4)
    first = ShuffledFly(graph, seed=9)
    second = ShuffledFly(graph, seed=9)
    observation = GridObservation(agent=(2, 2), goal=(1, 1), goal_delta=(-1, -1))
    assert first.graph == second.graph
    assert first.act(observation) == second.act(observation)
    assert len(first.edges) == len(graph.edges)
