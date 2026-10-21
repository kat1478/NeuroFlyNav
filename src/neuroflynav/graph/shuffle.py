"""Deterministic directed degree-preserving graph rewiring for the M0 control."""

from __future__ import annotations

from collections.abc import Sequence
from random import Random

import numpy as np

from neuroflynav.graph.interface import Edge, Graph

# Above this edge count use the fast swap loop (same swap rule, numpy-drawn indices).
# Smaller graphs keep the reference loop so M0/M0+ outputs stay byte-identical.
FAST_EDGE_LIMIT = 10_000


def shuffle_graph(graph: Graph, seed: int, swaps_per_edge: int = 10) -> Graph:
    """Return a seeded double-edge-swap shuffle of ``graph``.

    A swap replaces ``u→v`` and ``x→y`` with ``u→y`` and ``x→v``. It preserves
    every node's directed in-degree and out-degree, edge count, node roles, and
    the multiset of edge weights. Swaps that introduce self-loops are rejected.
    Parallel edges are allowed by the source graph interface and may result. Random rejection means
    the number of successful swaps is topology-dependent; it is not promised to
    produce a uniformly sampled graph or preserve motifs/path lengths/role mixing.
    """
    if swaps_per_edge < 0:
        raise ValueError("swaps_per_edge must not be negative")
    edges = list(graph.edges)
    if len(edges) < 2 or swaps_per_edge == 0:
        return graph

    if len(edges) > FAST_EDGE_LIMIT:
        return _fast_shuffle(graph, seed, swaps_per_edge)
    rng = Random(seed)
    attempts = swaps_per_edge * len(edges)
    for _ in range(attempts):
        first_index, second_index = rng.sample(range(len(edges)), 2)
        first = edges[first_index]
        second = edges[second_index]
        if first.source == second.source:
            continue

        rewired_first = Edge(first.source, second.target, first.weight)
        rewired_second = Edge(second.source, first.target, second.weight)
        if (
            rewired_first.source == rewired_first.target
            or rewired_second.source == rewired_second.target
        ):
            continue
        if rewired_first == first and rewired_second == second:
            continue

        edges[first_index] = rewired_first
        edges[second_index] = rewired_second

    if tuple(edges) == graph.edges:
        return graph
    return Graph(num_nodes=graph.num_nodes, edges=tuple(edges), roles=graph.roles)


def _fast_shuffle(graph: Graph, seed: int, swaps_per_edge: int) -> Graph:
    """Same double-edge-swap rule on plain int lists; draws all indices up front."""
    sources = [edge.source for edge in graph.edges]
    targets = [edge.target for edge in graph.edges]
    count = len(sources)
    rng = np.random.default_rng(seed)
    firsts = rng.integers(count, size=swaps_per_edge * count)
    seconds = rng.integers(count - 1, size=swaps_per_edge * count)
    seconds += seconds >= firsts  # two distinct edges, like Random.sample
    for first, second in zip(firsts.tolist(), seconds.tolist(), strict=True):
        source_a, source_b = sources[first], sources[second]
        if source_a == source_b:
            continue
        target_a, target_b = targets[first], targets[second]
        if source_a == target_b or source_b == target_a:
            continue
        targets[first], targets[second] = target_b, target_a
    edges = tuple(
        Edge(source, target, edge.weight)
        for source, target, edge in zip(sources, targets, graph.edges, strict=True)
    )
    return Graph(num_nodes=graph.num_nodes, edges=edges, roles=graph.roles)


def shuffle_within_target_groups(
    graph: Graph, groups: Sequence[str], seed: int, swaps_per_edge: int = 10
) -> Graph:
    """Double-edge swaps restricted to edges whose targets share a group label.

    Swapping the targets of ``u→v`` and ``x→y`` only when ``group(v) == group(y)``
    preserves every node's in/out degree *and* the number of edges between every
    pair of groups (the group-to-group block structure). Only the exact partners
    inside a group are randomized. Used as the H2 control with cell types as groups.
    """
    if len(groups) != graph.num_nodes:
        raise ValueError("groups must have one label per node")
    if swaps_per_edge < 0:
        raise ValueError("swaps_per_edge must not be negative")
    sources = [edge.source for edge in graph.edges]
    targets = [edge.target for edge in graph.edges]
    members: dict[str, list[int]] = {}
    for index, target in enumerate(targets):
        members.setdefault(groups[target], []).append(index)
    pools = [pool for pool in members.values() if len(pool) >= 2]
    if not pools or swaps_per_edge == 0:
        return graph
    sizes = np.array([len(pool) for pool in pools], dtype=np.float64)
    attempts = swaps_per_edge * int(sizes.sum())
    rng = np.random.default_rng(seed)
    pool_ids = rng.choice(len(pools), size=attempts, p=sizes / sizes.sum())
    first_draws = rng.random(attempts)
    second_draws = rng.random(attempts)
    for pool_id, first_draw, second_draw in zip(
        pool_ids.tolist(), first_draws.tolist(), second_draws.tolist(), strict=True
    ):
        pool = pools[pool_id]
        first_position = int(first_draw * len(pool))
        second_position = int(second_draw * (len(pool) - 1))
        second_position += second_position >= first_position
        first, second = pool[first_position], pool[second_position]
        source_a, source_b = sources[first], sources[second]
        if source_a == source_b:
            continue
        target_a, target_b = targets[first], targets[second]
        if source_a == target_b or source_b == target_a:
            continue
        targets[first], targets[second] = target_b, target_a
    edges = tuple(
        Edge(source, target, edge.weight)
        for source, target, edge in zip(sources, targets, graph.edges, strict=True)
    )
    return Graph(num_nodes=graph.num_nodes, edges=edges, roles=graph.roles)
