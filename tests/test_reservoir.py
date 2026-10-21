"""Tests for the reservoir substrate and linear policy (M0+ spike)."""

from __future__ import annotations

import numpy as np

from neuroflynav.agent.reservoir import (
    LinearPolicy,
    ReservoirConfig,
    ReservoirRepresentation,
    build_core_from_graph,
    build_random_core,
)
from neuroflynav.env.nav import Action
from neuroflynav.env.nav_memory import MemoryGridNavConfig, MemoryGridNavEnv
from neuroflynav.graph.interface import Graph
from neuroflynav.graph.synthetic import SyntheticCXGraphLoader


def _graph() -> Graph:
    return SyntheticCXGraphLoader(seed=0).load()


def test_core_step_is_deterministic_and_motor_sized() -> None:
    graph = _graph()
    config = ReservoirConfig(grid_size=7)
    core_a = build_core_from_graph(graph, config, seed=0)
    core_b = build_core_from_graph(graph, config, seed=0)
    features = np.ones(6, dtype=np.float64)
    out_a = core_a.step(features)
    out_b = core_b.step(features)
    assert out_a.shape == (core_a.size,)
    assert np.allclose(out_a, out_b)


def test_reset_clears_state() -> None:
    core = build_core_from_graph(_graph(), ReservoirConfig(), seed=1)
    features = np.ones(6, dtype=np.float64)
    first = core.step(features).copy()
    core.step(features)
    core.reset()
    again = core.step(features)
    assert np.allclose(first, again)


def test_build_random_core_has_matched_motor_count() -> None:
    core = build_random_core(
        num_nodes=20,
        num_edges=40,
        num_sensory=4,
        num_motor=4,
        weight=1.0,
        config=ReservoirConfig(),
        seed=0,
    )
    assert len(core.motor) == 4


def test_linear_policy_returns_valid_action() -> None:
    graph = _graph()
    config = ReservoirConfig(grid_size=7)
    core = build_core_from_graph(graph, config, seed=0)
    representation = ReservoirRepresentation(core, config.grid_size)
    readout = np.zeros((4, core.size), dtype=np.float64)
    policy = LinearPolicy(representation, readout)
    env = MemoryGridNavEnv(MemoryGridNavConfig(size=7, max_steps=10, cue_steps=3))
    observation, _ = env.reset(seed=0)
    policy.reset()
    assert isinstance(policy.act(observation), Action)
