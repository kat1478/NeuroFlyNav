"""Reservoir-style policies over a fixed graph substrate (development spike, M0+).

A graph's topology becomes a *frozen* recurrent weight matrix (the "reservoir").
The observation is injected at sensory nodes, the state evolves through the fixed
wiring, and only a small linear readout over the motor nodes is trained (elsewhere,
by ridge regression). This mirrors the project's intended shape — frozen connectome
topology, trained heads — and lets topology affect what the readout can decode, so
controls can discriminate. Dynamics (tanh), micro-steps and scaling are M0+
placeholders, NOT the frozen scientific choices of TASK-013/014 (see D-026).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from neuroflynav.env.nav import Action
from neuroflynav.env.nav_memory import (
    FEATURE_DIM,
    MemoryGridObservation,
    encode_observation,
)
from neuroflynav.graph.interface import Graph, NodeRole
from neuroflynav.graph.matrix import scaled_for_radius, to_weight_matrix

FloatArray = NDArray[np.float64]

_ACTIONS: tuple[Action, ...] = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)


@dataclass(frozen=True)
class ReservoirConfig:
    """Fixed hyper-parameters for the reservoir substrate (spike placeholders)."""

    grid_size: int = 7
    micro_steps: int = 3
    spectral_radius: float = 1.1
    input_scale: float = 0.6


class Representation(Protocol):
    """Maps an observation stream to per-step feature vectors, with episodic reset."""

    def reset(self) -> None: ...

    def features(self, observation: MemoryGridObservation) -> FloatArray: ...


class MemoryPolicy(Protocol):
    """A resettable policy over the hidden-cue navigation task."""

    def reset(self) -> None: ...

    def act(self, observation: MemoryGridObservation) -> Action: ...


def _sensory_input_matrix(
    num_nodes: int, sensory: tuple[int, ...], scale: float, seed: int
) -> FloatArray:
    rng = np.random.default_rng(seed)
    matrix = np.zeros((num_nodes, FEATURE_DIM), dtype=np.float64)
    for node in sensory:
        matrix[node, :] = rng.standard_normal(FEATURE_DIM) * scale
    return matrix


class ReservoirCore:
    """A fixed recurrent substrate: frozen weights, fixed input map, evolving state."""

    def __init__(
        self,
        weights: FloatArray,
        input_matrix: FloatArray,
        motor: tuple[int, ...],
        config: ReservoirConfig,
    ) -> None:
        self._weights = weights
        self._input = input_matrix
        self._motor = motor
        self._config = config
        self._state: FloatArray = np.zeros(weights.shape[0], dtype=np.float64)

    @property
    def motor(self) -> tuple[int, ...]:
        return self._motor

    @property
    def size(self) -> int:
        return int(self._weights.shape[0])

    def reset(self) -> None:
        self._state = np.zeros(self._weights.shape[0], dtype=np.float64)

    def step(self, features: FloatArray) -> FloatArray:
        """Advance the substrate by the configured micro-steps; return the full state.

        The linear readout is trained over the whole reservoir state (standard
        echo-state practice), so a topology with better memory can carry the hidden
        cue forward. A biologically-constrained motor-only readout is a later
        decision (TASK-013/014), not an M0+ choice.
        """
        drive: FloatArray = self._input @ features
        state = self._state
        for _ in range(self._config.micro_steps):
            activated: FloatArray = np.tanh(self._weights @ state + drive)
            state = activated
        self._state = state
        return state


def build_core_from_graph(
    graph: Graph, config: ReservoirConfig, seed: int
) -> ReservoirCore:
    """Build a reservoir whose frozen weights come from a graph's topology."""
    weights = scaled_for_radius(to_weight_matrix(graph), config.spectral_radius)
    sensory = graph.nodes_with_role(NodeRole.SENSORY)
    motor = graph.nodes_with_role(NodeRole.MOTOR)
    if not sensory or not motor:
        raise ValueError("graph must have at least one sensory and one motor node")
    input_matrix = _sensory_input_matrix(
        graph.num_nodes, sensory, config.input_scale, seed
    )
    return ReservoirCore(weights, input_matrix, motor, config)


def build_random_core(
    num_nodes: int,
    num_edges: int,
    num_sensory: int,
    num_motor: int,
    weight: float,
    config: ReservoirConfig,
    seed: int,
) -> ReservoirCore:
    """Build a reservoir from a random directed graph with a matched edge count."""
    if num_sensory + num_motor > num_nodes:
        raise ValueError("num_nodes is too small for the requested roles")
    rng = np.random.default_rng(seed)
    raw = np.zeros((num_nodes, num_nodes), dtype=np.float64)
    placed = 0
    while placed < num_edges:
        source = int(rng.integers(num_nodes))
        target = int(rng.integers(num_nodes))
        if source == target:
            continue
        raw[target, source] += weight
        placed += 1
    weights = scaled_for_radius(raw, config.spectral_radius)
    sensory = tuple(range(num_sensory))
    motor = tuple(range(num_nodes - num_motor, num_nodes))
    input_matrix = _sensory_input_matrix(
        num_nodes, sensory, config.input_scale, seed + 1
    )
    return ReservoirCore(weights, input_matrix, motor, config)


class ReservoirRepresentation:
    """Features = the motor-node state after advancing the reservoir with the observation."""

    def __init__(self, core: ReservoirCore, grid_size: int) -> None:
        self._core = core
        self._grid_size = grid_size

    def reset(self) -> None:
        self._core.reset()

    def features(self, observation: MemoryGridObservation) -> FloatArray:
        encoded = encode_observation(observation, self._grid_size)
        return self._core.step(encoded)


class ReactiveRepresentation:
    """Features = the raw observation encoding, with no memory (baseline)."""

    def __init__(self, grid_size: int) -> None:
        self._grid_size = grid_size

    def reset(self) -> None:
        return None

    def features(self, observation: MemoryGridObservation) -> FloatArray:
        return encode_observation(observation, self._grid_size)


class LinearPolicy:
    """Pick the argmax action of a trained linear readout over a representation."""

    def __init__(self, representation: Representation, readout: FloatArray) -> None:
        self._representation = representation
        self._readout = readout

    def reset(self) -> None:
        self._representation.reset()

    def act(self, observation: MemoryGridObservation) -> Action:
        features = self._representation.features(observation)
        logits: FloatArray = self._readout @ features
        return _ACTIONS[int(np.argmax(logits))]
