"""Tests for the graph-to-matrix conversion (M0+ spike)."""

from __future__ import annotations

import numpy as np
import pytest

from neuroflynav.graph.interface import Edge, Graph, NodeRole
from neuroflynav.graph.matrix import scaled_for_radius, to_weight_matrix


def _small_graph() -> Graph:
    return Graph(
        num_nodes=3,
        edges=(Edge(0, 1, 2.0), Edge(1, 2, 1.0)),
        roles=(NodeRole.SENSORY, NodeRole.INTERNAL, NodeRole.MOTOR),
    )


def test_to_weight_matrix_places_weight_at_target_source() -> None:
    matrix = to_weight_matrix(_small_graph())
    assert matrix.shape == (3, 3)
    assert matrix[1, 0] == 2.0
    assert matrix[2, 1] == 1.0
    assert matrix[0, 1] == 0.0


def test_scaled_for_radius_sets_spectral_radius() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.standard_normal((5, 5))
    scaled = scaled_for_radius(matrix, 0.9)
    radius = float(np.max(np.abs(np.linalg.eigvals(scaled))))
    assert abs(radius - 0.9) < 1e-9


def test_scaled_for_radius_rejects_nonpositive() -> None:
    with pytest.raises(ValueError, match="positive"):
        scaled_for_radius(np.zeros((2, 2), dtype=np.float64), 0.0)
