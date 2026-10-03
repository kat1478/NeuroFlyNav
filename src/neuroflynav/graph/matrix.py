"""Turn a :class:`Graph` into a numpy weight matrix (development spike, M0+).

Used to drive a fixed recurrent substrate from graph topology. This is spike
machinery: the matrix convention (``M[target, source] = weight``) and the spectral
scaling are engineering choices, not frozen scientific decisions about dynamics.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from neuroflynav.graph.interface import Graph

FloatArray = NDArray[np.float64]


def to_weight_matrix(graph: Graph) -> FloatArray:
    """Return an ``N x N`` matrix where entry ``[target, source]`` sums edge weights."""
    matrix = np.zeros((graph.num_nodes, graph.num_nodes), dtype=np.float64)
    for edge in graph.edges:
        matrix[edge.target, edge.source] += edge.weight
    return matrix


def scaled_for_radius(matrix: FloatArray, target_radius: float) -> FloatArray:
    """Rescale a square matrix to the given spectral radius (echo-state stability)."""
    if target_radius <= 0:
        raise ValueError("target_radius must be positive")
    eigenvalues = np.linalg.eigvals(matrix)
    spectral_radius = float(np.max(np.abs(eigenvalues)))
    if spectral_radius == 0.0:
        return matrix.copy()
    scaled: FloatArray = matrix * (target_radius / spectral_radius)
    return scaled
