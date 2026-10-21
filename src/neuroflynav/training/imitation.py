"""Imitation learning for a linear readout (development spike, M0+).

Collect (feature, expert-action) pairs by running a privileged greedy expert on the
hidden-cue task, then fit the readout in closed form by ridge regression. The
representation advances its own state as it produces features, so a recurrent
representation learns to carry the hidden cue. This is spike machinery, not a frozen
training protocol.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from neuroflynav.agent.reservoir import Representation
from neuroflynav.env.nav_memory import MemoryGridNavEnv

FloatArray = NDArray[np.float64]

_NUM_ACTIONS = 4


def collect_dataset(
    env: MemoryGridNavEnv, representation: Representation, seeds: Sequence[int]
) -> tuple[FloatArray, FloatArray]:
    """Return (features, one-hot expert actions) gathered over expert rollouts."""
    feature_rows: list[FloatArray] = []
    target_rows: list[FloatArray] = []
    for seed in seeds:
        observation, _ = env.reset(seed)
        representation.reset()
        for _ in range(env.config.max_steps):
            expert = env.optimal_action()
            feature_rows.append(representation.features(observation))
            target = np.zeros(_NUM_ACTIONS, dtype=np.float64)
            target[int(expert)] = 1.0
            target_rows.append(target)
            observation, _, terminated, truncated, _ = env.step(expert)
            if terminated or truncated:
                break
    features: FloatArray = np.array(feature_rows, dtype=np.float64)
    targets: FloatArray = np.array(target_rows, dtype=np.float64)
    return features, targets


def fit_ridge(
    features: FloatArray, targets: FloatArray, ridge_lambda: float
) -> FloatArray:
    """Return a readout ``R`` (num_actions x feature_dim) via ridge regression."""
    if ridge_lambda < 0:
        raise ValueError("ridge_lambda must not be negative")
    if features.ndim != 2 or targets.ndim != 2:
        raise ValueError("features and targets must be 2-D")
    dimension = features.shape[1]
    gram: FloatArray = features.T @ features + ridge_lambda * np.eye(dimension)
    right_hand_side: FloatArray = features.T @ targets
    solution: FloatArray = np.linalg.solve(gram, right_hand_side)
    readout: FloatArray = solution.T
    return readout
