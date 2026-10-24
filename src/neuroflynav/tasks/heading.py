"""Heading integration task (exploratory spike, M2 candidate).

The agent turns with a smoothly varying angular velocity. Its absolute heading is shown
for the first ``cue_steps`` steps and then only at a landmark every ``landmark_every``
steps (0 = never again); in between it receives only the angular velocity (self-motion)
and must report its current heading as one of ``sectors`` sectors. Solving it needs
*integration* — adding up turns since the last landmark — which is what the central-complex
compass (EPG/PEN ring) does in the fly, combining vision with self-motion.
Scoring uses only steps where the heading is not shown (``scored_steps``).

Input vector per step (``INPUT_DIM`` = 5): bias, angular velocity (scaled), cue flag,
cos(heading) and sin(heading) while the cue is visible (zero afterwards).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

INPUT_DIM = 5


@dataclass(frozen=True)
class HeadingTaskConfig:
    """Parameters of the heading integration task."""

    steps: int = 40
    cue_steps: int = 3
    sectors: int = 8
    turn_std: float = 0.3  # stationary std of angular velocity, radians per step
    persistence: float = 0.8  # AR(1) coefficient of angular velocity
    landmark_every: int = 0  # heading shown again every N steps; 0 = never

    def __post_init__(self) -> None:
        if not 0 < self.cue_steps < self.steps:
            raise ValueError("cue_steps must be between 1 and steps - 1")
        if self.sectors < 2:
            raise ValueError("sectors must be at least 2")
        if self.turn_std <= 0 or not 0 <= self.persistence < 1:
            raise ValueError("turn_std must be positive and persistence in [0, 1)")
        if self.landmark_every < 0:
            raise ValueError("landmark_every must not be negative")


def cue_visible(config: HeadingTaskConfig) -> NDArray[np.bool_]:
    """Steps at which the absolute heading is shown."""
    steps = np.arange(config.steps)
    visible = steps < config.cue_steps
    if config.landmark_every:
        visible |= steps % config.landmark_every == 0
    return visible


def scored_steps(config: HeadingTaskConfig) -> NDArray[np.bool_]:
    """Steps that count for accuracy: after the first cue and with no heading shown."""
    return ~cue_visible(config) & (np.arange(config.steps) >= config.cue_steps)


def heading_sector(theta: FloatArray, sectors: int) -> IntArray:
    """Map angles (radians) to sector indices ``0 .. sectors - 1``."""
    wrapped = np.mod(theta, 2 * math.pi)
    indices: IntArray = np.floor(wrapped / (2 * math.pi / sectors)).astype(np.int64)
    return np.minimum(indices, sectors - 1)


def generate_sequence(
    config: HeadingTaskConfig, seed: int
) -> tuple[FloatArray, IntArray]:
    """Return ``(inputs, sectors)`` with shapes ``(steps, INPUT_DIM)`` and ``(steps,)``."""
    rng = np.random.default_rng(seed)
    innovation = config.turn_std * math.sqrt(1 - config.persistence**2)
    omega = np.empty(config.steps)
    omega[0] = rng.normal(0.0, config.turn_std)
    for step in range(1, config.steps):
        omega[step] = config.persistence * omega[step - 1] + rng.normal(0.0, innovation)
    start = rng.uniform(0.0, 2 * math.pi)
    theta = start + np.cumsum(omega)
    visible = cue_visible(config)
    inputs = np.zeros((config.steps, INPUT_DIM))
    inputs[:, 0] = 1.0
    inputs[:, 1] = omega / config.turn_std
    inputs[:, 2] = visible
    inputs[:, 3] = np.where(visible, np.cos(theta), 0.0)
    inputs[:, 4] = np.where(visible, np.sin(theta), 0.0)
    return inputs, heading_sector(theta, config.sectors)


def hold_last_landmark_accuracy(
    config: HeadingTaskConfig, seeds: Sequence[int]
) -> float:
    """Reference: predict the sector seen at the last cue (perfect memory, no integration)."""
    visible = cue_visible(config)
    mask = scored_steps(config)
    hits = 0
    for seed in seeds:
        _, sectors = generate_sequence(config, seed)
        last_seen = np.maximum.accumulate(np.where(visible, np.arange(config.steps), 0))
        hits += int((sectors[last_seen] == sectors)[mask].sum())
    return hits / (int(mask.sum()) * len(seeds))
