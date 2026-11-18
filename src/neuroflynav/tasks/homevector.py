"""The network reports the home vector itself (exploratory, path integration).

In :mod:`neuroflynav.tasks.homing` the network only reports its heading and code sums the
steps into a home vector. Here the network has to do the summing: at every step it reports
where the nest is, relative to the fly, in map coordinates (x east, y north), scaled by
``HOME_SCALE_M``. Its inputs are the heading task's (self-motion, plus the heading at a
landmark every few steps); speed is constant, so distance is the number of steps.

Training uses synthetic flights of fixed length (no map); testing uses the foraging flights
over Warsaw from the homing task, which stop at a treat and can be longer than training.
The **homing error** is the distance between the reported and the true home vector at the
last outbound step — where the fly would land after flying back along its own estimate.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from neuroflynav.tasks.heading import generate_trajectory
from neuroflynav.tasks.homing import Flight, HomingConfig

FloatArray = NDArray[np.float64]

HOME_SCALE_M = 100.0
OUTPUT_DIM = 2


def training_config(steps: int = 100) -> HomingConfig:
    """Flight parameters of the homing task with a fixed training length."""
    return replace(HomingConfig(), max_steps=steps)


def positions(theta: FloatArray, step_m: float) -> FloatArray:
    """Fly position after every step, ``(steps, 2)`` metres from the nest."""
    steps = step_m * np.column_stack([np.cos(theta), np.sin(theta)])
    out: FloatArray = np.cumsum(steps, axis=0)
    return out


def home_targets(position: FloatArray) -> FloatArray:
    """The home vector (nest relative to the fly), scaled: ``−position / HOME_SCALE_M``."""
    return -np.asarray(position, dtype=float) / HOME_SCALE_M


def synthetic_flight(config: HomingConfig, seed: int) -> tuple[FloatArray, FloatArray]:
    """``(inputs, targets)`` of one map-free flight of ``config.max_steps`` steps."""
    inputs, _, theta = generate_trajectory(config.heading_config(), seed)
    return inputs, home_targets(positions(theta, config.step_m))


def dataset(
    config: HomingConfig, seeds: Sequence[int]
) -> tuple[FloatArray, FloatArray]:
    """Stacked synthetic flights: inputs ``(n, steps, 5)``, targets ``(n, steps, 2)``."""
    pairs = [synthetic_flight(config, seed) for seed in seeds]
    return np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])


def scored_steps(config: HomingConfig) -> NDArray[np.bool_]:
    """Steps that count in training: after the initial heading cue."""
    return np.arange(config.max_steps) >= config.cue_steps


def flight_targets(flight: Flight) -> FloatArray:
    """Scaled home vector after every step of a (Warsaw) flight."""
    return home_targets(flight.path[1:])


def endpoint_error(flight: Flight, predicted: FloatArray) -> float:
    """Metres between reported and true home vector at the flight's last step.

    ``predicted`` is the network output for this flight, ``(steps or more, 2)``, scaled.
    Flying back along the estimate lands exactly this far from the nest.
    """
    estimate = np.asarray(predicted[flight.steps - 1], dtype=float) * HOME_SCALE_M
    return float(np.linalg.norm(estimate + flight.path[-1]))


def along_error(flight: Flight, predicted: FloatArray, cue_steps: int) -> float:
    """Mean metres between reported and true home vector over the flight (after the cue)."""
    truth = flight_targets(flight) * HOME_SCALE_M
    estimate = np.asarray(predicted[: flight.steps], dtype=float) * HOME_SCALE_M
    gaps = np.linalg.norm(estimate - truth, axis=1)[cue_steps:]
    return float(gaps.mean()) if len(gaps) else float("nan")
