"""Foraging and homing by path integration (exploratory, "fly in the city").

A fly leaves its nest and flies a correlated random walk — the same turning statistics as
the heading task — until a treat is within its detection radius. To fly home it needs the
*home vector*: the sum of all flight steps. A network that reports its heading as one of
``sectors`` sectors gives an estimated home vector from the sector centres; flying back along
it ends ``homing_error`` metres from the nest.

Coordinates are local metres around the nest (x east, y north); angles follow the maths
convention (0 = east, counter-clockwise), as in :mod:`neuroflynav.tasks.heading`.
"""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neuroflynav.tasks.heading import HeadingTaskConfig, generate_trajectory

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]

EARTH_M_PER_DEG_LAT = 110_540.0
EARTH_M_PER_DEG_LON_EQUATOR = 111_320.0


@dataclass(frozen=True)
class HomingConfig:
    """Flight parameters; turning matches the trained heading task."""

    step_m: float = 5.0
    detect_m: float = 15.0
    max_steps: int = 300
    cue_steps: int = 3
    landmark_every: int = 8
    turn_std: float = 0.15
    persistence: float = 0.8
    sectors: int = 8

    def heading_config(self) -> HeadingTaskConfig:
        return HeadingTaskConfig(
            steps=self.max_steps,
            cue_steps=self.cue_steps,
            sectors=self.sectors,
            turn_std=self.turn_std,
            persistence=self.persistence,
            landmark_every=self.landmark_every,
        )


@dataclass(frozen=True)
class Flight:
    """One outbound flight; ``path`` has ``steps + 1`` points starting at the nest."""

    inputs: FloatArray  # (steps, INPUT_DIM) network input, as in the heading task
    sectors: IntArray  # (steps,) true heading sector
    theta: FloatArray  # (steps,) true heading, radians
    path: FloatArray  # (steps + 1, 2) positions in metres
    treat: int  # index of the treat found, -1 if none within max_steps

    @property
    def steps(self) -> int:
        return len(self.sectors)


def to_local(
    lon: FloatArray, lat: FloatArray, lon0: float, lat0: float
) -> tuple[FloatArray, FloatArray]:
    """Equirectangular projection to metres around (lon0, lat0); fine at city scale."""
    scale = EARTH_M_PER_DEG_LON_EQUATOR * math.cos(math.radians(lat0))
    return (lon - lon0) * scale, (lat - lat0) * EARTH_M_PER_DEG_LAT


def to_lonlat(
    x: FloatArray, y: FloatArray, lon0: float, lat0: float
) -> tuple[FloatArray, FloatArray]:
    scale = EARTH_M_PER_DEG_LON_EQUATOR * math.cos(math.radians(lat0))
    return lon0 + x / scale, lat0 + y / EARTH_M_PER_DEG_LAT


def load_treats(path: Path, lon0: float, lat0: float) -> FloatArray:
    """Treat positions (n, 2) in metres from a GeoJSON file of points."""
    features = json.loads(path.read_text())["features"]
    lon = np.array([f["geometry"]["coordinates"][0] for f in features], dtype=float)
    lat = np.array([f["geometry"]["coordinates"][1] for f in features], dtype=float)
    x, y = to_local(lon, lat, lon0, lat0)
    return np.column_stack([x, y])


def outbound_flight(config: HomingConfig, treats: FloatArray, seed: int) -> Flight:
    """Fly from the nest until a treat is within ``detect_m`` (or ``max_steps``)."""
    inputs, sectors, theta = generate_trajectory(config.heading_config(), seed)
    steps = config.step_m * np.column_stack([np.cos(theta), np.sin(theta)])
    path = np.vstack([np.zeros((1, 2)), np.cumsum(steps, axis=0)])
    treat = -1
    end = config.max_steps
    if len(treats):
        # distance of every visited point (after the first step) to every treat
        gaps = np.linalg.norm(path[1:, None, :] - treats[None, :, :], axis=2)
        hits = np.argwhere(gaps <= config.detect_m)
        if len(hits):
            first = int(hits[:, 0].min())
            end = first + 1
            treat = int(hits[hits[:, 0] == first][0, 1])
    return Flight(inputs[:end], sectors[:end], theta[:end], path[: end + 1], treat)


def sector_centres(sectors: IntArray, count: int) -> FloatArray:
    return (np.asarray(sectors, dtype=float) + 0.5) * (2 * math.pi / count)


def home_vector(headings: FloatArray, step_m: float) -> FloatArray:
    """Sum of flight steps along ``headings`` — where the nest is *not*; fly back along −v."""
    h = np.asarray(headings, dtype=float)
    return np.array([np.cos(h).sum(), np.sin(h).sum()]) * step_m


def homing_error(flight: Flight, headings: FloatArray, step_m: float) -> float:
    """Distance from the nest after flying back along the estimated home vector."""
    return float(np.linalg.norm(flight.path[-1] - home_vector(headings, step_m)))


def errors_for_sectors(
    flights: Sequence[Flight], predicted: Sequence[IntArray], config: HomingConfig
) -> list[float]:
    """Homing error of each flight when headings are decoded from predicted sectors."""
    return [
        homing_error(f, sector_centres(p[: f.steps], config.sectors), config.step_m)
        for f, p in zip(flights, predicted, strict=True)
    ]


def last_landmark_headings(flight: Flight, config: HomingConfig) -> FloatArray:
    """Reference without integration: keep the sector seen at the last landmark."""
    visible = flight.inputs[:, 2] > 0
    index = np.maximum.accumulate(np.where(visible, np.arange(flight.steps), 0))
    return sector_centres(flight.sectors[index], config.sectors)
