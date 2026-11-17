import math

import numpy as np
import pytest

from neuroflynav.tasks.homing import (
    HomingConfig,
    errors_for_sectors,
    home_vector,
    homing_error,
    last_landmark_headings,
    outbound_flight,
    sector_centres,
    to_local,
    to_lonlat,
)

CONFIG = HomingConfig(max_steps=120)


def test_projection_round_trip() -> None:
    lon, lat = np.array([21.01, 21.03]), np.array([52.23, 52.24])
    x, y = to_local(lon, lat, 21.0035, 52.2357)
    back = to_lonlat(x, y, 21.0035, 52.2357)
    np.testing.assert_allclose(back[0], lon)
    np.testing.assert_allclose(back[1], lat)
    # 0.01° of latitude is about 1.1 km
    assert abs(to_local(np.array([0.0]), np.array([0.01]), 0.0, 0.0)[1][0] - 1105.4) < 1


def test_true_headings_bring_the_fly_home_exactly() -> None:
    flight = outbound_flight(CONFIG, np.empty((0, 2)), seed=3)
    assert flight.steps == CONFIG.max_steps and flight.treat == -1
    assert homing_error(flight, flight.theta, CONFIG.step_m) < 1e-9


def test_flight_stops_at_the_first_treat_in_range() -> None:
    probe = outbound_flight(CONFIG, np.empty((0, 2)), seed=5)
    target = probe.path[40] + np.array([CONFIG.detect_m / 2, 0.0])
    flight = outbound_flight(CONFIG, np.array([[1e6, 1e6], target]), seed=5)
    assert flight.treat == 1
    assert flight.steps <= 40
    assert np.linalg.norm(flight.path[-1] - target) <= CONFIG.detect_m
    np.testing.assert_allclose(flight.path, probe.path[: flight.steps + 1])


def test_quantised_and_no_integration_references() -> None:
    flight = outbound_flight(CONFIG, np.empty((0, 2)), seed=8)
    quantised = homing_error(
        flight, sector_centres(flight.sectors, CONFIG.sectors), CONFIG.step_m
    )
    stale = homing_error(flight, last_landmark_headings(flight, CONFIG), CONFIG.step_m)
    assert 0 < quantised < stale
    # the true sector centre is never more than half a sector from the true heading
    gap = np.angle(np.exp(1j * (sector_centres(flight.sectors, 8) - flight.theta)))
    assert np.all(np.abs(gap) <= math.pi / 8 + 1e-9)


def test_errors_use_only_the_flight_steps() -> None:
    flight = outbound_flight(CONFIG, np.empty((0, 2)), seed=2)
    padded = np.concatenate([flight.sectors, np.zeros(10, dtype=np.int64)])
    assert errors_for_sectors([flight], [padded], CONFIG) == pytest.approx(
        [homing_error(flight, sector_centres(flight.sectors, 8), CONFIG.step_m)]
    )


def test_home_vector_of_a_square_is_zero() -> None:
    headings = np.repeat([0.0, math.pi / 2, math.pi, 3 * math.pi / 2], 5)
    np.testing.assert_allclose(home_vector(headings, 2.0), [0.0, 0.0], atol=1e-9)


def _job(variant: str, seed: int, error: float) -> dict[str, object]:
    return {
        "variant": variant,
        "seed": seed,
        "core": False,
        "nodes": 10,
        "edges": 20,
        "weights": "log1p",
        "train": 8,
        "training": {},
        "homing": {},
        "nest": [21.0, 52.2],
        "flights": 5,
        "error_m": {"mean": error},
        "heading_accuracy": 0.8,
        "quantised_error_m": {"mean": 10.0},
        "no_integration_error_m": {"mean": 30.0},
    }


def test_aggregate_homing_pairs_by_seed_and_rejects_gaps() -> None:
    from neuroflynav.experiments.aggregate_homing import aggregate

    jobs = [
        _job("FlyConnectome", 1, 20.0),
        _job("RandomMask", 1, 25.0),
        _job("FlyConnectome", 2, 22.0),
        _job("RandomMask", 2, 29.0),
    ]
    result = aggregate(jobs)
    assert result["fly_minus_error_m"]["RandomMask"]["mean"] == pytest.approx(-6.0)  # type: ignore[index]
    assert result["seeds"] == [1, 2]
    with pytest.raises(ValueError, match="missing jobs: RandomMask/2"):
        aggregate(jobs[:3])
    with pytest.raises(ValueError, match="duplicate"):
        aggregate([*jobs, jobs[0]])


def test_padded_inputs_keep_each_flight_prefix() -> None:
    from neuroflynav.experiments.run_homing import padded_inputs

    flights = [outbound_flight(CONFIG, np.empty((0, 2)), seed=s) for s in (1, 2)]
    short = flights[0].__class__(
        flights[0].inputs[:10],
        flights[0].sectors[:10],
        flights[0].theta[:10],
        flights[0].path[:11],
        -1,
    )
    batch = padded_inputs([short, flights[1]], CONFIG.max_steps)
    np.testing.assert_array_equal(batch[0, :10], short.inputs)
    assert not batch[0, 10:].any()
