import json
import math
from pathlib import Path

import numpy as np
import pytest

from neuroflynav.experiments import run_homevector
from neuroflynav.experiments.aggregate_homevector import aggregate
from neuroflynav.graph.interface import Edge, Graph, NodeRole
from neuroflynav.tasks.homevector import (
    HOME_SCALE_M,
    along_error,
    dataset,
    endpoint_error,
    error_by_length,
    flight_targets,
    positions,
    scored_steps,
    synthetic_flight,
    training_config,
)
from neuroflynav.tasks.homing import HomingConfig, outbound_flight
from neuroflynav.training.masked_rnn import (
    TrainingConfig,
    predict_values,
    train_regression,
)


def test_targets_point_from_the_fly_back_to_the_nest() -> None:
    config = training_config(50)
    inputs, targets = synthetic_flight(config, seed=4)
    assert inputs.shape == (50, 5) and targets.shape == (50, 2)
    # moving along the steps and then along the target returns to the nest
    from neuroflynav.tasks.heading import generate_trajectory

    _, _, theta = generate_trajectory(config.heading_config(), 4)
    np.testing.assert_allclose(
        positions(theta, config.step_m) + targets * HOME_SCALE_M, 0.0, atol=1e-9
    )
    # one step east from the nest: the nest is 5 m west
    np.testing.assert_allclose(positions(np.array([0.0]), 5.0), [[5.0, 0.0]])


def test_dataset_and_scored_steps() -> None:
    config = training_config(20)
    inputs, targets = dataset(config, [1, 2, 3])
    assert inputs.shape == (3, 20, 5) and targets.shape == (3, 20, 2)
    mask = scored_steps(config)
    assert not mask[: config.cue_steps].any() and mask[config.cue_steps :].all()


def test_perfect_estimate_has_zero_error_and_stay_put_is_the_distance() -> None:
    config = HomingConfig(max_steps=60)
    flight = outbound_flight(config, np.empty((0, 2)), seed=7)
    truth = flight_targets(flight)
    assert endpoint_error(flight, truth) < 1e-9
    assert along_error(flight, truth, config.cue_steps) < 1e-9
    stay = endpoint_error(flight, np.zeros_like(truth))
    assert math.isclose(stay, float(np.linalg.norm(flight.path[-1])))
    # padded predictions beyond the flight are ignored
    padded = np.vstack([truth, np.full((10, 2), 99.0)])
    assert endpoint_error(flight, padded) < 1e-9


def _ring(n: int = 12) -> Graph:
    edges = tuple(Edge(i, (i + 1) % n, 1.0) for i in range(n)) + tuple(
        Edge(i, (i + 3) % n, 0.5) for i in range(n)
    )
    roles = tuple(NodeRole.SENSORY if i < 4 else NodeRole.INTERNAL for i in range(n))
    return Graph(num_nodes=n, edges=edges, roles=roles)


def test_regression_training_reduces_the_error() -> None:
    config = training_config(15)
    train = dataset(config, range(32))
    mask = scored_steps(config).tolist()
    before = train_regression(
        _ring(), train, mask, TrainingConfig(steps=0, batch_size=8), seed=0
    )
    after = train_regression(
        _ring(), train, mask, TrainingConfig(steps=60, batch_size=8), seed=0
    )
    out_before, out_after = (
        predict_values(before, train[0]),
        predict_values(after, train[0]),
    )
    assert out_after.shape == (32, 15, 2)
    err = [float(np.mean((o - train[1]) ** 2)) for o in (out_before, out_after)]
    assert err[1] < err[0]
    # deterministic for a seed
    again = train_regression(
        _ring(), train, mask, TrainingConfig(steps=60, batch_size=8), seed=0
    )
    np.testing.assert_array_equal(predict_values(again, train[0]), out_after)


def _job(variant: str, seed: int, error: float) -> dict[str, object]:
    return {
        "variant": variant,
        "seed": seed,
        "core": False,
        "nodes": 10,
        "edges": 20,
        "weights": "log1p",
        "train": 256,
        "length": 100,
        "training": {"steps": 300},
        "homing": {"step_m": 5.0},
        "nest": [21.0, 52.2],
        "flights": 100,
        "error_m": {"mean": error, "median": error},
        "along_error_m": {"mean": error / 2, "median": error / 2},
        "synthetic_error_m": {"mean": error, "median": error},
        "stay_error_m": {"mean": 200.0, "median": 150.0},
        "quantised_error_m": {"mean": 12.0, "median": 10.0},
        "no_integration_error_m": {"mean": 32.0, "median": 25.0},
    }


def test_aggregate_pairs_over_seeds() -> None:
    jobs = [
        _job(v, s, base + offset)
        for s, base in ((1, 40.0), (2, 50.0), (3, 45.0))
        for v, offset in (("FlyConnectome", 0.0), ("RandomMask", 3.0))
    ]
    result = aggregate(jobs)
    diff = result["fly_minus_error_m"]["RandomMask"]  # type: ignore[index]
    assert diff["mean"] == pytest.approx(-3.0)
    assert diff["low"] == pytest.approx(-3.0) == diff["high"]
    assert result["seeds"] == [1, 2, 3]
    with pytest.raises(ValueError, match="missing"):
        aggregate(jobs[:-1])
    with pytest.raises(ValueError, match="differs"):
        aggregate([*jobs[:-1], {**jobs[-1], "length": 50}])


def test_job_writes_a_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """End to end on a tiny graph, without the hemibrain download."""
    graph = _ring()
    monkeypatch.setattr(run_homevector, "load_cx_subgraph", lambda _: "subgraph")
    monkeypatch.setattr(run_homevector, "to_graph", lambda *_: graph)
    monkeypatch.setattr(
        run_homevector, "node_groups", lambda _: ["a"] * graph.num_nodes
    )
    output = tmp_path / "job.json"
    run_homevector.main(
        [
            "--array-index",
            "4",
            "--first-seed",
            "100",
            "--train",
            "8",
            "--length",
            "12",
            "--steps",
            "3",
            "--test",
            "4",
            "--flights",
            "3",
            "--output",
            str(output),
        ]
    )
    result = json.loads(output.read_text())
    assert result["variant"] == "RandomMask" and result["seed"] == 100
    assert result["flights"] == 3 and len(result["errors_m"]) == 3
    assert result["error_m"]["mean"] >= 0


def test_error_by_length_bins_flights() -> None:
    bins = error_by_length([10, 50, 51, 120, 300], [1.0, 3.0, 5.0, 7.0, 9.0])
    assert [b["max_steps"] for b in bins] == [50, 100, 200, 300]
    assert [b["flights"] for b in bins] == [2, 1, 1, 1]
    assert [b["mean_m"] for b in bins] == [2.0, 5.0, 7.0, 9.0]
    empty = error_by_length([10], [4.0])
    assert empty[0]["mean_m"] == 4.0 and math.isnan(float(empty[1]["mean_m"]))
    with pytest.raises(ValueError):
        error_by_length([1, 2], [1.0])
