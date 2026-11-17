import json
import math
from pathlib import Path

import numpy as np
import pytest

from neuroflynav.experiments.run_trained import heading_task
from neuroflynav.graph.interface import Edge, Graph, NodeRole
from neuroflynav.reporting.app_data import (
    APP_DATA_DIR,
    build,
    family_neurons,
    replay,
)
from neuroflynav.reporting.figures import DATA_DIR
from neuroflynav.tasks.heading import (
    generate_sequence,
    generate_trajectory,
    heading_sector,
    scored_steps,
)
from neuroflynav.training.masked_rnn import TrainingConfig, train_and_score

CONFIG = TrainingConfig(steps=3)


def _graph() -> tuple[Graph, tuple[str, ...]]:
    rng = np.random.default_rng(1)
    edges: set[tuple[int, int]] = set()
    while len(edges) < 40:
        s, t = (int(v) for v in rng.integers(12, size=2))
        if s != t:
            edges.add((s, t))
    roles = (NodeRole.SENSORY,) * 4 + (NodeRole.INTERNAL,) * 4 + (NodeRole.MOTOR,) * 4
    graph = Graph(
        num_nodes=12,
        edges=tuple(Edge(s, t, 1.0) for s, t in sorted(edges)),
        roles=roles,
    )
    return graph, tuple(f"type{i % 3}" for i in range(12))


def test_trajectory_matches_sequence_and_sectors() -> None:
    task = heading_task()
    inputs, sectors, theta = generate_trajectory(task, 7)
    old_inputs, old_sectors = generate_sequence(task, 7)
    np.testing.assert_array_equal(inputs, old_inputs)
    np.testing.assert_array_equal(sectors, old_sectors)
    np.testing.assert_array_equal(heading_sector(theta, task.sectors), sectors)


def test_replay_accuracy_equals_train_and_score() -> None:
    graph, groups = _graph()
    task = heading_task()
    data = replay(graph, groups, task, CONFIG, seed=3, num_train=4, num_test=3)
    train = tuple(
        np.stack(a)
        for a in zip(*(generate_sequence(task, s) for s in range(30_000, 30_004)))
    )
    test = tuple(
        np.stack(a)
        for a in zip(*(generate_sequence(task, s) for s in range(35_000, 35_003)))
    )
    expected = train_and_score(
        graph,
        train,  # type: ignore[arg-type]
        test,  # type: ignore[arg-type]
        scored_steps(task).tolist(),
        task.sectors,
        CONFIG,
        3,
    )
    fly = data["variants"]["FlyConnectome"]  # type: ignore[index]
    assert fly["accuracy"] == round(expected, 4)
    assert len(data["sequences"]) == 3  # type: ignore[arg-type]
    first = data["sequences"][0]  # type: ignore[index]
    assert all(0 <= v < 2 * math.pi for v in first["theta"])


def test_family_neurons_counts_by_family() -> None:
    counts = family_neurons(["EPG", "EPG", "PEN_a", "unknown"], ["EPG", "PEN", "other"])
    assert counts == [2, 1, 1]


def test_committed_app_data_rebuilds_identically() -> None:
    previous = {
        name: json.loads((APP_DATA_DIR / f"{name}.json").read_text())
        for name in ("results", "wiring", "replay")
    }
    summary = json.loads((DATA_DIR / "summary.json").read_text())
    files = build(summary, DATA_DIR / "cx_family_matrix.csv", previous, {})
    for name, data in files.items():
        text = json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"
        assert text == Path(APP_DATA_DIR / f"{name}.json").read_text(), name


def test_city_converts_flights_and_checks_they_match() -> None:
    from neuroflynav.reporting.app_data import city

    trace = {
        "path": [[0.0, 0.0], [5.0, 0.0], [10.0, 0.0]],
        "treat": 0,
        "sectors": [0, 0],
        "predicted": [0, 1],
        "estimate": [7.0, 1.0],
    }
    job = {
        "seed": 100,
        "nest": [21.0, 52.2],
        "homing": {"step_m": 5.0},
        "heading_accuracy": 0.8,
        "error_m": {"mean": 3.0},
        "flights": 1,
        "quantised_error_m": {"mean": 1.0},
        "no_integration_error_m": {"mean": 2.0},
        "traces": [trace],
    }
    treats: dict[str, object] = {
        "features": [{"properties": {"id": 0, "kind": "café", "name": "X"}}]
    }
    data = city(job, {**job, "traces": [{**trace, "estimate": [10.0, 0.0]}]}, treats)
    flight = data["flights"][0]  # type: ignore[index]
    assert flight["path"][0] == [21.0, 52.2]
    assert flight["treat"]["name"] == "X"
    assert flight["returns"]["FlyConnectome"]["error_m"] == round(10.0**0.5, 1)
    assert flight["returns"]["RandomMask"]["error_m"] == 0.0
    other = {
        **job,
        "traces": [{**trace, "path": [[0.0, 0.0], [0.0, 5.0], [0.0, 10.0]]}],
    }
    with pytest.raises(ValueError, match="flights differ"):
        city(job, other, treats)
