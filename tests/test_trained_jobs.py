import numpy as np
import pytest

from neuroflynav.experiments.aggregate_trained import aggregate
from neuroflynav.experiments.run_trained import (
    VARIANTS,
    build_variant,
    run_job,
    summarize,
)
from neuroflynav.graph.interface import Edge, Graph, NodeRole
from neuroflynav.training.masked_rnn import TrainingConfig

CONFIG = TrainingConfig(steps=2)
META = {"core": True, "nodes": 12, "edges": 0, "training": {}}


def _graph() -> tuple[Graph, tuple[str, ...]]:
    rng = np.random.default_rng(0)
    edges: set[tuple[int, int]] = set()
    while len(edges) < 40:
        s, t = (int(v) for v in rng.integers(12, size=2))
        if s != t:
            edges.add((s, t))
    roles = (NodeRole.SENSORY,) * 4 + (NodeRole.INTERNAL,) * 4 + (NodeRole.MOTOR,) * 4
    graph = Graph(
        num_nodes=12,
        edges=tuple(Edge(s, t, 1.0 + (s + t) % 3) for s, t in sorted(edges)),
        roles=roles,
    )
    return graph, tuple(f"type{i % 3}" for i in range(12))


def _job(
    variant: str, seed: int, accuracy: float, **overrides: object
) -> dict[str, object]:
    job = {
        **META,
        "weights": "log1p",
        "train": 4,
        "test": 2,
        "variant": variant,
        "seed": seed,
        "accuracy": accuracy,
        "oracle": 0.9,
    }
    job.update(overrides)
    return job


def test_every_variant_builds_and_unknown_fails() -> None:
    graph, groups = _graph()
    for name in VARIANTS:
        masked, config = build_variant(name, graph, groups, CONFIG, seed=1)
        assert len(masked.edges) == len(graph.edges)
        assert config.train_recurrent == (name != "FrozenFly")
    with pytest.raises(ValueError, match="unknown variant"):
        build_variant("Nope", graph, groups, CONFIG, seed=1)


def test_run_job_is_deterministic() -> None:
    graph, groups = _graph()
    first = run_job(graph, groups, CONFIG, 3, "ShuffledFly", 4, 2)
    second = run_job(graph, groups, CONFIG, 3, "ShuffledFly", 4, 2)
    assert first == second
    assert 0.0 <= first[0] <= 1.0


def test_aggregate_matches_sequential_summary() -> None:
    scores = {
        seed: {v: 0.1 * i + 0.01 * seed for i, v in enumerate(VARIANTS)}
        for seed in (0, 1)
    }
    jobs = [
        _job(v, seed, acc) for seed, row in scores.items() for v, acc in row.items()
    ]
    expected = summarize(META, [0, 1], [scores[0], scores[1]], [0.9, 0.9])
    assert aggregate(list(reversed(jobs))) == expected


def test_aggregate_rejects_incomplete_duplicate_or_mixed_jobs() -> None:
    complete = [_job(v, 0, 0.5) for v in VARIANTS]
    with pytest.raises(ValueError, match="missing jobs: RandomMask/0"):
        aggregate(complete[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        aggregate([*complete, complete[0]])
    with pytest.raises(ValueError, match="differs in"):
        aggregate([*complete[:-1], _job("RandomMask", 0, 0.5, train=8)])
