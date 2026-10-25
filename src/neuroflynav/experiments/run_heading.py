"""Heading integration on a frozen reservoir: real CX wiring vs controls (exploratory).

Each policy turns an input sequence (see ``neuroflynav.tasks.heading``) into per-step
features; a ridge readout is fit on training sequences to predict the heading sector
after the cue disappears, then scored on held-out sequences. Policies:

- FlyConnectome — reservoir from the graph;
- ShuffledFly — degree-preserving rewiring;
- TypeShuffledFly — rewiring within cell types (only with node groups, i.e. hemibrain);
- RandomRecurrent — random graph with the same edge count;
- Reactive — the raw input, no memory (cannot integrate).

Dynamics, input injection at ER neurons and the full-state readout are placeholders.

Example:
    PYTHONPATH=src python -m neuroflynav.experiments.run_heading --graph hemibrain --seeds 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from neuroflynav.agent.reservoir import (
    ReservoirConfig,
    ReservoirCore,
    build_core_from_graph,
    build_random_core,
)
from neuroflynav.experiments.run_m1_seeds import Estimate, estimate
from neuroflynav.graph.hemibrain import (
    DEFAULT_DATA_DIR,
    WEIGHT_TRANSFORMS,
    CXSubgraph,
    inhibitory_nodes,
    load_cx_subgraph,
    node_groups,
    random_inhibitory_nodes,
    to_graph,
)
from neuroflynav.graph.interface import Graph, NodeRole
from neuroflynav.graph.shuffle import shuffle_graph, shuffle_within_target_groups
from neuroflynav.graph.synthetic import SyntheticCXGraphLoader
from neuroflynav.tasks.heading import (
    INPUT_DIM,
    HeadingTaskConfig,
    generate_sequence,
    hold_last_landmark_accuracy,
    scored_steps,
)
from neuroflynav.training.imitation import fit_ridge

FloatArray = NDArray[np.float64]
FeatureFn = Callable[[FloatArray], FloatArray]

LATE_STEPS = 10


@dataclass(frozen=True)
class HeadingScore:
    """Sector accuracy after the cue, overall and over the last ``LATE_STEPS`` steps."""

    accuracy: float
    late_accuracy: float


def _reservoir_features(core: ReservoirCore) -> FeatureFn:
    def features(inputs: FloatArray) -> FloatArray:
        core.reset()
        return np.stack([core.step(row) for row in inputs])

    return features


def _policies(
    graph: Graph,
    groups: Sequence[str] | None,
    config: ReservoirConfig,
    seed: int,
    extra_graphs: Mapping[str, Graph] | None = None,
) -> list[tuple[str, FeatureFn]]:
    sensory = len(graph.nodes_with_role(NodeRole.SENSORY))
    motor = len(graph.nodes_with_role(NodeRole.MOTOR))
    mean_weight = sum(abs(edge.weight) for edge in graph.edges) / max(
        len(graph.edges), 1
    )
    negative = sum(edge.weight < 0 for edge in graph.edges) / max(len(graph.edges), 1)
    policies: list[tuple[str, FeatureFn]] = [
        (
            "FlyConnectome",
            _reservoir_features(build_core_from_graph(graph, config, seed, INPUT_DIM)),
        ),
        (
            "ShuffledFly",
            _reservoir_features(
                build_core_from_graph(
                    shuffle_graph(graph, seed=seed), config, seed + 1, INPUT_DIM
                )
            ),
        ),
    ]
    if groups is not None:
        type_graph = shuffle_within_target_groups(graph, groups, seed=seed)
        policies.append(
            (
                "TypeShuffledFly",
                _reservoir_features(
                    build_core_from_graph(type_graph, config, seed + 3, INPUT_DIM)
                ),
            )
        )
    random_core = build_random_core(
        num_nodes=graph.num_nodes,
        num_edges=len(graph.edges),
        num_sensory=sensory,
        num_motor=motor,
        weight=mean_weight,
        config=config,
        seed=seed + 2,
        input_dim=INPUT_DIM,
        inhibitory_fraction=negative,
    )
    policies.append(("RandomRecurrent", _reservoir_features(random_core)))
    policies.append(("Reactive", lambda inputs: inputs))
    for name, extra in (extra_graphs or {}).items():
        policies.append(
            (
                name,
                _reservoir_features(
                    build_core_from_graph(extra, config, seed, INPUT_DIM)
                ),
            )
        )
    return policies


def score_policy(
    features: FeatureFn,
    task: HeadingTaskConfig,
    train_seeds: Sequence[int],
    test_seeds: Sequence[int],
    ridge_lambda: float,
) -> HeadingScore:
    """Fit a ridge readout on scored steps of training sequences; score held-out ones."""
    mask = scored_steps(task)
    late = mask & (np.arange(task.steps) >= task.steps - LATE_STEPS)
    rows: list[FloatArray] = []
    targets: list[FloatArray] = []
    for seed in train_seeds:
        inputs, sectors = generate_sequence(task, seed)
        rows.append(features(inputs)[mask])
        targets.append(np.eye(task.sectors)[sectors[mask]])
    readout = fit_ridge(np.vstack(rows), np.vstack(targets), ridge_lambda)
    hits = 0
    late_hits = 0
    for seed in test_seeds:
        inputs, sectors = generate_sequence(task, seed)
        correct = np.argmax(features(inputs) @ readout.T, axis=1) == sectors
        hits += int(correct[mask].sum())
        late_hits += int(correct[late].sum())
    count = len(test_seeds)
    return HeadingScore(
        hits / (int(mask.sum()) * count), late_hits / max(int(late.sum()) * count, 1)
    )


def run_heading(
    graph: Graph,
    groups: Sequence[str] | None,
    seed: int,
    task: HeadingTaskConfig | None = None,
    train_sequences: int = 100,
    test_sequences: int = 50,
    ridge_lambda: float = 1.0,
    reservoir: ReservoirConfig | None = None,
    extra_graphs: Mapping[str, Graph] | None = None,
) -> dict[str, HeadingScore]:
    """Score every policy for one seed (task sequences and control draws depend on it)."""
    settings = task or HeadingTaskConfig()
    train_seeds = range(seed * 10_000, seed * 10_000 + train_sequences)
    test_seeds = range(seed * 10_000 + 5_000, seed * 10_000 + 5_000 + test_sequences)
    return {
        name: score_policy(features, settings, train_seeds, test_seeds, ridge_lambda)
        for name, features in _policies(
            graph, groups, reservoir or ReservoirConfig(), seed, extra_graphs
        )
    }


def _sign_controls(subgraph: CXSubgraph, weights: str, seed: int) -> dict[str, Graph]:
    """RandomSignFly (same number of inhibitory neurons, random ones) and UnsignedFly."""
    count = len(inhibitory_nodes(subgraph))
    return {
        "RandomSignFly": to_graph(
            subgraph,
            weights,
            inhibitory=random_inhibitory_nodes(subgraph, count, seed + 7),
        ),
        "UnsignedFly": to_graph(subgraph, weights),
    }


def _summarize(
    per_seed: Sequence[dict[str, HeadingScore]], field: str
) -> tuple[dict[str, Estimate], dict[str, Estimate]]:
    policies = list(per_seed[0])
    values = [{p: getattr(s[p], field) for p in policies} for s in per_seed]
    means = {p: estimate([v[p] for v in values]) for p in policies}
    differences = {
        p: estimate([v["FlyConnectome"] - v[p] for v in values])
        for p in policies
        if p != "FlyConnectome"
    }
    return means, differences


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Heading integration on a reservoir.")
    parser.add_argument(
        "--graph", choices=("synthetic", "hemibrain"), default="hemibrain"
    )
    parser.add_argument("--weights", choices=sorted(WEIGHT_TRANSFORMS), default="log1p")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--train", type=int, default=100)
    parser.add_argument("--test", type=int, default=50)
    # Defaults calibrated on pilot sequences (seeds 100+), see docs/spike_heading.md.
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--landmark-every", type=int, default=8)
    parser.add_argument("--turn-std", type=float, default=0.15)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--micro-steps", type=int, default=1)
    parser.add_argument("--radius", type=float, default=1.1)
    parser.add_argument("--leak", type=float, default=1.0)
    parser.add_argument(
        "--signs",
        choices=("none", "types"),
        default="none",
        help="'types': Δ7 and ER outputs inhibitory; adds RandomSignFly and UnsignedFly",
    )
    parser.add_argument("--output", type=Path, default=Path("results/heading/run.json"))
    arguments = parser.parse_args(argv)
    if arguments.seeds < 1:
        print("run_heading: error: --seeds must be at least 1", file=sys.stderr)
        return 1

    groups: tuple[str, ...] | None = None
    if arguments.graph == "hemibrain":
        subgraph = load_cx_subgraph(arguments.data_dir)
        inhibitory = (
            inhibitory_nodes(subgraph) if arguments.signs == "types" else frozenset()
        )
        graph = to_graph(subgraph, arguments.weights, inhibitory=inhibitory)
        groups = node_groups(subgraph)
    else:
        graph = SyntheticCXGraphLoader(seed=0).load()

    seeds = range(arguments.first_seed, arguments.first_seed + arguments.seeds)
    task = HeadingTaskConfig(
        steps=arguments.steps,
        landmark_every=arguments.landmark_every,
        turn_std=arguments.turn_std,
    )
    reservoir = ReservoirConfig(
        micro_steps=arguments.micro_steps,
        spectral_radius=arguments.radius,
        leak=arguments.leak,
    )
    per_seed: list[dict[str, HeadingScore]] = []
    oracle: list[float] = []
    for seed in seeds:
        started = time.monotonic()
        scores = run_heading(
            graph,
            groups,
            seed,
            train_sequences=arguments.train,
            test_sequences=arguments.test,
            ridge_lambda=arguments.ridge,
            task=task,
            reservoir=reservoir,
            extra_graphs=_sign_controls(subgraph, arguments.weights, seed)
            if arguments.graph == "hemibrain" and arguments.signs == "types"
            else None,
        )
        per_seed.append(scores)
        test_start = seed * 10_000 + 5_000
        oracle.append(
            hold_last_landmark_accuracy(
                task, range(test_start, test_start + arguments.test)
            )
        )
        summary = ", ".join(f"{p} {s.accuracy:.3f}" for p, s in scores.items())
        print(
            f"seed {seed} ({time.monotonic() - started:.0f}s): {summary}",
            file=sys.stderr,
        )

    accuracy, accuracy_diff = _summarize(per_seed, "accuracy")
    late, late_diff = _summarize(per_seed, "late_accuracy")
    result = {
        "graph": arguments.graph,
        "weights": arguments.weights,
        "seeds": list(seeds),
        "ridge": arguments.ridge,
        "task": asdict(task),
        "reservoir": asdict(reservoir),
        "signs": arguments.signs,
        "accuracy": {p: asdict(e) for p, e in accuracy.items()},
        "fly_minus": {p: asdict(e) for p, e in accuracy_diff.items()},
        "late_accuracy": {p: asdict(e) for p, e in late.items()},
        "late_fly_minus": {p: asdict(e) for p, e in late_diff.items()},
        "per_seed": [{p: asdict(s) for p, s in scores.items()} for scores in per_seed],
        "hold_last_landmark": asdict(estimate(oracle)),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
