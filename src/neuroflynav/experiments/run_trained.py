"""Heading integration with recurrent weights trained on a graph mask (exploratory, TASK-014).

Every graph gets the same model and the same training budget; only the mask (which edges
exist) differs:

- FlyConnectome — the real CX (or compass-core) wiring, weights trained;
- FrozenFly — same wiring, recurrent weights frozen (input and readout still trained);
- ShuffledFly — degree-preserving rewiring of the mask;
- TypeShuffledFly — rewiring within cell types;
- RandomMask — the same number of random edges, carrying the real weights in random order.

Example:
    PYTHONPATH=src python -m neuroflynav.experiments.run_trained --core --seeds 3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from neuroflynav.experiments.run_m1_seeds import estimate
from neuroflynav.graph.hemibrain import (
    COMPASS_CORE_PREFIXES,
    DEFAULT_DATA_DIR,
    load_cx_subgraph,
    node_groups,
    restrict,
    to_graph,
)
from neuroflynav.graph.interface import Edge, Graph
from neuroflynav.graph.shuffle import shuffle_graph, shuffle_within_target_groups
from neuroflynav.tasks.heading import (
    HeadingTaskConfig,
    generate_sequence,
    hold_last_landmark_accuracy,
    scored_steps,
)
from neuroflynav.training.masked_rnn import TrainingConfig, train_and_score


def random_mask(graph: Graph, seed: int) -> Graph:
    """Same edge count and roles; random endpoints; the real weights in random order."""
    rng = np.random.default_rng(seed)
    weights = rng.permutation([edge.weight for edge in graph.edges])
    edges: list[Edge] = []
    while len(edges) < len(weights):
        source, target = (int(v) for v in rng.integers(graph.num_nodes, size=2))
        if source != target:
            edges.append(Edge(source, target, float(weights[len(edges)])))
    return Graph(num_nodes=graph.num_nodes, edges=tuple(edges), roles=graph.roles)


def _dataset(task: HeadingTaskConfig, seeds: range) -> tuple[np.ndarray, np.ndarray]:
    pairs = [generate_sequence(task, seed) for seed in seeds]
    return np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Heading with trained masked RNNs.")
    parser.add_argument("--core", action="store_true", help="compass core only")
    parser.add_argument("--weights", default="log1p")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--train", type=int, default=256)
    parser.add_argument("--test", type=int, default=64)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--output", type=Path, default=Path("results/trained/run.json"))
    arguments = parser.parse_args(argv)

    subgraph = load_cx_subgraph(arguments.data_dir)
    if arguments.core:
        subgraph = restrict(subgraph, COMPASS_CORE_PREFIXES)
    graph = to_graph(subgraph, arguments.weights)
    groups = node_groups(subgraph)
    task = HeadingTaskConfig(landmark_every=8, turn_std=0.15)
    mask = scored_steps(task).tolist()
    config = TrainingConfig(steps=arguments.steps, learning_rate=arguments.lr)
    print(f"graph: {graph.num_nodes} nodes, {len(graph.edges)} edges", file=sys.stderr)

    per_seed: list[dict[str, float]] = []
    oracle: list[float] = []
    for seed in range(arguments.first_seed, arguments.first_seed + arguments.seeds):
        started = time.monotonic()
        train = _dataset(task, range(seed * 10_000, seed * 10_000 + arguments.train))
        test_seeds = range(
            seed * 10_000 + 5_000, seed * 10_000 + 5_000 + arguments.test
        )
        test = _dataset(task, test_seeds)
        variants: dict[str, tuple[Graph, TrainingConfig]] = {
            "FlyConnectome": (graph, config),
            "FrozenFly": (graph, replace(config, train_recurrent=False)),
            "ShuffledFly": (shuffle_graph(graph, seed=seed), config),
            "TypeShuffledFly": (
                shuffle_within_target_groups(graph, groups, seed=seed),
                config,
            ),
            "RandomMask": (random_mask(graph, seed + 2), config),
        }
        scores = {
            name: train_and_score(g, train, test, mask, task.sectors, c, seed)
            for name, (g, c) in variants.items()
        }
        per_seed.append(scores)
        oracle.append(hold_last_landmark_accuracy(task, test_seeds))
        summary = ", ".join(f"{p} {s:.3f}" for p, s in scores.items())
        print(
            f"seed {seed} ({time.monotonic() - started:.0f}s): {summary}, "
            f"oracle {oracle[-1]:.3f}",
            file=sys.stderr,
        )

    policies = list(per_seed[0])
    result = {
        "core": arguments.core,
        "nodes": graph.num_nodes,
        "edges": len(graph.edges),
        "seeds": [arguments.first_seed + i for i in range(arguments.seeds)],
        "training": asdict(config),
        "accuracy": {p: asdict(estimate([s[p] for s in per_seed])) for p in policies},
        "fly_minus": {
            p: asdict(estimate([s["FlyConnectome"] - s[p] for s in per_seed]))
            for p in policies
            if p != "FlyConnectome"
        },
        "minus_oracle": {
            p: asdict(
                estimate([s[p] - o for s, o in zip(per_seed, oracle, strict=True)])
            )
            for p in policies
        },
        "oracle": asdict(estimate(oracle)),
        "per_seed": per_seed,
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
