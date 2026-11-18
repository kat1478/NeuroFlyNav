"""Heading integration with recurrent weights trained on a graph mask (exploratory).

Every graph gets the same model and the same training budget; only the mask (which edges
exist) differs:

- FlyConnectome — the real CX (or compass-core) wiring, weights trained;
- FrozenFly — same wiring, recurrent weights frozen (input and readout still trained);
- ShuffledFly — degree-preserving rewiring of the mask;
- TypeShuffledFly — rewiring within cell types;
- RandomMask — the same number of random edges, carrying the real weights in random order.

Examples:
    PYTHONPATH=src python -m neuroflynav.experiments.run_trained --core --seeds 3
    # one (variant, seed) job, as run by one cloud array task; merge with aggregate_trained
    PYTHONPATH=src python -m neuroflynav.experiments.run_trained --core \
        --variant ShuffledFly --seed 4 --output results/trained/jobs/ShuffledFly-4.json
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


VARIANTS = (
    "FlyConnectome",
    "FrozenFly",
    "ShuffledFly",
    "TypeShuffledFly",
    "RandomMask",
)


def job_from_index(index: int, first_seed: int) -> tuple[str, int]:
    """Map an array-job index to (variant, seed): five consecutive indices per seed."""
    if index < 0:
        raise ValueError("array index must be non-negative")
    return VARIANTS[index % len(VARIANTS)], first_seed + index // len(VARIANTS)


def _dataset(task: HeadingTaskConfig, seeds: range) -> tuple[np.ndarray, np.ndarray]:
    pairs = [generate_sequence(task, seed) for seed in seeds]
    return np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])


def heading_task() -> HeadingTaskConfig:
    return HeadingTaskConfig(landmark_every=8, turn_std=0.15)


def build_variant(
    name: str,
    graph: Graph,
    groups: Sequence[str],
    config: TrainingConfig,
    seed: int,
) -> tuple[Graph, TrainingConfig]:
    """The mask and training config of one variant for one seed."""
    if name == "FlyConnectome":
        return graph, config
    if name == "FrozenFly":
        return graph, replace(config, train_recurrent=False)
    if name == "ShuffledFly":
        return shuffle_graph(graph, seed=seed), config
    if name == "TypeShuffledFly":
        return shuffle_within_target_groups(graph, groups, seed=seed), config
    if name == "RandomMask":
        return random_mask(graph, seed + 2), config
    raise ValueError(f"unknown variant {name!r}; expected one of {VARIANTS}")


def run_job(
    graph: Graph,
    groups: Sequence[str],
    config: TrainingConfig,
    seed: int,
    variant: str,
    num_train: int,
    num_test: int,
) -> tuple[float, float]:
    """Train and score one (variant, seed); return (accuracy, oracle accuracy)."""
    task = heading_task()
    mask = scored_steps(task).tolist()
    train = _dataset(task, range(seed * 10_000, seed * 10_000 + num_train))
    test_seeds = range(seed * 10_000 + 5_000, seed * 10_000 + 5_000 + num_test)
    test = _dataset(task, test_seeds)
    masked, job_config = build_variant(variant, graph, groups, config, seed)
    accuracy = train_and_score(
        masked, train, test, mask, task.sectors, job_config, seed
    )
    return accuracy, hold_last_landmark_accuracy(task, test_seeds)


def summarize(
    meta: dict[str, object],
    seeds: Sequence[int],
    per_seed: Sequence[dict[str, float]],
    oracle: Sequence[float],
) -> dict[str, object]:
    """The summary JSON written by the sequential runner and by aggregate_trained."""
    policies = list(per_seed[0])
    return {
        **meta,
        "seeds": list(seeds),
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
        "per_seed": list(per_seed),
    }


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
    parser.add_argument(
        "--variant", choices=VARIANTS, help="run one job: this variant only"
    )
    parser.add_argument("--seed", type=int, help="run one job: this seed only")
    parser.add_argument(
        "--array-index",
        type=int,
        help="run one job chosen by array index (with --first-seed), as on AWS Batch",
    )
    parser.add_argument("--output", type=Path, default=Path("results/trained/run.json"))
    arguments = parser.parse_args(argv)
    if arguments.array_index is not None:
        if arguments.variant is not None or arguments.seed is not None:
            parser.error("--array-index replaces --variant and --seed")
        arguments.variant, arguments.seed = job_from_index(
            arguments.array_index, arguments.first_seed
        )
    if (arguments.variant is None) != (arguments.seed is None):
        parser.error("--variant and --seed must be given together")

    subgraph = load_cx_subgraph(arguments.data_dir)
    if arguments.core:
        subgraph = restrict(subgraph, COMPASS_CORE_PREFIXES)
    graph = to_graph(subgraph, arguments.weights)
    groups = node_groups(subgraph)
    config = TrainingConfig(steps=arguments.steps, learning_rate=arguments.lr)
    print(f"graph: {graph.num_nodes} nodes, {len(graph.edges)} edges", file=sys.stderr)
    meta: dict[str, object] = {
        "core": arguments.core,
        "nodes": graph.num_nodes,
        "edges": len(graph.edges),
        "training": asdict(config),
    }

    if arguments.variant is not None:
        started = time.monotonic()
        accuracy, oracle_accuracy = run_job(
            graph,
            groups,
            config,
            arguments.seed,
            arguments.variant,
            arguments.train,
            arguments.test,
        )
        result: dict[str, object] = {
            **meta,
            "weights": arguments.weights,
            "train": arguments.train,
            "test": arguments.test,
            "variant": arguments.variant,
            "seed": arguments.seed,
            "accuracy": accuracy,
            "oracle": oracle_accuracy,
            "seconds": round(time.monotonic() - started, 1),
        }
    else:
        seeds = list(
            range(arguments.first_seed, arguments.first_seed + arguments.seeds)
        )
        per_seed: list[dict[str, float]] = []
        oracle: list[float] = []
        for seed in seeds:
            started = time.monotonic()
            scores: dict[str, float] = {}
            for name in VARIANTS:
                scores[name], seed_oracle = run_job(
                    graph,
                    groups,
                    config,
                    seed,
                    name,
                    arguments.train,
                    arguments.test,
                )
            per_seed.append(scores)
            oracle.append(seed_oracle)
            summary = ", ".join(f"{p} {s:.3f}" for p, s in scores.items())
            print(
                f"seed {seed} ({time.monotonic() - started:.0f}s): {summary}, "
                f"oracle {oracle[-1]:.3f}",
                file=sys.stderr,
            )
        result = summarize(meta, seeds, per_seed, oracle)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
