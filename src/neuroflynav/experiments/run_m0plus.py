"""Compare reservoir policies on the hidden-cue task (development spike, M0+).

Builds a frozen graph substrate from the synthetic CX-like graph (FlyConnectome),
a degree-preserving shuffle of it (ShuffledFly), a matched-edge random graph
(RandomRecurrent), and a memoryless baseline (Reactive). Each trains only a linear
readout by imitation, then is evaluated on the hidden-cue task and on a harder
shorter-cue robustness probe.

This is a NON-AUTHORITATIVE development spike (see private ops D-032). All topology
is synthetic and all dynamics/training choices are placeholders. The comparison has
no statistical or biological interpretation; its purpose is that the machinery runs,
is reproducible, and *can* distinguish topologies — not that any policy is better.

Example:
    PYTHONPATH=src python -m neuroflynav.experiments.run_m0plus --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass

from neuroflynav.agent.reservoir import (
    LinearPolicy,
    MemoryPolicy,
    ReactiveRepresentation,
    Representation,
    ReservoirConfig,
    ReservoirRepresentation,
    build_core_from_graph,
    build_random_core,
)
from neuroflynav.env.nav_memory import MemoryGridNavConfig, MemoryGridNavEnv
from neuroflynav.graph.interface import Graph, NodeRole
from neuroflynav.graph.shuffle import shuffle_graph
from neuroflynav.graph.synthetic import SyntheticCXGraphLoader
from neuroflynav.training.imitation import collect_dataset, fit_ridge


@dataclass(frozen=True)
class PolicyMetrics:
    """Evaluation metrics for one policy on the hidden-cue task."""

    policy: str
    graph: str
    success_rate: float
    mean_steps: float
    robust_success_rate: float


@dataclass(frozen=True)
class M0PlusSummary:
    """Results for all M0+ policies under shared seeds and a shared task."""

    grid_size: int
    cue_steps: int
    robust_cue_steps: int
    train_episodes: int
    eval_episodes: int
    base_seed: int
    graph_nodes: int
    graph_edges: int
    policies: tuple[PolicyMetrics, ...]


def evaluate(
    env: MemoryGridNavEnv, policy: MemoryPolicy, seeds: Sequence[int]
) -> tuple[float, float]:
    """Return (success_rate, mean_steps) for a policy over the given episode seeds."""
    successes = 0
    total_steps = 0
    for seed in seeds:
        observation, _ = env.reset(seed)
        policy.reset()
        steps = 0
        for _ in range(env.config.max_steps):
            observation, _, terminated, truncated, _ = env.step(policy.act(observation))
            steps += 1
            if terminated:
                successes += 1
                break
            if truncated:
                break
        total_steps += steps
    count = len(seeds)
    return successes / count, total_steps / count


def _mean_edge_weight(graph: Graph) -> float:
    if not graph.edges:
        return 1.0
    return sum(edge.weight for edge in graph.edges) / len(graph.edges)


def _named_representations(
    graph: Graph, config: ReservoirConfig, base_seed: int
) -> list[tuple[str, str, Representation]]:
    sensory = len(graph.nodes_with_role(NodeRole.SENSORY))
    motor = len(graph.nodes_with_role(NodeRole.MOTOR))
    fly_core = build_core_from_graph(graph, config, seed=base_seed)
    shuffled_graph = shuffle_graph(graph, seed=base_seed)
    shuffled_core = build_core_from_graph(shuffled_graph, config, seed=base_seed + 1)
    random_core = build_random_core(
        num_nodes=graph.num_nodes,
        num_edges=len(graph.edges),
        num_sensory=sensory,
        num_motor=motor,
        weight=_mean_edge_weight(graph),
        config=config,
        seed=base_seed + 2,
    )
    return [
        (
            "FlyConnectome",
            "synthetic",
            ReservoirRepresentation(fly_core, config.grid_size),
        ),
        (
            "ShuffledFly",
            f"shuffled(seed={base_seed})",
            ReservoirRepresentation(shuffled_core, config.grid_size),
        ),
        (
            "RandomRecurrent",
            "random(matched-edges)",
            ReservoirRepresentation(random_core, config.grid_size),
        ),
        ("Reactive", "none", ReactiveRepresentation(config.grid_size)),
    ]


def run_m0plus(
    grid_size: int = 7,
    cue_steps: int = 3,
    robust_cue_steps: int = 1,
    max_steps: int = 60,
    train_episodes: int = 80,
    eval_episodes: int = 40,
    base_seed: int = 0,
) -> M0PlusSummary:
    """Train and evaluate all M0+ policies; return a reproducible summary."""
    if train_episodes < 1 or eval_episodes < 1:
        raise ValueError("train_episodes and eval_episodes must be at least 1")

    graph = SyntheticCXGraphLoader(seed=base_seed).load()
    config = ReservoirConfig(grid_size=grid_size)
    train_env = MemoryGridNavEnv(
        MemoryGridNavConfig(size=grid_size, max_steps=max_steps, cue_steps=cue_steps)
    )
    eval_env = train_env
    robust_env = MemoryGridNavEnv(
        MemoryGridNavConfig(
            size=grid_size, max_steps=max_steps, cue_steps=robust_cue_steps
        )
    )
    train_seeds = tuple(range(base_seed, base_seed + train_episodes))
    eval_seeds = tuple(range(base_seed + 100_000, base_seed + 100_000 + eval_episodes))

    metrics: list[PolicyMetrics] = []
    for name, graph_name, representation in _named_representations(
        graph, config, base_seed
    ):
        features, targets = collect_dataset(train_env, representation, train_seeds)
        readout = fit_ridge(features, targets, ridge_lambda=1e-2)
        policy: MemoryPolicy = LinearPolicy(representation, readout)
        success_rate, mean_steps = evaluate(eval_env, policy, eval_seeds)
        robust_success_rate, _ = evaluate(robust_env, policy, eval_seeds)
        metrics.append(
            PolicyMetrics(
                policy=name,
                graph=graph_name,
                success_rate=success_rate,
                mean_steps=mean_steps,
                robust_success_rate=robust_success_rate,
            )
        )

    return M0PlusSummary(
        grid_size=grid_size,
        cue_steps=cue_steps,
        robust_cue_steps=robust_cue_steps,
        train_episodes=train_episodes,
        eval_episodes=eval_episodes,
        base_seed=base_seed,
        graph_nodes=graph.num_nodes,
        graph_edges=len(graph.edges),
        policies=tuple(metrics),
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare M0+ reservoir policies on the hidden-cue task (spike)."
    )
    parser.add_argument("--grid-size", type=int, default=7)
    parser.add_argument("--cue-steps", type=int, default=3)
    parser.add_argument("--robust-cue-steps", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=60)
    parser.add_argument("--train", type=int, default=80)
    parser.add_argument("--eval", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the M0+ comparison and print a JSON summary to stdout."""
    arguments = _argument_parser().parse_args(argv)
    try:
        summary = run_m0plus(
            grid_size=arguments.grid_size,
            cue_steps=arguments.cue_steps,
            robust_cue_steps=arguments.robust_cue_steps,
            max_steps=arguments.max_steps,
            train_episodes=arguments.train,
            eval_episodes=arguments.eval,
            base_seed=arguments.seed,
        )
    except ValueError as error:
        print(f"run_m0plus: error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
