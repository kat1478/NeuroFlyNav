"""Compare M0 navigation policies on a synthetic graph and grid task.

This is a non-authoritative development spike, not a scientific experiment. All
topology is synthetic; route scoring and shuffled-control choices are arbitrary
computational placeholders, not biological facts or frozen decisions.

Example:
    PYTHONPATH=src python -m neuroflynav.experiments.run_spike --episodes 20 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

from neuroflynav.agent.base import Policy
from neuroflynav.agent.fly_connectome import FlyConnectome
from neuroflynav.agent.random_agent import RandomAgent
from neuroflynav.agent.shuffled_fly import ShuffledFly
from neuroflynav.env.nav import GridNavConfig, GridNavEnv
from neuroflynav.graph.synthetic import SyntheticCXGraphLoader


@dataclass(frozen=True)
class EpisodeResult:
    """The outcome of a single episode."""

    success: bool
    steps: int
    total_reward: float


@dataclass(frozen=True)
class PolicySummary:
    """Aggregated outcomes for one policy and its named topology."""

    policy: str
    graph: str
    successes: int
    success_rate: float
    mean_steps: float
    mean_reward: float


@dataclass(frozen=True)
class SpikeSummary:
    """Results for all M0 policies under a shared set of episode seeds."""

    episodes: int
    base_seed: int
    episode_seeds: tuple[int, ...]
    graph_nodes: int
    graph_edges: int
    shuffle_seed: int
    policies: tuple[PolicySummary, ...]


def run_episode(env: GridNavEnv, policy: Policy, seed: int) -> EpisodeResult:
    """Run one episode to termination or truncation and report its outcome."""
    observation, _ = env.reset(seed)
    total_reward = 0.0
    steps = 0
    while True:
        action = policy.act(observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        steps += 1
        if terminated or truncated:
            return EpisodeResult(
                success=terminated, steps=steps, total_reward=total_reward
            )


def _summarize(
    policy_name: str,
    graph_name: str,
    results: list[EpisodeResult],
) -> PolicySummary:
    episodes = len(results)
    successes = sum(result.success for result in results)
    return PolicySummary(
        policy=policy_name,
        graph=graph_name,
        successes=successes,
        success_rate=successes / episodes,
        mean_steps=sum(result.steps for result in results) / episodes,
        mean_reward=sum(result.total_reward for result in results) / episodes,
    )


def run_spike(
    episodes: int = 20,
    grid_size: int = 5,
    max_steps: int = 50,
    base_seed: int = 0,
) -> SpikeSummary:
    """Run RandomAgent, FlyConnectome and ShuffledFly on shared episode seeds.

    The synthetic topology and its shuffled control are fixed for this run. Each
    episode resets the environment with ``base_seed + index`` for every policy;
    RandomAgent also receives that episode seed. The topology shuffle uses the
    run's base seed. This seed policy is reproducibility plumbing for M0, not a
    frozen experimental seed plan.
    """
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    graph = SyntheticCXGraphLoader(seed=base_seed).load()
    shuffle_seed = base_seed
    shuffled_policy = ShuffledFly(graph, seed=shuffle_seed)
    env = GridNavEnv(GridNavConfig(size=grid_size, max_steps=max_steps))
    episode_seeds = tuple(base_seed + index for index in range(episodes))

    policy_specs: tuple[tuple[str, str, Callable[[int], Policy]], ...] = (
        ("RandomAgent", "synthetic", lambda seed: RandomAgent(seed=seed)),
        ("FlyConnectome", "synthetic", lambda _seed: FlyConnectome(graph)),
        (
            "ShuffledFly",
            f"shuffled(seed={shuffle_seed})",
            lambda _seed: shuffled_policy,
        ),
    )
    summaries: list[PolicySummary] = []
    for policy_name, graph_name, policy_factory in policy_specs:
        results = [
            run_episode(env, policy_factory(episode_seed), episode_seed)
            for episode_seed in episode_seeds
        ]
        summaries.append(_summarize(policy_name, graph_name, results))

    return SpikeSummary(
        episodes=episodes,
        base_seed=base_seed,
        episode_seeds=episode_seeds,
        graph_nodes=graph.num_nodes,
        graph_edges=len(graph.edges),
        shuffle_seed=shuffle_seed,
        policies=tuple(summaries),
    )


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare M0 policies on a synthetic graph (non-scientific spike)."
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the spike and print a JSON summary to stdout."""
    arguments = _argument_parser().parse_args(argv)
    try:
        summary = run_spike(
            episodes=arguments.episodes,
            grid_size=arguments.grid_size,
            max_steps=arguments.max_steps,
            base_seed=arguments.seed,
        )
    except ValueError as error:
        print(f"run_spike: error: {error}", file=sys.stderr)
        return 1
    print(json.dumps(asdict(summary), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
