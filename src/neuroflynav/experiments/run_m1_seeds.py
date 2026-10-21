"""Repeat the M0+ comparison over many seeds on the hemibrain CX graph (spike, M1).

For each weight transform the CX graph is built once; every seed then re-draws the
training/evaluation episodes, the ShuffledFly rewiring, the RandomRecurrent graph and
the input weights. The output reports, per policy, the mean success rate with a 95%
normal-approximation confidence interval, plus the paired per-seed difference
FlyConnectome − each control (same episodes, so the pairing removes task noise).

Exploratory only: placeholder dynamics, readout and sensory/motor conventions.

Example:
    PYTHONPATH=src python -m neuroflynav.experiments.run_m1_seeds --seeds 10
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from neuroflynav.experiments.run_m0plus import M0PlusSummary, run_m0plus
from neuroflynav.graph.hemibrain import (
    DEFAULT_DATA_DIR,
    WEIGHT_TRANSFORMS,
    load_cx_subgraph,
    node_groups,
    to_graph,
)

_Z95 = 1.96


@dataclass(frozen=True)
class Estimate:
    """Mean over seeds with a 95% normal-approximation confidence interval."""

    mean: float
    low: float
    high: float


@dataclass(frozen=True)
class TransformResult:
    """Aggregated results for one weight transform."""

    weights: str
    seeds: tuple[int, ...]
    success: dict[str, Estimate]
    robust_success: dict[str, Estimate]
    fly_minus: dict[str, Estimate]
    fly_minus_robust: dict[str, Estimate]


def estimate(values: Sequence[float]) -> Estimate:
    """Return the mean and a 95% CI half-width of 1.96 standard errors."""
    if not values:
        raise ValueError("need at least one value")
    mean = statistics.fmean(values)
    if len(values) < 2:
        return Estimate(mean, mean, mean)
    half = _Z95 * statistics.stdev(values) / math.sqrt(len(values))
    return Estimate(mean, mean - half, mean + half)


def _rates(summary: M0PlusSummary, robust: bool) -> dict[str, float]:
    return {
        metric.policy: metric.robust_success_rate if robust else metric.success_rate
        for metric in summary.policies
    }


def aggregate(
    weights: str, seeds: Sequence[int], summaries: Sequence[M0PlusSummary]
) -> TransformResult:
    """Combine per-seed summaries into per-policy estimates and paired differences."""
    plain = [_rates(summary, robust=False) for summary in summaries]
    robust = [_rates(summary, robust=True) for summary in summaries]
    policies = list(plain[0])
    controls = [p for p in policies if p != "FlyConnectome"]
    return TransformResult(
        weights=weights,
        seeds=tuple(seeds),
        success={p: estimate([rates[p] for rates in plain]) for p in policies},
        robust_success={p: estimate([rates[p] for rates in robust]) for p in policies},
        fly_minus={
            p: estimate([rates["FlyConnectome"] - rates[p] for rates in plain])
            for p in controls
        },
        fly_minus_robust={
            p: estimate([rates["FlyConnectome"] - rates[p] for rates in robust])
            for p in controls
        },
    )


def _format(value: Estimate) -> str:
    return f"{value.mean:+.3f} [{value.low:+.3f}, {value.high:+.3f}]"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Multi-seed M0+ on hemibrain CX.")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument(
        "--type-shuffle",
        action="store_true",
        help="add the cell-type-preserving TypeShuffledFly control (H2)",
    )
    parser.add_argument(
        "--weights",
        nargs="+",
        choices=sorted(WEIGHT_TRANSFORMS),
        default=["raw", "log1p", "binary"],
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=Path("results/m1/seeds.json"))
    arguments = parser.parse_args(argv)
    if arguments.seeds < 1:
        print("run_m1_seeds: error: --seeds must be at least 1", file=sys.stderr)
        return 1

    subgraph = load_cx_subgraph(arguments.data_dir)
    seeds = tuple(range(arguments.first_seed, arguments.first_seed + arguments.seeds))
    groups = node_groups(subgraph) if arguments.type_shuffle else None
    results: list[TransformResult] = []
    for weights in arguments.weights:
        graph = to_graph(subgraph, weights)
        summaries = []
        for seed in seeds:
            started = time.monotonic()
            summaries.append(
                run_m0plus(
                    base_seed=seed,
                    graph=graph,
                    graph_label=f"hemibrain-cx(weights={weights})",
                    node_groups=groups,
                )
            )
            print(
                f"{weights} seed {seed} done in {time.monotonic() - started:.0f}s",
                file=sys.stderr,
            )
        result = aggregate(weights, seeds, summaries)
        results.append(result)
        for control, difference in result.fly_minus.items():
            print(
                f"{weights}: Fly − {control} success {_format(difference)}",
                file=sys.stderr,
            )

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps([asdict(result) for result in results], indent=2, sort_keys=True)
    )
    print(json.dumps([asdict(result) for result in results], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
