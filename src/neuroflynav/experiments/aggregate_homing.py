"""Merge per-job results of ``run_homing`` into one summary.

Checks that every seed has every variant with the same setup, then reports per variant the
mean homing error and heading accuracy (averaged over seeds, 95% CI over seeds), and paired
differences FlyConnectome − variant of the per-seed mean error (negative = real wiring homes
closer). Variant-independent references (quantisation only, no integration) are averaged too.

Example:
    PYTHONPATH=src python -m neuroflynav.experiments.aggregate_homing \\
        results/homing/jobs --output results/homing/summary.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from neuroflynav.experiments.run_m1_seeds import estimate
from neuroflynav.experiments.run_trained import VARIANTS

_SETUP = (
    "core",
    "nodes",
    "edges",
    "weights",
    "train",
    "training",
    "homing",
    "nest",
    "flights",
)


def aggregate(jobs: Sequence[dict[str, object]]) -> dict[str, object]:
    if not jobs:
        raise ValueError("no job results")
    setup = {key: jobs[0][key] for key in _SETUP}
    error: dict[int, dict[str, float]] = {}
    accuracy: dict[int, dict[str, float]] = {}
    references: dict[int, tuple[float, float]] = {}
    for job in jobs:
        differing = [key for key in _SETUP if job[key] != setup[key]]
        if differing:
            raise ValueError(
                f"job {job['variant']}/{job['seed']} differs in {differing}"
            )
        seed, variant = int(job["seed"]), str(job["variant"])  # type: ignore[call-overload]
        if variant in error.setdefault(seed, {}):
            raise ValueError(f"duplicate job {variant}/{seed}")
        error[seed][variant] = float(job["error_m"]["mean"])  # type: ignore[index]
        accuracy.setdefault(seed, {})[variant] = float(job["heading_accuracy"])  # type: ignore[arg-type]
        references[seed] = (
            float(job["quantised_error_m"]["mean"]),  # type: ignore[index]
            float(job["no_integration_error_m"]["mean"]),  # type: ignore[index]
        )
    variants = [v for v in VARIANTS if all(v in row for row in error.values())]
    missing = sorted(
        f"{v}/{s}"
        for s, row in error.items()
        for v in set().union(*error.values())
        if v not in row
    )
    if missing or "FlyConnectome" not in variants:
        raise ValueError(f"missing jobs: {', '.join(missing) or 'FlyConnectome'}")
    seeds = sorted(error)
    return {
        **setup,
        "seeds": seeds,
        "mean_error_m": {
            v: asdict(estimate([error[s][v] for s in seeds])) for v in variants
        },
        "heading_accuracy": {
            v: asdict(estimate([accuracy[s][v] for s in seeds])) for v in variants
        },
        "fly_minus_error_m": {
            v: asdict(
                estimate([error[s]["FlyConnectome"] - error[s][v] for s in seeds])
            )
            for v in variants
            if v != "FlyConnectome"
        },
        "quantised_error_m": asdict(estimate([references[s][0] for s in seeds])),
        "no_integration_error_m": asdict(estimate([references[s][1] for s in seeds])),
        "per_seed_error_m": [{v: error[s][v] for v in variants} for s in seeds],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge per-job homing results.")
    parser.add_argument("jobs_dir", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("results/homing/summary.json")
    )
    arguments = parser.parse_args(argv)
    paths = sorted(arguments.jobs_dir.glob("*.json"))
    result = aggregate([json.loads(p.read_text()) for p in paths])
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"merged {len(paths)} jobs -> {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
