"""Merge per-job results of ``run_homevector`` into one summary.

Checks that every seed has every variant with the same setup, then reports per variant the
mean homing error (Warsaw flights, endpoint), the mean error along the flights and on
synthetic test flights (averaged over seeds, 95% CI over seeds), and paired differences
FlyConnectome − variant of the per-seed mean homing error (negative = real wiring lands
closer). Variant-independent references (stay put, quantised sectors, no integration) are
averaged too.

Example:
    PYTHONPATH=src python -m neuroflynav.experiments.aggregate_homevector \\
        results/homevector/jobs --output results/homevector/summary.json
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
    "length",
    "training",
    "homing",
    "nest",
    "flights",
)
_METRICS = ("error_m", "along_error_m", "synthetic_error_m")
_REFERENCES = ("stay_error_m", "quantised_error_m", "no_integration_error_m")


def aggregate(jobs: Sequence[dict[str, object]]) -> dict[str, object]:
    if not jobs:
        raise ValueError("no job results")
    setup = {key: jobs[0][key] for key in _SETUP}
    values: dict[str, dict[int, dict[str, float]]] = {m: {} for m in _METRICS}
    references: dict[int, dict[str, float]] = {}
    for job in jobs:
        differing = [key for key in _SETUP if job[key] != setup[key]]
        if differing:
            raise ValueError(
                f"job {job['variant']}/{job['seed']} differs in {differing}"
            )
        seed, variant = int(job["seed"]), str(job["variant"])  # type: ignore[call-overload]
        if variant in values["error_m"].setdefault(seed, {}):
            raise ValueError(f"duplicate job {variant}/{seed}")
        for metric in _METRICS:
            values[metric].setdefault(seed, {})[variant] = float(
                job[metric]["mean"]  # type: ignore[index]
            )
        references[seed] = {
            r: float(job[r]["mean"])  # type: ignore[index]
            for r in _REFERENCES
        }
    error = values["error_m"]
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
    summary: dict[str, object] = {**setup, "seeds": seeds}
    for metric in _METRICS:
        summary[f"mean_{metric}"] = {
            v: asdict(estimate([values[metric][s][v] for s in seeds])) for v in variants
        }
    summary["fly_minus_error_m"] = {
        v: asdict(estimate([error[s]["FlyConnectome"] - error[s][v] for s in seeds]))
        for v in variants
        if v != "FlyConnectome"
    }
    for reference in _REFERENCES:
        summary[reference] = asdict(estimate([references[s][reference] for s in seeds]))
    summary["per_seed_error_m"] = [{v: error[s][v] for v in variants} for s in seeds]
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge per-job home-vector results.")
    parser.add_argument("jobs_dir", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("results/homevector/summary.json")
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
