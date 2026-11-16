"""Merge per-job results of ``run_trained --variant --seed`` into the usual summary.

Each cloud array task writes one JSON for one (variant, seed). This script checks that
all jobs share the same setup and that every seed has every variant, then writes the same
summary format as the sequential ``run_trained`` runner.

Example:
    PYTHONPATH=src python -m neuroflynav.experiments.aggregate_trained \\
        results/trained/jobs --output results/trained/whole_cx.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from neuroflynav.experiments.run_trained import VARIANTS, summarize

_META_KEYS = ("core", "nodes", "edges", "training", "weights", "train", "test")


def aggregate(jobs: Sequence[dict[str, object]]) -> dict[str, object]:
    """Summary of a complete variant × seed grid of job results."""
    if not jobs:
        raise ValueError("no job results")
    meta = {key: jobs[0][key] for key in _META_KEYS}
    by_seed: dict[int, dict[str, float]] = {}
    oracle: dict[int, float] = {}
    for job in jobs:
        differing = [key for key in _META_KEYS if job[key] != meta[key]]
        if differing:
            raise ValueError(
                f"job {job['variant']}/{job['seed']} differs in {differing}"
            )
        seed = int(job["seed"])  # type: ignore[call-overload]
        variant = str(job["variant"])
        scores = by_seed.setdefault(seed, {})
        if variant in scores:
            raise ValueError(f"duplicate job {variant}/{seed}")
        scores[variant] = float(job["accuracy"])  # type: ignore[arg-type]
        oracle[seed] = float(job["oracle"])  # type: ignore[arg-type]
    missing = [
        f"{v}/{seed}"
        for seed, scores in by_seed.items()
        for v in VARIANTS
        if v not in scores
    ]
    if missing:
        raise ValueError(f"missing jobs: {', '.join(sorted(missing))}")
    seeds = sorted(by_seed)
    per_seed = [{v: by_seed[seed][v] for v in VARIANTS} for seed in seeds]
    summary_meta = {key: meta[key] for key in ("core", "nodes", "edges", "training")}
    return summarize(summary_meta, seeds, per_seed, [oracle[s] for s in seeds])


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge per-job trained-mask results.")
    parser.add_argument("jobs_dir", type=Path, help="directory with one JSON per job")
    parser.add_argument("--output", type=Path, default=Path("results/trained/run.json"))
    arguments = parser.parse_args(argv)
    paths = sorted(arguments.jobs_dir.glob("*.json"))
    result = aggregate([json.loads(path.read_text()) for path in paths])
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(f"merged {len(paths)} jobs -> {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
