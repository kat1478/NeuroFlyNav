"""The network computes the home vector itself (exploratory, path integration).

One (variant, seed) job: train the masked RNN to report, at every step, where the nest is
relative to the fly (map coordinates, ``tasks.homevector``) on synthetic flights of fixed
length; then test it on ``--flights`` foraging flights over central Warsaw (the same flights
as ``run_homing``) and measure how far from the nest the fly would land flying back along its
own estimate. Variants as in ``run_trained``. References (independent of the variant): staying
put (error = distance from the nest), and v0.7.0's code-summed home vector from the true
sectors (quantisation only) and from the last landmark (no integration).

Example:
    PYTHONPATH=src python -m neuroflynav.experiments.run_homevector --core \\
        --variant FlyConnectome --seed 100 --flights 50 --output results/homevector/pilot.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from neuroflynav.experiments.run_homing import (
    DEFAULT_NEST,
    DEFAULT_TREATS,
    flights_for,
    padded_inputs,
    summary,
)
from neuroflynav.experiments.run_trained import VARIANTS, build_variant, job_from_index
from neuroflynav.graph.hemibrain import (
    COMPASS_CORE_PREFIXES,
    DEFAULT_DATA_DIR,
    load_cx_subgraph,
    node_groups,
    restrict,
    to_graph,
)
from neuroflynav.tasks.homevector import (
    HOME_SCALE_M,
    along_error,
    dataset,
    endpoint_error,
    scored_steps,
    training_config,
)
from neuroflynav.tasks.homing import (
    HomingConfig,
    homing_error,
    last_landmark_headings,
    load_treats,
    sector_centres,
)
from neuroflynav.training.masked_rnn import (
    TrainingConfig,
    predict_values,
    train_regression,
)


def synthetic_errors(
    predicted: np.ndarray, targets: np.ndarray
) -> tuple[list[float], list[float]]:
    """Endpoint and mean-along-flight errors (m) on synthetic test flights."""
    gaps = np.linalg.norm(predicted - targets, axis=2) * HOME_SCALE_M
    return gaps[:, -1].tolist(), gaps.mean(axis=1).tolist()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Home vector computed by the network.")
    parser.add_argument("--core", action="store_true", help="compass core only")
    parser.add_argument("--weights", default="log1p")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--array-index", type=int)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--train", type=int, default=256)
    parser.add_argument("--length", type=int, default=100, help="training flight steps")
    parser.add_argument("--steps", type=int, default=300, help="Adam steps")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--test", type=int, default=64, help="synthetic test flights")
    parser.add_argument("--flights", type=int, default=100, help="Warsaw test flights")
    parser.add_argument("--treats", type=Path, default=DEFAULT_TREATS)
    parser.add_argument(
        "--output", type=Path, default=Path("results/homevector/job.json")
    )
    arguments = parser.parse_args(argv)
    if arguments.array_index is not None:
        arguments.variant, arguments.seed = job_from_index(
            arguments.array_index, arguments.first_seed
        )
    if arguments.variant is None or arguments.seed is None:
        parser.error("give --variant and --seed, or --array-index")

    started = time.monotonic()
    subgraph = load_cx_subgraph(arguments.data_dir)
    if arguments.core:
        subgraph = restrict(subgraph, COMPASS_CORE_PREFIXES)
    graph = to_graph(subgraph, arguments.weights)
    groups = node_groups(subgraph)
    config = TrainingConfig(steps=arguments.steps, learning_rate=arguments.lr)
    task = training_config(arguments.length)
    seed = arguments.seed
    train = dataset(task, range(seed * 10_000, seed * 10_000 + arguments.train))
    test = dataset(
        task, range(seed * 10_000 + 5_000, seed * 10_000 + 5_000 + arguments.test)
    )
    masked, job_config = build_variant(arguments.variant, graph, groups, config, seed)
    model = train_regression(
        masked, train, scored_steps(task).tolist(), job_config, seed
    )
    synthetic_end, synthetic_along = synthetic_errors(
        predict_values(model, test[0]), test[1]
    )

    homing = HomingConfig()
    treats = load_treats(arguments.treats, *DEFAULT_NEST)
    flights = flights_for(homing, treats, seed, arguments.flights)
    predicted = predict_values(model, padded_inputs(flights, homing.max_steps))
    errors = [endpoint_error(f, p) for f, p in zip(flights, predicted, strict=True)]
    along = [
        along_error(f, p, homing.cue_steps)
        for f, p in zip(flights, predicted, strict=True)
    ]
    stay = [float(np.linalg.norm(f.path[-1])) for f in flights]
    quantised = [
        homing_error(f, sector_centres(f.sectors, homing.sectors), homing.step_m)
        for f in flights
    ]
    no_integration = [
        homing_error(f, last_landmark_headings(f, homing), homing.step_m)
        for f in flights
    ]
    result: dict[str, object] = {
        "variant": arguments.variant,
        "seed": seed,
        "core": arguments.core,
        "nodes": graph.num_nodes,
        "edges": len(graph.edges),
        "weights": arguments.weights,
        "train": arguments.train,
        "length": arguments.length,
        "training": asdict(config),
        "homing": asdict(homing),
        "nest": DEFAULT_NEST,
        "flights": len(flights),
        "mean_steps": round(statistics.fmean(f.steps for f in flights), 2),
        "error_m": summary(errors),
        "along_error_m": summary(along),
        "synthetic_error_m": summary(synthetic_end),
        "synthetic_along_error_m": summary(synthetic_along),
        "stay_error_m": summary(stay),
        "quantised_error_m": summary(quantised),
        "no_integration_error_m": summary(no_integration),
        "errors_m": [round(e, 2) for e in errors],
        "seconds": round(time.monotonic() - started, 1),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2))
    brief = {
        k: result[k]
        for k in ("error_m", "synthetic_error_m", "no_integration_error_m", "seconds")
    }
    print(json.dumps(brief), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
