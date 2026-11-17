"""Homing by path integration with a trained heading network (exploratory, "fly in the city").

One (variant, seed) job: train the masked RNN on the heading task exactly as
``run_trained`` does, then fly ``--flights`` foraging flights from the nest through the real
treat layout of central Warsaw and measure how far from the nest each fly ends after flying
back along its estimated home vector. References (independent of the variant): headings
from the true sectors (quantisation only) and from the last landmark (no integration).

Example:
    PYTHONPATH=src python -m neuroflynav.experiments.run_homing --core \\
        --variant FlyConnectome --seed 100 --flights 50 --output results/homing/pilot.json
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

from neuroflynav.experiments.run_trained import (
    VARIANTS,
    _dataset,
    build_variant,
    heading_task,
    job_from_index,
)
from neuroflynav.graph.hemibrain import (
    COMPASS_CORE_PREFIXES,
    DEFAULT_DATA_DIR,
    load_cx_subgraph,
    node_groups,
    restrict,
    to_graph,
)
from neuroflynav.tasks.heading import scored_steps
from neuroflynav.tasks.homing import (
    Flight,
    HomingConfig,
    errors_for_sectors,
    home_vector,
    homing_error,
    last_landmark_headings,
    load_treats,
    outbound_flight,
    sector_centres,
)
from neuroflynav.training.masked_rnn import TrainingConfig, predict, train_model

DEFAULT_TREATS = Path("app/public/data/treats.geojson")
# Nest: Park Świętokrzyski, central Warsaw (lon, lat).
DEFAULT_NEST = (21.0035, 52.2357)


def flights_for(
    config: HomingConfig, treats: np.ndarray, seed: int, count: int
) -> list[Flight]:
    """The flights of one seed; flight seeds never overlap training or test seeds."""
    return [
        outbound_flight(config, treats, seed * 10_000 + 7_000 + k) for k in range(count)
    ]


def padded_inputs(flights: Sequence[Flight], steps: int) -> np.ndarray:
    """Stack flight inputs, zero-padded to ``steps`` (the RNN is causal, so padding is harmless)."""
    batch = np.zeros((len(flights), steps, flights[0].inputs.shape[1]))
    for i, flight in enumerate(flights):
        batch[i, : flight.steps] = flight.inputs
    return batch


def heading_accuracy(
    flights: Sequence[Flight], predicted: np.ndarray, config: HomingConfig
) -> float:
    """Sector accuracy on scored steps (no landmark visible) of the flights."""
    mask = scored_steps(config.heading_config())
    hits = total = 0
    for flight, row in zip(flights, predicted, strict=True):
        scored = mask[: flight.steps]
        hits += int((row[: flight.steps] == flight.sectors)[scored].sum())
        total += int(scored.sum())
    return hits / total if total else float("nan")


def summary(errors: Sequence[float]) -> dict[str, float]:
    return {
        "mean": round(statistics.fmean(errors), 3),
        "median": round(statistics.median(errors), 3),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Homing with a trained heading network."
    )
    parser.add_argument("--core", action="store_true", help="compass core only")
    parser.add_argument("--weights", default="log1p")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--variant", choices=VARIANTS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--array-index", type=int)
    parser.add_argument("--first-seed", type=int, default=0)
    parser.add_argument("--train", type=int, default=256)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--flights", type=int, default=100)
    parser.add_argument("--treats", type=Path, default=DEFAULT_TREATS)
    parser.add_argument(
        "--traces", type=int, default=0, help="store N flights for the app"
    )
    parser.add_argument("--output", type=Path, default=Path("results/homing/job.json"))
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
    task = heading_task()
    seed = arguments.seed
    train = _dataset(task, range(seed * 10_000, seed * 10_000 + arguments.train))
    masked, job_config = build_variant(arguments.variant, graph, groups, config, seed)
    model = train_model(
        masked, train, scored_steps(task).tolist(), task.sectors, job_config, seed
    )

    homing = HomingConfig()
    treats = load_treats(arguments.treats, *DEFAULT_NEST)
    flights = flights_for(homing, treats, seed, arguments.flights)
    predicted = predict(model, padded_inputs(flights, homing.max_steps))
    errors = errors_for_sectors(flights, list(predicted), homing)
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
        "training": asdict(config),
        "homing": asdict(homing),
        "nest": DEFAULT_NEST,
        "flights": len(flights),
        "found_treat": sum(f.treat >= 0 for f in flights),
        "mean_steps": round(statistics.fmean(f.steps for f in flights), 2),
        "mean_distance_m": round(
            statistics.fmean(float(np.linalg.norm(f.path[-1])) for f in flights), 2
        ),
        "heading_accuracy": round(heading_accuracy(flights, predicted, homing), 4),
        "error_m": summary(errors),
        "quantised_error_m": summary(quantised),
        "no_integration_error_m": summary(no_integration),
        "errors_m": [round(e, 2) for e in errors],
        "seconds": round(time.monotonic() - started, 1),
    }
    if arguments.traces:
        result["traces"] = [
            {
                "path": np.round(f.path, 2).tolist(),
                "treat": f.treat,
                "sectors": f.sectors.tolist(),
                "predicted": row[: f.steps].tolist(),
                "estimate": np.round(
                    home_vector(
                        sector_centres(row[: f.steps], homing.sectors), homing.step_m
                    ),
                    2,
                ).tolist(),
            }
            for f, row in zip(flights[: arguments.traces], predicted, strict=False)
        ]
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2))
    brief = {k: result[k] for k in ("heading_accuracy", "error_m", "mean_steps")}
    print(json.dumps(brief), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
