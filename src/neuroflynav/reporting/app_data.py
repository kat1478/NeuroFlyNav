"""Data files for the interactive web app in ``app/`` (committed in ``app/public/data/``).

Two modes:

- default: rebuild ``results.json`` and ``wiring.json`` from the committed README data
  (``docs/figures/data/``), keeping the refresh-only parts already in ``app/public/data/``;
- ``--refresh``: also recompute the refresh-only parts from local, uncommitted inputs —
  per-seed results from ``results/``, neurons per cell-type family from the hemibrain
  export, and the heading-task replay (trains two whole-CX models, ~15 min on a laptop).

Only aggregates leave the machine: family-level counts, per-seed accuracies and replay
traces of the synthetic heading task. No neuron-level connectome data is written.

Example:
    PYTHONPATH=src python -m neuroflynav.reporting.app_data --refresh
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from neuroflynav.experiments.run_trained import build_variant, heading_task
from neuroflynav.graph.interface import Graph
from neuroflynav.reporting.figures import DATA_DIR, family_of, read_family_matrix
from neuroflynav.tasks.heading import (
    HeadingTaskConfig,
    cue_visible,
    generate_sequence,
    generate_trajectory,
    scored_steps,
)
from neuroflynav.training.masked_rnn import TrainingConfig, predict, train_model

APP_DATA_DIR = Path("app/public/data")
HEMIBRAIN_DIR = Path("data/hemibrain/v1.2/exported-traced-adjacencies-v1.2")

# Per-seed results behind the forest plot rows; the grid-task row has none stored.
PER_SEED_SOURCES = (
    (
        "heading-tanh",
        "Heading task, frozen tanh reservoir",
        "heading/prereg_seeds0-19.json",
    ),
    (
        "heading-leaky",
        "Heading task, frozen leaky reservoir",
        "heading/prereg_leaky_seeds40-59.json",
    ),
    (
        "trained-core",
        "Heading task, trained weights, compass core",
        "trained/prereg_core_seeds0-19.json",
    ),
    (
        "trained-whole-cx",
        "Heading task, trained weights, whole CX",
        "trained/prereg_whole_cx_seeds40-59.json",
    ),
)
# The app uses one name for the same-size random control of both model families.
_RENAME = {"RandomMask": "RandomRecurrent"}

REPLAY_VARIANTS = ("FlyConnectome", "RandomMask")


def _accuracy(value: object) -> float:
    if isinstance(value, dict):
        return float(value["accuracy"])
    return float(value)  # type: ignore[arg-type]


def per_seed_results(results_dir: Path) -> list[dict[str, object]]:
    """Per-seed accuracies of each pre-registered experiment with stored seeds."""
    experiments: list[dict[str, object]] = []
    for key, label, relative in PER_SEED_SOURCES:
        data = json.loads((results_dir / relative).read_text())
        rows = [
            {_RENAME.get(name, name): round(_accuracy(v), 4) for name, v in row.items()}
            for row in data["per_seed"]
        ]
        experiments.append(
            {"id": key, "label": label, "seeds": data["seeds"], "accuracy": rows}
        )
    return experiments


def family_neurons(cell_types: Sequence[str], labels: Sequence[str]) -> list[int]:
    """Number of neurons in each cell-type family, in the order of ``labels``."""
    counts = Counter(family_of(cell_type) for cell_type in cell_types)
    return [counts.get(label, 0) for label in labels]


def replay(
    graph: Graph,
    groups: Sequence[str],
    task: HeadingTaskConfig,
    config: TrainingConfig,
    seed: int,
    num_train: int = 256,
    num_test: int = 64,
    num_shown: int = 6,
) -> dict[str, object]:
    """Train each replay variant and record its predictions on a few test sequences."""
    mask = scored_steps(task)
    train_seeds = range(seed * 10_000, seed * 10_000 + num_train)
    test_seeds = range(seed * 10_000 + 5_000, seed * 10_000 + 5_000 + num_test)
    train_pairs = [generate_sequence(task, s) for s in train_seeds]
    train = (
        np.stack([p[0] for p in train_pairs]),
        np.stack([p[1] for p in train_pairs]),
    )
    trajectories = [generate_trajectory(task, s) for s in test_seeds]
    test_inputs = np.stack([t[0] for t in trajectories])
    test_labels = np.stack([t[1] for t in trajectories])
    variants: dict[str, dict[str, object]] = {}
    for name in REPLAY_VARIANTS:
        masked, job_config = build_variant(name, graph, groups, config, seed)
        model = train_model(
            masked, train, mask.tolist(), task.sectors, job_config, seed
        )
        predicted = predict(model, test_inputs)
        hits = predicted[:, mask] == test_labels[:, mask]
        variants[name] = {
            "accuracy": round(float(hits.mean()), 4),
            "predicted": predicted[:num_shown].tolist(),
        }
    sector_width = 2 * math.pi / task.sectors
    return {
        "seed": seed,
        "nodes": graph.num_nodes,
        "edges": len(graph.edges),
        "sectors": task.sectors,
        "sector_width": round(sector_width, 6),
        "steps": task.steps,
        "landmark": cue_visible(task).tolist(),
        "scored": mask.tolist(),
        "sequences": [
            {
                "theta": [round(float(v) % (2 * math.pi), 4) for v in t[2]],
                "sector": t[1].tolist(),
            }
            for t in trajectories[:num_shown]
        ],
        "variants": variants,
    }


def build(
    summary: dict[str, object],
    matrix_path: Path,
    previous: dict[str, dict[str, object]],
    refreshed: dict[str, object],
) -> dict[str, dict[str, object]]:
    """Assemble the app files from committed data and refresh-only parts."""
    labels, matrix, header = read_family_matrix(matrix_path)
    fields = dict(part.split("=") for part in header.lstrip("# ").split())
    old_results = previous.get("results", {})
    old_wiring = previous.get("wiring", {})
    results = {
        "_source": summary["_source"],
        "forest": summary["forest"],
        "progression": summary["progression"],
        "learning": summary["learning"],
        "references": summary["references"],
        "per_seed": refreshed.get("per_seed", old_results.get("per_seed", [])),
    }
    wiring = {
        "_source": "Synapses between cell-type families in the hemibrain v1.2 CX "
        "subgraph (rows: source, columns: target); aggregated, CC BY 4.0.",
        "labels": labels,
        "matrix": [[int(v) for v in row] for row in matrix],
        "neurons": refreshed.get("neurons", old_wiring.get("neurons", [])),
        "total_neurons": int(fields["neurons"]),
        "total_synapses": int(fields["synapses"]),
    }
    files = {"results": results, "wiring": wiring}
    replay_data = refreshed.get("replay", previous.get("replay"))
    if replay_data is not None:
        files["replay"] = replay_data  # type: ignore[assignment]
    return files


def _read(path: Path) -> dict[str, object] | None:
    return json.loads(path.read_text()) if path.exists() else None


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the web app data files.")
    parser.add_argument("--out", type=Path, default=APP_DATA_DIR)
    parser.add_argument("--refresh", action="store_true", help="recompute local parts")
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--data-dir", type=Path, default=HEMIBRAIN_DIR)
    parser.add_argument(
        "--replay-core",
        action="store_true",
        help="replay on the compass core (~1 min) instead of the whole CX (~15 min)",
    )
    arguments = parser.parse_args(argv)

    previous = {
        name: data
        for name in ("results", "wiring", "replay")
        if (data := _read(arguments.out / f"{name}.json")) is not None
    }
    refreshed: dict[str, object] = {}
    if arguments.refresh:
        from neuroflynav.graph.hemibrain import (
            COMPASS_CORE_PREFIXES,
            load_cx_subgraph,
            node_groups,
            restrict,
            to_graph,
        )

        labels, _, _ = read_family_matrix(DATA_DIR / "cx_family_matrix.csv")
        subgraph = load_cx_subgraph(arguments.data_dir)
        refreshed["per_seed"] = per_seed_results(arguments.results_dir)
        refreshed["neurons"] = family_neurons(subgraph.cell_types, labels)
        if arguments.replay_core:
            subgraph = restrict(subgraph, COMPASS_CORE_PREFIXES)
        refreshed["replay"] = {
            "graph": "compass core" if arguments.replay_core else "whole CX",
            **replay(
                to_graph(subgraph, "log1p"),
                node_groups(subgraph),
                heading_task(),
                TrainingConfig(),
                seed=100,
            ),
        }

    summary = json.loads((DATA_DIR / "summary.json").read_text())
    files = build(summary, DATA_DIR / "cx_family_matrix.csv", previous, refreshed)
    arguments.out.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        (arguments.out / f"{name}.json").write_text(text + "\n")
        print(f"{arguments.out / name}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
