"""Load the hemibrain central-complex (CX) subgraph from the public adjacency export (spike, M1).

Source: the public ``exported-traced-adjacencies-v1.2`` export of hemibrain v1.2
(non-cropped Traced neurons only; CC BY 4.0; cite Scheffer et al. 2020). Download it to
``data/hemibrain/v1.2/`` (gitignored); see ``docs/spike_m1.md``.

Selection follows the owner-approved subgraph rule (ops D-035) as far as this export
allows: a neuron is included when it has at least one synapse in any of the six primary
CX regions, and every directed edge between included neurons with total ``weight >= 1``
is kept. Export limits: only Traced neurons, ROI membership is seen only through
synapses with other traced neurons, and ``weightHP`` is not available.

Sensory/motor roles (ring ``ER*`` neurons in, ``PFL*`` neurons out) and the weight
transforms are exploratory spike conventions, not frozen modeling choices.
"""

from __future__ import annotations

import csv
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from neuroflynav.graph.interface import Edge, Graph, NodeRole

CX_ROIS = frozenset({"AB(L)", "AB(R)", "EB", "FB", "NO", "PB"})
DEFAULT_DATA_DIR = Path("data/hemibrain/v1.2/exported-traced-adjacencies-v1.2")

WEIGHT_TRANSFORMS: dict[str, Callable[[int], float]] = {
    "raw": float,
    "log1p": lambda count: math.log1p(count),
    "binary": lambda count: 1.0,
}


@dataclass(frozen=True)
class CXSubgraph:
    """The selected CX neurons (sorted by bodyId) and raw directed synapse counts."""

    body_ids: tuple[int, ...]
    cell_types: tuple[str, ...]
    edges: tuple[tuple[int, int, int], ...]  # (source index, target index, raw weight)


def _cx_body_ids(roi_connections: Path) -> set[int]:
    members: set[int] = set()
    with roi_connections.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["roi"] in CX_ROIS and int(row["weight"]) >= 1:
                members.add(int(row["bodyId_pre"]))
                members.add(int(row["bodyId_post"]))
    return members


def load_cx_subgraph(data_dir: Path = DEFAULT_DATA_DIR) -> CXSubgraph:
    """Select the CX subgraph from the three CSV files of the adjacency export."""
    members = _cx_body_ids(data_dir / "traced-roi-connections.csv")
    types: dict[int, str] = {}
    with (data_dir / "traced-neurons.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            body_id = int(row["bodyId"])
            if body_id in members:
                types[body_id] = row["type"] or ""
    body_ids = tuple(sorted(types))
    index = {body_id: position for position, body_id in enumerate(body_ids)}
    edges: list[tuple[int, int, int]] = []
    with (data_dir / "traced-total-connections.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            source = index.get(int(row["bodyId_pre"]))
            target = index.get(int(row["bodyId_post"]))
            weight = int(row["weight"])
            if source is not None and target is not None and weight >= 1:
                edges.append((source, target, weight))
    edges.sort()
    return CXSubgraph(
        body_ids=body_ids,
        cell_types=tuple(types[body_id] for body_id in body_ids),
        edges=tuple(edges),
    )


def _role(cell_type: str, sensory: tuple[str, ...], motor: tuple[str, ...]) -> NodeRole:
    if cell_type.startswith(sensory):
        return NodeRole.SENSORY
    if cell_type.startswith(motor):
        return NodeRole.MOTOR
    return NodeRole.INTERNAL


def to_graph(
    subgraph: CXSubgraph,
    weights: str = "raw",
    sensory_prefixes: tuple[str, ...] = ("ER",),
    motor_prefixes: tuple[str, ...] = ("PFL",),
) -> Graph:
    """Turn the subgraph into a `Graph` using one candidate weight transform."""
    if weights not in WEIGHT_TRANSFORMS:
        raise ValueError(f"unknown weight transform {weights!r}")
    transform = WEIGHT_TRANSFORMS[weights]
    return Graph(
        num_nodes=len(subgraph.body_ids),
        edges=tuple(Edge(s, t, transform(w)) for s, t, w in subgraph.edges),
        roles=tuple(
            _role(cell_type, sensory_prefixes, motor_prefixes)
            for cell_type in subgraph.cell_types
        ),
    )


@dataclass(frozen=True)
class HemibrainCXGraphLoader:
    """A `GraphLoader` for the hemibrain CX subgraph with a chosen weight transform."""

    data_dir: Path = DEFAULT_DATA_DIR
    weights: str = "raw"

    def load(self) -> Graph:
        return to_graph(load_cx_subgraph(self.data_dir), self.weights)
