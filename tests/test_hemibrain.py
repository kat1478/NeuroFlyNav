import math
from pathlib import Path

import pytest

from neuroflynav.graph.hemibrain import load_cx_subgraph, to_graph
from neuroflynav.graph.interface import NodeRole


def _write_export(directory: Path) -> None:
    (directory / "traced-neurons.csv").write_text(
        "bodyId,type,instance\n30,PFL3,a\n10,ER4d,b\n20,EPG,c\n40,MBON01,d\n"
    )
    # 40 has synapses only outside the CX, so it is excluded.
    (directory / "traced-roi-connections.csv").write_text(
        "bodyId_pre,bodyId_post,roi,weight\n10,20,EB,3\n20,30,PB,2\n30,40,LAL(R),5\n"
    )
    (directory / "traced-total-connections.csv").write_text(
        "bodyId_pre,bodyId_post,weight\n10,20,4\n20,30,2\n30,40,5\n20,10,1\n"
    )


def test_selects_cx_neurons_and_edges(tmp_path: Path) -> None:
    _write_export(tmp_path)
    subgraph = load_cx_subgraph(tmp_path)

    assert subgraph.body_ids == (10, 20, 30)
    assert subgraph.edges == ((0, 1, 4), (1, 0, 1), (1, 2, 2))

    graph = to_graph(subgraph, weights="log1p")
    assert graph.roles == (NodeRole.SENSORY, NodeRole.INTERNAL, NodeRole.MOTOR)
    assert graph.edges[0].weight == pytest.approx(math.log1p(4))
    assert {edge.weight for edge in to_graph(subgraph, "binary").edges} == {1.0}


def test_rejects_unknown_transform(tmp_path: Path) -> None:
    _write_export(tmp_path)
    with pytest.raises(ValueError):
        to_graph(load_cx_subgraph(tmp_path), weights="sqrt")
