from pathlib import Path

import numpy as np
import onnx
import torch

from neuroflynav.experiments.run_trained import _dataset, heading_task
from neuroflynav.graph.interface import Edge, Graph, NodeRole
from neuroflynav.reporting.export_onnx import StepCell, export_cell, onnx_sectors
from neuroflynav.training.masked_rnn import MaskedRateRNN, TrainingConfig, predict


def _model() -> MaskedRateRNN:
    rng = np.random.default_rng(3)
    edges: set[tuple[int, int]] = set()
    while len(edges) < 60:
        s, t = (int(v) for v in rng.integers(16, size=2))
        if s != t:
            edges.add((s, t))
    roles = (NodeRole.SENSORY,) * 4 + (NodeRole.INTERNAL,) * 8 + (NodeRole.MOTOR,) * 4
    graph = Graph(
        num_nodes=16,
        edges=tuple(Edge(s, t, 1.0 + (s * t) % 4) for s, t in sorted(edges)),
        roles=roles,
    )
    torch.manual_seed(0)
    model = MaskedRateRNN(graph, 5, 8, TrainingConfig())
    with torch.no_grad():  # move away from the initial weights, as training would
        model.edge_weight.add_(torch.randn_like(model.edge_weight) * 0.3)
    return model.eval()


def test_step_cell_matches_the_full_model() -> None:
    model = _model()
    inputs, _ = _dataset(heading_task(), range(10))
    cell = StepCell(model)
    state = torch.zeros(len(inputs), model.num_nodes)
    with torch.no_grad():
        full = model(torch.tensor(inputs).float())
        for k in range(inputs.shape[1]):
            rows = [
                cell(state[b : b + 1], torch.tensor(inputs[b : b + 1, k]).float())
                for b in range(len(inputs))
            ]
            state = torch.cat([r[0] for r in rows])
            logits = torch.cat([r[1] for r in rows])
            torch.testing.assert_close(logits, full[:, k], rtol=1e-4, atol=1e-5)


def test_exported_onnx_reproduces_pytorch_sectors(tmp_path: Path) -> None:
    model = _model()
    path = tmp_path / "cell.onnx"
    export_cell(model, str(path))
    inputs, _ = _dataset(heading_task(), range(20, 26))
    np.testing.assert_array_equal(
        onnx_sectors(str(path), inputs), predict(model, inputs)
    )
    index_types = {
        t.data_type
        for t in onnx.load(str(path)).graph.initializer
        if np.prod(t.dims) == len(model.edge_weight)
        and t.data_type != onnx.TensorProto.FLOAT
    }
    assert index_types == {onnx.TensorProto.INT32}
