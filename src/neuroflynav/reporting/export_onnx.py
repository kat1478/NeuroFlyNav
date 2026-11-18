"""Export a trained masked RNN as a one-step ONNX cell for the browser (exploratory).

The cell maps ``(state, input) -> (next_state, logits)`` for one time step and keeps the
recurrent weights sparse: the state is gathered at every edge's source, multiplied by the
edge weight and scatter-added at the edge's target (ONNX ``ScatterElements`` with
``reduction="add"``), so a 2,966-neuron graph needs ~467k weights instead of a dense
2,966 × 2,966 matrix. The browser runs the time loop and feeds the state back in.
"""

from __future__ import annotations

import numpy as np
import onnx
import torch
from onnx import numpy_helper

from neuroflynav.training.masked_rnn import MaskedRateRNN


class StepCell(torch.nn.Module):
    """One time step of a :class:`MaskedRateRNN`, batch size 1, sparse recurrence."""

    sources: torch.Tensor
    targets: torch.Tensor
    edge_weight: torch.Tensor
    drive: torch.Tensor
    readout_weight: torch.Tensor
    readout_bias: torch.Tensor

    def __init__(self, model: MaskedRateRNN) -> None:
        super().__init__()
        n = model.num_nodes
        input_dim = model.input_weight.shape[1]
        drive = torch.zeros(n, input_dim)
        drive[model.sensory] = model.input_weight.detach()
        self.leak = float(model.leak)
        self.register_buffer("sources", model.sources.clone())
        self.register_buffer("targets", model.targets.clone().unsqueeze(0))
        self.register_buffer("edge_weight", model.edge_weight.detach().clone())
        self.register_buffer("drive", drive)
        self.register_buffer("readout_weight", model.readout.weight.detach().clone())
        self.register_buffer("readout_bias", model.readout.bias.detach().clone())

    def forward(
        self, state: torch.Tensor, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``state`` (1, n), ``inputs`` (1, input_dim) -> next state (1, n), logits (1, k)."""
        messages = state[:, self.sources] * self.edge_weight
        recurrent = torch.zeros_like(state).scatter_add(1, self.targets, messages)
        activated = torch.tanh(recurrent + inputs @ self.drive.T)
        next_state = (1 - self.leak) * state + self.leak * activated
        logits = next_state @ self.readout_weight.T + self.readout_bias
        return next_state, logits


def export_cell(model: MaskedRateRNN, path: str, opset: int = 18) -> None:
    """Write the one-step cell of ``model`` to ``path`` as ONNX."""
    cell = StepCell(model).eval()
    n = model.num_nodes
    input_dim = model.input_weight.shape[1]
    torch.onnx.export(
        cell,
        (torch.zeros(1, n), torch.zeros(1, input_dim)),
        path,
        input_names=["state", "input"],
        output_names=["next_state", "logits"],
        opset_version=opset,
        dynamo=False,
    )
    narrow_index_initializers(path, min_size=len(model.edge_weight))


def narrow_index_initializers(path: str, min_size: int) -> None:
    """Store the large edge-index tensors as int32 (Gather/ScatterElements accept int32).

    The node count is far below 2**31, so the values are unchanged; the file shrinks by ~40 %.
    """
    model = onnx.load(path)
    for tensor in model.graph.initializer:
        if (
            tensor.data_type == onnx.TensorProto.INT64
            and np.prod(tensor.dims) >= min_size
        ):
            values = numpy_helper.to_array(tensor)
            if values.max() >= 2**31:
                raise ValueError(f"{tensor.name} does not fit int32")
            tensor.CopyFrom(
                numpy_helper.from_array(values.astype(np.int32), tensor.name)
            )
    onnx.checker.check_model(model)
    onnx.save(model, path)


# ---------------------------------------------------------------------------
# Command line: train the seed-100 networks shown in the City view and export them.
# ---------------------------------------------------------------------------

MODELS = {"fly": "FlyConnectome", "random": "RandomMask"}


def onnx_sectors(path: str, inputs: np.ndarray) -> np.ndarray:
    """Predicted sector at every step, running the exported cell step by step."""
    import onnxruntime as ort

    session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    nodes = session.get_inputs()[0].shape[1]
    sectors = np.zeros(inputs.shape[:2], dtype=np.int64)
    for b in range(inputs.shape[0]):
        state = np.zeros((1, nodes), dtype=np.float32)
        for k in range(inputs.shape[1]):
            state, logits = session.run(
                None, {"state": state, "input": inputs[b : b + 1, k].astype(np.float32)}
            )
            sectors[b, k] = int(np.argmax(logits))
    return sectors


def main(argv: list[str] | None = None) -> int:
    import argparse
    import hashlib
    import json
    from pathlib import Path

    from neuroflynav.experiments.run_trained import (
        _dataset,
        build_variant,
        heading_task,
    )
    from neuroflynav.graph.hemibrain import (
        DEFAULT_DATA_DIR,
        load_cx_subgraph,
        node_groups,
        to_graph,
    )
    from neuroflynav.tasks.heading import scored_steps
    from neuroflynav.training.masked_rnn import TrainingConfig, predict, train_model

    parser = argparse.ArgumentParser(
        description="Export the City-view networks to ONNX."
    )
    parser.add_argument("--out", type=Path, default=Path("results/models"))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument(
        "--golden", type=int, default=3, help="test sequences in golden.json"
    )
    arguments = parser.parse_args(argv)

    seed = arguments.seed
    subgraph = load_cx_subgraph(arguments.data_dir)
    graph = to_graph(subgraph, "log1p")
    groups = node_groups(subgraph)
    task = heading_task()
    mask = scored_steps(task)
    config = TrainingConfig()
    train = _dataset(task, range(seed * 10_000, seed * 10_000 + 256))
    test_inputs, test_labels = _dataset(
        task, range(seed * 10_000 + 5_000, seed * 10_000 + 5_000 + 64)
    )
    arguments.out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, object] = {
        "seed": seed,
        "graph": "hemibrain v1.2 CX subgraph (whole CX), log1p weights",
        "nodes": graph.num_nodes,
        "edges": len(graph.edges),
        "license": "Derived from hemibrain v1.2 (Janelia FlyEM, Scheffer et al. 2020), CC BY 4.0",
        "inputs": "state (1, nodes), input (1, 5); outputs next_state, logits (1, 8)",
        "models": {},
    }
    golden: dict[str, object] = {
        "inputs": test_inputs[: arguments.golden].astype(np.float32).tolist(),
        "sectors": {},
    }
    for name, variant in MODELS.items():
        masked, job_config = build_variant(variant, graph, groups, config, seed)
        model = train_model(
            masked, train, mask.tolist(), task.sectors, job_config, seed
        ).eval()
        torch.save(model.state_dict(), arguments.out / f"{name}.pt")
        path = arguments.out / f"{name}.onnx"
        export_cell(model, str(path))
        reference = predict(model, test_inputs)
        exported = onnx_sectors(str(path), test_inputs)
        agreement = float((reference == exported).mean())
        hits = exported[:, mask] == test_labels[:, mask]
        accuracy = float(torch.tensor(hits).float().mean())
        golden["sectors"][name] = reference[: arguments.golden].tolist()  # type: ignore[index]
        manifest["models"][name] = {  # type: ignore[index]
            "variant": variant,
            "file": path.name,
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "agreement_with_pytorch": agreement,
            "accuracy_onnx": round(accuracy, 4),
        }
        print(f"{name}: agreement {agreement:.4f}, accuracy {accuracy:.4f}")
    (arguments.out / "golden.json").write_text(json.dumps(golden))
    (arguments.out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
