import numpy as np
import torch

from neuroflynav.graph.interface import Edge, Graph, NodeRole
from neuroflynav.training.masked_rnn import MaskedRateRNN, TrainingConfig


def _graph() -> Graph:
    edges = (Edge(0, 1, 1.0), Edge(1, 2, 2.0), Edge(2, 0, 1.0), Edge(2, 1, 0.5))
    roles = (NodeRole.SENSORY, NodeRole.INTERNAL, NodeRole.MOTOR)
    return Graph(num_nodes=3, edges=edges, roles=roles)


def test_forward_shapes_and_gradient_only_on_existing_edges() -> None:
    torch.manual_seed(0)
    model = MaskedRateRNN(_graph(), input_dim=5, num_classes=8, config=TrainingConfig())
    logits = model(torch.randn(2, 6, 5))
    assert logits.shape == (2, 6, 8)
    logits.sum().backward()
    assert model.edge_weight.grad is not None
    assert model.edge_weight.shape == (4,)  # one parameter per graph edge, nothing else


def test_initial_weights_are_scaled_to_spectral_radius() -> None:
    model = MaskedRateRNN(_graph(), 5, 8, TrainingConfig(spectral_radius=0.9))
    dense = np.zeros((3, 3))
    for s, t, w in zip(
        model.sources.tolist(), model.targets.tolist(), model.edge_weight.tolist()
    ):
        dense[t, s] += w
    assert abs(np.max(np.abs(np.linalg.eigvals(dense))) - 0.9) < 1e-5


def test_frozen_recurrent_weights_have_no_gradient() -> None:
    config = TrainingConfig(train_recurrent=False)
    model = MaskedRateRNN(_graph(), 5, 8, config)
    model(torch.randn(1, 3, 5)).sum().backward()
    assert model.edge_weight.grad is None
