"""Leaky rate RNN whose recurrent weights live only on a graph's edges (exploratory, TASK-014).

The graph is a fixed *mask*: only existing edges carry a weight, and each weight (strength
and sign) is a free parameter learned by backpropagation through time. Initial weights are
the graph's edge weights scaled to a target spectral radius, so an untrained model equals the
leaky reservoir used so far. Input weights into the sensory nodes and a linear readout are
learned too. This is camp A of the TASK-013 sign proposal: topology fixed, effective values
learned — the learned signs are not biological evidence.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray

from neuroflynav.graph.interface import Graph, NodeRole
from neuroflynav.graph.matrix import largest_eigenvalue_modulus, to_weight_matrix

FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class TrainingConfig:
    """Fixed training budget, identical for every graph that is compared."""

    steps: int = 300
    batch_size: int = 32
    learning_rate: float = 3e-3
    leak: float = 0.2
    spectral_radius: float = 1.1
    input_scale: float = 0.6
    train_recurrent: bool = True


class MaskedRateRNN(torch.nn.Module):
    """``x ← (1−α)x + α·tanh(W_mask x + B u)``; logits = readout(x) at every step."""

    sources: torch.Tensor
    targets: torch.Tensor
    sensory: torch.Tensor

    def __init__(
        self, graph: Graph, input_dim: int, num_classes: int, config: TrainingConfig
    ) -> None:
        super().__init__()
        sensory = graph.nodes_with_role(NodeRole.SENSORY)
        if not sensory:
            raise ValueError("graph needs at least one sensory node")
        dense = to_weight_matrix(graph)
        radius = largest_eigenvalue_modulus(dense)
        scale = config.spectral_radius / radius if radius > 0 else 1.0
        sources = torch.tensor([edge.source for edge in graph.edges], dtype=torch.long)
        targets = torch.tensor([edge.target for edge in graph.edges], dtype=torch.long)
        initial = torch.tensor(
            [edge.weight * scale for edge in graph.edges], dtype=torch.float32
        )
        self.num_nodes = graph.num_nodes
        self.leak = config.leak
        self.register_buffer("sources", sources)
        self.register_buffer("targets", targets)
        self.register_buffer("sensory", torch.tensor(sensory, dtype=torch.long))
        self.edge_weight = torch.nn.Parameter(
            initial, requires_grad=config.train_recurrent
        )
        self.input_weight = torch.nn.Parameter(
            torch.randn(len(sensory), input_dim) * config.input_scale
        )
        self.readout = torch.nn.Linear(graph.num_nodes, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """``inputs``: (batch, steps, input_dim) → logits (batch, steps, num_classes)."""
        batch, steps, _ = inputs.shape
        # Scatter the edge weights into a dense matrix once per pass: entries without an
        # edge stay exactly zero (the mask), and each step is then a plain matmul.
        weights = inputs.new_zeros(self.num_nodes, self.num_nodes).index_put(
            (self.targets, self.sources), self.edge_weight, accumulate=True
        )
        drive = inputs.new_zeros(batch, steps, self.num_nodes)
        drive[:, :, self.sensory] = inputs @ self.input_weight.T
        state = inputs.new_zeros(batch, self.num_nodes)
        states = []
        for step in range(steps):
            activated = torch.tanh(state @ weights.T + drive[:, step])
            state = (1 - self.leak) * state + self.leak * activated
            states.append(state)
        logits: torch.Tensor = self.readout(torch.stack(states, dim=1))
        return logits


def train_and_score(
    graph: Graph,
    train_data: tuple[FloatArray, NDArray[np.int64]],
    test_data: tuple[FloatArray, NDArray[np.int64]],
    scored: Sequence[bool],
    num_classes: int,
    config: TrainingConfig,
    seed: int,
) -> float:
    """Train on ``train_data`` with a fixed budget; return accuracy on scored test steps."""
    torch.manual_seed(seed)
    generator = np.random.default_rng(seed)
    inputs, labels = (torch.tensor(a) for a in train_data)
    inputs = inputs.float()
    mask = torch.tensor(np.asarray(scored, dtype=bool))
    model = MaskedRateRNN(graph, inputs.shape[2], num_classes, config)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=config.learning_rate
    )
    for _ in range(config.steps):
        batch = torch.tensor(
            generator.choice(len(inputs), size=config.batch_size, replace=False)
        )
        logits = model(inputs[batch])[:, mask]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, num_classes), labels[batch][:, mask].reshape(-1)
        )
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
    with torch.no_grad():
        test_inputs, test_labels = (torch.tensor(a) for a in test_data)
        predicted = model(test_inputs.float()).argmax(dim=2)[:, mask]
        return float((predicted == test_labels[:, mask]).float().mean())
