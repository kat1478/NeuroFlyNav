# Development spike (M0): synthetic end-to-end navigation loop

> **This is a non-authoritative development spike.** It compares three policies on a *synthetic*
> graph and a trivial grid task. The graph routing rule and topology shuffle are arbitrary M0
> computational placeholders, not biological facts or a scientific result. It proves the pipeline
> runs end to end on CPU and is reproducible for fixed seeds. Nothing here is frozen. Edge weights,
> synaptic signs, dynamics, the real connectome source, and the experiment protocol remain open.

## What M0 contains

- `neuroflynav.graph` — a source-agnostic `Graph` and `GraphLoader` protocol, plus a deterministic
  synthetic CX-like graph generator. A real connectome loader will implement the same protocol at M1.
- `neuroflynav.env.nav` — a tiny discrete-grid navigation-to-target environment with a
  Gymnasium-style `reset` / `step` API, standard library only.
- `neuroflynav.agent` — a shared `Policy.act(observation)` protocol, `RandomAgent`,
  `FlyConnectome`, and `ShuffledFly`.
- `neuroflynav.graph.shuffle` — seeded directed double-edge swaps for the ShuffledFly control.
- `neuroflynav.experiments.run_spike` — runs all three policies through the same episode runner,
  gives every policy the same episode seeds, and prints per-policy JSON summaries.

## M0 policy conventions

`FlyConnectome` consumes the synthetic `Graph`. In ascending node-index order, its first four
sensory nodes correspond to UP, DOWN, LEFT and RIGHT signals; its first four motor nodes map to
those same actions. For each direction toward the goal, it scores the shortest directed path from
that sensory node to its corresponding motor node as `1 / (1 + hops)`. It selects the highest
score; ties use the fixed action order. If no desired-direction route exists, it picks the first
desired direction. This is an intentionally simple graph-routing placeholder. It does not implement
neurons, synaptic signs, biological weights, temporal dynamics, learning or a real CX circuit.

`ShuffledFly` applies the same route rule to a graph shuffled once per run. The seeded directed
double-edge swap preserves node roles, node/edge counts, each node's in-degree and out-degree, and
the multiset of edge weights. It rejects self-loops introduced by swaps; parallel edges are allowed.
It does not preserve motifs, path lengths, role-to-role connectivity or a uniform distribution over
rewired graphs. The exact shuffle and route rules are M0 engineering choices, not a frozen
scientific control design.

## How to run

From the repository root, in the activated `neuroflynav` environment:

```bash
PYTHONPATH=src python -m neuroflynav.experiments.run_spike --episodes 20 --seed 0
```

This prints a JSON summary with the same episode seeds and one result per policy. The output below
is from that exact command/environment on this branch; it is an execution example, not an
interpreted result:

```json
{"base_seed": 0, "episode_seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19], "episodes": 20, "graph_edges": 160, "graph_nodes": 64, "policies": [{"graph": "synthetic", "mean_reward": 0.687, "mean_steps": 23.55, "policy": "RandomAgent", "success_rate": 0.75, "successes": 15}, {"graph": "synthetic", "mean_reward": 1.1845, "mean_steps": 3.05, "policy": "FlyConnectome", "success_rate": 1.0, "successes": 20}, {"graph": "shuffled(seed=0)", "mean_reward": 1.1845, "mean_steps": 3.05, "policy": "ShuffledFly", "success_rate": 1.0, "successes": 20}], "shuffle_seed": 0}
```

The example values are illustrative; run the command for actual values. Outputs are deterministic
for a fixed configuration/seed and stable implementation, but the small comparison has no
statistical interpretation. It must not be read as evidence of policy or biological superiority.

## Quality gate

```bash
make quality
```

runs the same checks as the rest of the repository (pytest, `ruff format --check`, `ruff check`,
`mypy`) over the spike code.

## Next steps (not part of this M0 slice)

1. At **M1**, resolve the exact approved source files and license conditions before any authorized
   real-data access; then build a source-dependent loader behind the same
   `GraphLoader` protocol — ideally the only module that changes.
2. At later milestones, decide and freeze scientific topology, dynamics, sign, controls and metrics
   before any formal experiment.
