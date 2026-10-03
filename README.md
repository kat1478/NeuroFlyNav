# NeuroFlyNav

NeuroFlyNav investigates whether connectivity derived from the Drosophila navigation
system can provide a useful inductive bias for artificial navigation agents. Biological
topology is a hypothesis to test; the project does not assume that it will outperform
other approaches.

## Development environment

Create the project environment with Mamba or Conda, then activate it:

```bash
mamba env create -f environment.yml
conda activate neuroflynav
```

Use the environment's Python and tools to run the local quality gate:

```bash
make quality
```

The source tree uses a `src/` layout. Pytest adds `src` to its import path through
`pyproject.toml`, so the package can be tested without installing it with pip.

## Development spike (M0)

A minimal, non-scientific end-to-end slice runs on CPU: a synthetic graph, a tiny grid
navigation task, and a comparison of RandomAgent, FlyConnectome and a degree-preserving
ShuffledFly topology control. The graph-driven policies use arbitrary M0 routing conventions;
they are not biological models. This proves the pipeline runs reproducibly, not that any policy
is superior. Nothing in the spike is frozen. Run it with:

```bash
PYTHONPATH=src python -m neuroflynav.experiments.run_spike --episodes 20 --seed 0
```

See [docs/spike.md](docs/spike.md) for the policy conventions, what the graph shuffle preserves,
and the planned next steps.

## Development spike (M0+)

M0+ makes the comparison able to *discriminate* topologies, which the plain M0 task could not. It adds
a navigation task with a *hidden goal cue* (so memory is required), turns the graph into a frozen
recurrent substrate with a trained linear readout, and compares FlyConnectome, a degree-preserving
ShuffledFly, a matched-edge RandomRecurrent, and a memoryless Reactive baseline, including a robustness
probe. It is still a non-scientific spike: the result is illustrative only and nothing is frozen. It
adds one dependency, `numpy`. Run it with:

```bash
PYTHONPATH=src python -m neuroflynav.experiments.run_m0plus --seed 0
```

See [docs/spike_m0plus.md](docs/spike_m0plus.md) for the design, how to read the output, and next steps.
