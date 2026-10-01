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
