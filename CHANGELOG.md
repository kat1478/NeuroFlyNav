# Changelog

All notable changes to this project are documented here. Versions follow
[Semantic Versioning](https://semver.org/); 0.x releases are exploratory.

## [0.1.0] — 2026-10-21

First public release: the pipeline runs end to end on the real *Drosophila* central-complex wiring.
Results are exploratory; no experiment is frozen.

### Added
- Python 3.12 project skeleton with a Conda environment and a local quality gate
  (`make quality`: pytest, ruff format/lint, mypy), reproduced by GitHub Actions CI.
- Lightweight run provenance capture (`neuroflynav.provenance`).
- M0: synthetic CX-like graph, grid navigation task, RandomAgent / FlyConnectome / ShuffledFly.
- M0+: hidden-cue memory task, frozen graph reservoir with a ridge-trained linear readout,
  ShuffledFly / RandomRecurrent / Reactive controls and a robustness probe.
- M1: loader for the public hemibrain v1.2 adjacency export and CX subgraph selection
  (2,966 neurons, 467,579 edges) with `raw`, `log1p` and `binary` edge weights;
  `run_m0plus --graph hemibrain`.
- Multi-seed runner `run_m1_seeds` with 95% confidence intervals and the paired
  FlyConnectome − ShuffledFly difference.
- MIT license.

### Changed
- Large reservoirs use sparse weights, ARPACK spectral radius and vectorized controls
  (~10× faster on the CX graph); small graphs keep the original exact code paths.
- Provenance reads a tool's version from its executable when Conda ships no Python
  metadata (conda-forge ruff ≥ 0.17).

[0.1.0]: https://github.com/kat1478/NeuroFlyNav/releases/tag/v0.1.0
