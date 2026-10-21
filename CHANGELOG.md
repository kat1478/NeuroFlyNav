# Changelog

All notable changes to this project are documented here. Versions follow
[Semantic Versioning](https://semver.org/); 0.x releases are exploratory.

## [Unreleased]

### Added
- Heading integration task (`neuroflynav.tasks.heading`, `run_heading`): landmarks every 8 steps,
  self-motion in between, hold-last-landmark oracle as reference; pre-registered result in
  `docs/spike_heading.md`.
- Reservoir builders accept `input_dim` (default unchanged).
- Exploratory cell-type sign rule (Δ7 and ER outputs inhibitory), `run_heading --signs types`
  with RandomSignFly and UnsignedFly controls; pre-registered negative result in
  `docs/spike_heading.md`.

## [0.2.0] — 2026-10-21

First answer to H2: on the hidden-cue task, the edge of the real CX wiring over a
degree-preserving shuffle replicates on fresh seeds and is explained by cell-type structure.

### Added
- `TypeShuffledFly` control: degree-preserving rewiring restricted to the same cell type (H2).
- `run_m1_seeds --first-seed --type-shuffle`; results report FlyConnectome minus every control.

### Changed
- README and `docs/spike_m1.md`: pre-registered replication on seeds 20–39. The edge over
  ShuffledFly replicates with `log1p` weights and is explained by cell-type structure; the
  earlier claim that it also appears with `binary` weights did not replicate and is corrected.

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

[0.2.0]: https://github.com/kat1478/NeuroFlyNav/releases/tag/v0.2.0
[0.1.0]: https://github.com/kat1478/NeuroFlyNav/releases/tag/v0.1.0
