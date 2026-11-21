# Changelog

All notable changes to this project are documented here. Versions follow
[Semantic Versioning](https://semver.org/); 0.x releases are exploratory.

## [Unreleased]

## [0.9.0] — 2026-11-21

The network computes the home vector itself: pre-registered, the real central-complex wiring
computes a better home vector than a random wiring of the same size. The job image is now built by
GitHub Actions.

### Added
- Home-vector task (`tasks/homevector.py`, `run_homevector`, `aggregate_homevector`): the network
  reports where the nest is at every step (regression readout). Pre-registered result on seeds 80–99
  (100 AWS Batch jobs): the real wiring computes a better home vector than a same-size random wiring,
  −14.7 m [−18.8, −10.7]; vs within-cell-type shuffle −2.0 m [−5.0, +0.9] (`docs/spike_homevector.md`).
- `.github/workflows/image.yml`: GitHub Actions builds the job image and publishes it to
  `ghcr.io/kat1478/neuroflynav:<sha12>` with the built-in token (no AWS access, no stored keys).
- Terraform `image_repository` variable: the job definition can use the GHCR image
  (default unchanged: the project ECR repository).

### Changed
- App, City view: new card with the pre-registered home-vector result (verdict, chart of real minus
  each control with 95% CI, table, caveats); `app_data --homevector` exports it into `city.json`.
- `docs/report.md` updated for v0.9.0: home-vector result (task 4, row 9, reading, limitations).
- `docs/report.md` updated for v0.8.0: the trained networks in the browser (ONNX export, parity,
  ONNX Runtime Web), test counts, next steps.

### Fixed
- `make deploy-aws` uploads the ONNX Runtime `.wasm` and the `.onnx` models pre-gzipped
  (CloudFront does not compress files over 10 MB or `application/octet-stream`): the first fly on
  the AWS mirror downloads ~9.8 MB instead of ~26 MB.

## [0.8.0] — 2026-11-18

Drop a fly anywhere: the trained whole-CX networks run live in the browser. No new experiments,
no changed results.

### Added
- `reporting/export_onnx.py`: exports a trained masked RNN as a one-step ONNX cell with sparse
  recurrence (Gather → Mul → ScatterElements add, int32 indices, opset 18; 5.8 MB per model);
  CLI retrains the seed-100 City networks and writes models, `golden.json` and a manifest with
  SHA-256, parity with PyTorch (agreement 1.0) and accuracy (0.8253 real, 0.8277 random).
  `onnx` and `onnxruntime` from conda-forge.
- Models published as the release asset `models-v1` (never in Git); `app/scripts/fetch-models.sh`
  downloads and verifies them for Pages, `make deploy-aws` and CI.
- App, City view: **Drop a fly anywhere** — place a nest inside the mapped area (or at a random
  spot) and let the fly search or pick a treat; both networks run in the browser with ONNX
  Runtime Web 1.31.1 (WebAssembly, lazy-loaded), progress shown, seed per fly.
- `app/src/fly/`: TypeScript flight generator, network runner and home-vector maths; Vitest unit
  tests and a golden test against PyTorch's outputs.

### Fixed
- CHANGELOG: missing 0.7.0 heading.

## [0.7.1] — 2026-11-18

Documentation and presentation release: no new experiments, no changed results.

### Added
- README for first-time visitors: live-app link with an animated demo of the City view, a
  "What this project demonstrates" table linking each skill to code, and two Mermaid architecture
  diagrams (science pipeline, delivery pipeline); "How it works" covers all models, tasks and controls.
- `docs/preregistration.md`: the design and decision rule of all eight pre-registered comparisons,
  with the public commit each run used.
- `docs/report.md`: a short technical report (PDF attached to this release).

### Fixed
- Public code and docs no longer refer to private planning notes.
- CHANGELOG: missing 0.6.0 heading and link references.
- `geo/environment.yml` now includes `osmium-tool`, required by the map build.
- Docs: infrastructure README covers the web-app mirror; actual cost from AWS Cost Explorer
  (0.63 USD for the day of the whole-CX sweep, pilot included) instead of an estimate; stale
  "next steps" marked done; six CX regions named correctly.
## [0.7.0] — 2026-11-17

Fly in the city: a simulated fly forages over central Warsaw (OpenStreetMap) and flies home by
path integration, using the heading estimated by the trained central-complex network. A
pre-registered test finds **no homing advantage** of the real wiring over a random one.

### Added
- Map data pipeline `geo/build_map.py` (env `neuroflynav-geo`: osmnx, geopandas, tippecanoe,
  osmium-tool): dated Geofabrik extract → central Warsaw → PMTiles (1.5 MB) and 326 treats;
  manifest with SHA-256; ODbL attribution.
- Homing task `tasks/homing.py`, experiment `run_homing` (per variant × seed, AWS Batch via
  `MODULE=run_homing`), `aggregate_homing`; references for quantisation only and no integration.
- App: **City** view (MapLibre + self-hosted PMTiles, lazily loaded), `city.json` export
  (`app_data --city`), OSM credit in the footer.
- Docs: `docs/spike_homing.md`; README paragraph; learning-friendly app docs.

### Results (pre-registered, whole CX, seeds 60–79, 100 flights per seed)
- Mean homing error: real wiring 27.2 m, random 28.4 m, degree-shuffled 29.2 m, type-shuffled
  27.1 m, frozen recurrent weights 166 m; references: true sectors 12.1 m, no integration 32.0 m.
- Fly − RandomMask **−1.2 m [−4.1, +1.7]** — interval includes zero: no homing advantage.

### Checks
- Independent science review (re-aggregated 100 jobs) and UI/UX audit of the City view; all
  findings fixed.

## [0.6.0] — 2026-11-16

Interactive results explorer (start of M4): a static web app with the pre-registered results,
the central-complex wiring and a replay of the heading task, published on GitHub Pages and
mirrored on AWS (S3 + CloudFront). No new experiments.

### Added
- Web app in `app/` (Vite, TypeScript, Observable Plot; Node from conda-forge, npm dependencies
  pinned by lockfile). Three views:
  - **Results:** headline result, forest plot of all pre-registered comparisons, per-seed
    differences, learning curve; hover tips and a table view for every chart.
  - **Wiring:** synapses between the 19 cell-type families of the hemibrain CX, family cards,
    and a glossary of abbreviations and brain regions.
  - **Replay:** two trained whole-CX networks (real vs random wiring) tracking heading step by
    step, with an honest note that the edge appears only over 20 seeds.
- Data export `neuroflynav.reporting.app_data` (family-level aggregates and synthetic-task
  traces only); `masked_rnn.train_model` / `predict`, `heading.generate_trajectory`.
- Deployment: GitHub Pages workflow (on release); Terraform `infra/site.tf` (private S3 bucket,
  CloudFront with Origin Access Control, HTTPS only, managed security headers);
  `make deploy-aws`. Docs: `docs/app.md`.
- CI: `App` job (lint, format, typecheck, tests, build).

### Checks
- Independent UI/UX audit (20 findings, all fixed); light/dark and narrow layouts checked.
- Stored results reproduce exactly after the training refactor (`train_and_score` unchanged).

## [0.5.0] — 2026-11-16

Cloud (M3): the trained-mask experiment runs as containerised AWS Batch sweeps defined in
Terraform. The first cloud run — the whole CX, too slow for a laptop — gives the first result
where the real wiring beats every control.

### Added
- Per-job CLI: `run_trained --variant --seed` or `--array-index` runs one (variant, seed);
  `aggregate_trained` merges per-job JSON files into the usual summary (identical output).
- Container (`Dockerfile`, `docker/entrypoint.sh`): digest-pinned micromamba base, same conda
  environment, non-root; downloads the hemibrain export and checks its SHA-256; uploads the
  result to S3. Docs: `docs/cloud.md`.
- Terraform stack (`infra/`): ECR, private encrypted S3 bucket, least-privilege IAM roles,
  AWS Batch on Fargate Spot, CloudWatch logs, cost budget with alerts.
- Low-data regime for the trained compass core (16 / 32 / 64 training sequences), pre-registered:
  the real mask trains slightly worse than a random mask of the same density (pooled −0.024).
- README: whole-CX result in the forest plot, "Cloud runs" section.

### Results (pre-registered, log1p)
- Whole CX (2,966 neurons, 5.3% density), trained masks, seeds 40–59, 100 AWS Batch jobs:
  FlyConnectome 0.845; Fly − RandomMask **+0.028 [+0.021, +0.036]**, Fly − ShuffledFly +0.037,
  Fly − TypeShuffledFly +0.011. Unlike the dense compass core, the real wiring trains better.

### Fixed
- Training batch is capped at the number of training sequences.

## [0.4.0] — 2026-10-25

Learning dynamics: leaky (slower) dynamics make the network integrate self-motion, and training
the recurrent weights on the compass-core mask makes it a strong integrator. In both cases the
real CX wiring is no better than shuffled or random wiring of the same density.

### Added
- Leaky rate update for the reservoir (`ReservoirConfig.leak`, `run_heading --leak`); default
  unchanged. Pre-registered heading result: integration beats a memory-only oracle, for the CX
  graph and for every control alike; signs no longer matter.
- Masked rate RNN with recurrent weights trained on a graph mask (`training.masked_rnn`,
  `run_trained`), compass-core restriction (`graph.hemibrain.restrict`); PyTorch (CPU) added.
  Pre-registered result in `docs/spike_trained.md`: training helps, the real mask does not beat
  a random mask of the same density.

### Fixed
- `.gitignore` ignores `/data` also when it is a symlink (worktrees).

## [0.3.0] — 2026-10-24

A harder task and a first look at synaptic signs. Both results point to the reservoir dynamics,
not the wiring or the signs, as the current limit.

### Added
- Heading integration task (`neuroflynav.tasks.heading`, `run_heading`): landmarks every 8 steps,
  self-motion in between, hold-last-landmark oracle as reference; pre-registered result in
  `docs/spike_heading.md`.
- Reservoir builders accept `input_dim` (default unchanged).
- Exploratory cell-type sign rule (Δ7 and ER outputs inhibitory), `run_heading --signs types`
  with RandomSignFly and UnsignedFly controls; pre-registered negative result in
  `docs/spike_heading.md`.

### Fixed
- Reservoir docstrings now say the readout uses the full reservoir state (as implemented),
  not only the motor nodes.

## [0.2.0] — 2026-10-21

First answer to H2 (exact wiring vs cell-type structure): on the hidden-cue task, the edge of the real CX wiring over a
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

[Unreleased]: https://github.com/kat1478/NeuroFlyNav/compare/v0.9.0...develop
[0.9.0]: https://github.com/kat1478/NeuroFlyNav/releases/tag/v0.9.0
[0.8.0]: https://github.com/kat1478/NeuroFlyNav/releases/tag/v0.8.0
[0.7.1]: https://github.com/kat1478/NeuroFlyNav/releases/tag/v0.7.1
[0.7.0]: https://github.com/kat1478/NeuroFlyNav/releases/tag/v0.7.0
[0.6.0]: https://github.com/kat1478/NeuroFlyNav/releases/tag/v0.6.0
[0.5.0]: https://github.com/kat1478/NeuroFlyNav/releases/tag/v0.5.0
[0.4.0]: https://github.com/kat1478/NeuroFlyNav/releases/tag/v0.4.0
[0.3.0]: https://github.com/kat1478/NeuroFlyNav/releases/tag/v0.3.0
[0.2.0]: https://github.com/kat1478/NeuroFlyNav/releases/tag/v0.2.0
[0.1.0]: https://github.com/kat1478/NeuroFlyNav/releases/tag/v0.1.0
