# NeuroFlyNav

[![Quality](https://github.com/kat1478/NeuroFlyNav/actions/workflows/quality.yml/badge.svg?branch=develop)](https://github.com/kat1478/NeuroFlyNav/actions/workflows/quality.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**Does the wiring of a fly's navigation centre make a better brain for an artificial agent?**

NeuroFlyNav takes the real connectome of the *Drosophila* central complex (CX) — the brain
region flies use to keep track of heading and steer — and uses it as the fixed recurrent core
of a navigation agent. It then compares that agent against fair controls: the same wiring
shuffled, a random network of the same size, and a network with no memory at all. Biological
topology is a hypothesis to test, not an assumption; a negative result is a valid outcome.

> **Status: v0.1.0, exploratory.** The pipeline runs end to end on the real CX wiring (2,966
> neurons, 467,579 connections). Dynamics, inputs and readout are placeholders, and no
> experiment has been frozen yet. Numbers below are a first look, not a scientific claim.

## How it works

```
hemibrain v1.2 connectome ──► CX subgraph ──► frozen recurrent "reservoir" ──► linear readout ──► action
 (public export, CC BY)       6 CX regions     W scaled to spectral radius 1.1    trained by
                              weight ≥ 1        state = tanh(W·state + input)      imitation (ridge)
```

1. **Graph** — every neuron with at least one synapse in the six CX regions (AB, EB, FB, NO,
   PB), and every directed connection between them (`neuroflynav.graph.hemibrain`).
2. **Reservoir** — the wiring becomes a fixed weight matrix; observations enter at the ring
   neurons (`ER*`), the state evolves through the real connections (`neuroflynav.agent.reservoir`).
3. **Task** — a grid world where the goal direction is shown only for the first few steps, so
   the agent must *remember* it (`neuroflynav.env.nav_memory`).
4. **Training** — only the readout is learned, by imitating an expert (`neuroflynav.training`).
5. **Controls** — `ShuffledFly` (degree-preserving rewiring), `RandomRecurrent` (matched edge
   count), `Reactive` (no memory).

## First look (20 seeds)

Success rate on the hidden-cue task, mean and 95% CI, `log1p` edge weights:

| Agent | Success rate |
| --- | --- |
| FlyConnectome (real CX wiring) | 0.64 [0.59, 0.68] |
| ShuffledFly (same degrees, rewired) | 0.56 [0.53, 0.60] |
| RandomRecurrent (same size) | 0.57 [0.53, 0.61] |
| Reactive (no memory) | 0.31 [0.30, 0.32] |

Paired difference FlyConnectome − ShuffledFly: **+0.08 [+0.02, +0.13]** — a small edge for the
real wiring that also appears with wiring-only (`binary`) weights. It is exploratory: three
weight transforms were compared without correction and the dynamics are placeholders. A
robustness gap seen in a single seed disappeared over 20 seeds. Full tables:
[docs/spike_m1.md](docs/spike_m1.md).

## Quick start

```bash
mamba env create -f environment.yml
conda activate neuroflynav
make quality                                   # pytest, ruff format/lint, mypy

# synthetic graph, no download needed
PYTHONPATH=src python -m neuroflynav.experiments.run_m0plus --seed 0

# real CX wiring: download the public export first (46 MB, see docs/spike_m1.md)
PYTHONPATH=src python -m neuroflynav.experiments.run_m0plus --graph hemibrain --weights log1p
PYTHONPATH=src python -m neuroflynav.experiments.run_m1_seeds --seeds 20
```

Dependencies are managed with Conda only (`environment.yml`, conda-forge). The `src/` layout is
importable without installation; one hemibrain seed takes about 30 s on a laptop CPU.

## Documentation

| Doc | What it covers |
| --- | --- |
| [docs/spike.md](docs/spike.md) | M0: first end-to-end loop on a synthetic graph |
| [docs/spike_m0plus.md](docs/spike_m0plus.md) | M0+: hidden-cue task and reservoir, so topologies can be told apart |
| [docs/spike_m1.md](docs/spike_m1.md) | M1: real hemibrain CX wiring, data download, multi-seed results |
| [docs/reproducibility.md](docs/reproducibility.md) | Provenance capture for runs |
| [CHANGELOG.md](CHANGELOG.md) | Release history |

## Roadmap

- **M2 — honest science:** cell-type-preserving shuffle (does the edge come from exact partners
  or from cell-type structure?), a pre-registered comparison, frozen protocol and holdout.
- **M3 — cloud:** containerised multi-seed sweeps on AWS Batch (CPU) with S3 artifacts.
- **M4 — write-up:** results notebook, figures, short report.

## Data and citation

Connectome data: hemibrain v1.2, Janelia FlyEM, CC BY 4.0 — Scheffer et al. (2020), *eLife*
9:e57443, https://doi.org/10.7554/eLife.57443. Central-complex context: Hulse et al. (2021),
*eLife* 10:e66039. The data is downloaded locally and never committed.

## License

Code: [MIT](LICENSE) © 2026 Katarzyna Kadyszewska. Connectome data keeps its own license (CC BY 4.0).
