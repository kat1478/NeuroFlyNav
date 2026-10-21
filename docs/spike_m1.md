# Development spike (M1): the real CX wiring in the M0+ pipeline

> **Exploratory spike.** One seed, 40 evaluation episodes, placeholder dynamics and readout.
> The numbers below show that the pipeline runs on real connectome wiring; they are not evidence
> that any topology is better.

## Data

Public hemibrain v1.2 adjacency export (Janelia FlyEM, CC BY 4.0). It covers all non-cropped
Traced neurons. Cite: Scheffer et al. (2020), *eLife* 9:e57443, https://doi.org/10.7554/eLife.57443;
for central-complex context, Hulse et al. (2021), *eLife* 10:e66039.

```bash
mkdir -p data/hemibrain/v1.2 && cd data/hemibrain/v1.2
curl -LO https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz
sha256sum exported-traced-adjacencies-v1.2.tar.gz
# 07d8946eb0c4e3a5cb23d5769c9817847494f9fcadbc0ca239eed7bbd5555cf7 (downloaded 2026-10-21, 45,872,577 bytes)
tar xzf exported-traced-adjacencies-v1.2.tar.gz
```

`data/` is gitignored; the dataset is never committed.

## Subgraph

`neuroflynav.graph.hemibrain` selects every neuron with at least one synapse in the six primary
CX regions — `AB(L)`, `AB(R)`, `EB`, `FB`, `NO`, `PB` — and keeps every directed edge between
selected neurons with total synapse count `weight >= 1`. Result: **2,966 neurons, 467,579 edges,
2.72 M synapses**.

Limits of this export: only Traced neurons; ROI membership is seen only through synapses between
traced neurons; no `weightHP` field.

Spike conventions (not modeling decisions):
- inputs are the ring neurons (`ER*`, 257 cells), outputs the `PFL*` neurons (50 cells);
- edge-weight transform `raw` (synapse count), `log1p`, or `binary` (wiring only).

## Run

```bash
PYTHONPATH=src python -m neuroflynav.experiments.run_m0plus --graph hemibrain --weights log1p --seed 0
```

One run takes about 5 minutes on a laptop CPU: the 2,966 × 2,966 reservoir matrix is dense.

## First look (seed 0, 80 training / 40 evaluation episodes)

Success rate on the hidden-cue task / on the harder robustness probe (1 cue step):

| Graph | FlyConnectome | ShuffledFly | RandomRecurrent | Reactive |
| --- | --- | --- | --- | --- |
| synthetic (64 nodes) | 0.55 / 0.30 | 0.53 / 0.20 | 0.43 / 0.13 | 0.28 / 0.13 |
| hemibrain CX, raw | 0.68 / 0.30 | 0.65 / 0.25 | 0.50 / 0.23 | 0.28 / 0.13 |
| hemibrain CX, log1p | 0.70 / 0.53 | 0.60 / 0.30 | 0.50 / 0.23 | 0.28 / 0.13 |
| hemibrain CX, binary | 0.58 / 0.38 | 0.60 / 0.28 | 0.50 / 0.23 | 0.28 / 0.13 |

How to read it:
- On real wiring, both recurrent substrates beat the memoryless baseline, as on synthetic data.
- FlyConnectome vs ShuffledFly differences are a few episodes out of 40 for a single seed, which
  is within noise. The `log1p` robustness gap (0.53 vs 0.30) is the one lead worth re-testing
  over many seeds.
- RandomRecurrent is identical across transforms because it uses a single mean weight and
  spectral rescaling removes the scale.

## Next

- Run many seeds with confidence intervals before reading anything into the gaps.
- Speed: sparse recurrent matrix (~5% density), or a smaller CX core (e.g. intrinsic neurons only).
- Controls for H2: a shuffle that also preserves cell-type block structure.
- Readout restricted to `PFL` outputs, and real sensory encoding for the ring neurons, are TASK-014 questions.
