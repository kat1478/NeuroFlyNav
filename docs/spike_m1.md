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

One seed takes about 30 s on a laptop CPU (sparse reservoir, ARPACK spectral radius,
vectorized controls). Many seeds with confidence intervals:

```bash
PYTHONPATH=src python -m neuroflynav.experiments.run_m1_seeds --seeds 20
```

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

## 20 seeds (seeds 0–19, same settings)

Mean success rate with a 95% normal-approximation CI; last row is the paired per-seed
difference FlyConnectome − ShuffledFly (same episodes for both).

| Policy | raw | log1p | binary |
| --- | --- | --- | --- |
| FlyConnectome | 0.67 [0.63, 0.72] | 0.64 [0.59, 0.68] | 0.61 [0.57, 0.66] |
| ShuffledFly | 0.61 [0.57, 0.64] | 0.56 [0.53, 0.60] | 0.56 [0.53, 0.60] |
| RandomRecurrent | 0.57 [0.53, 0.61] | 0.57 [0.53, 0.61] | 0.57 [0.53, 0.61] |
| Reactive | 0.31 [0.30, 0.32] | 0.31 [0.30, 0.32] | 0.31 [0.30, 0.32] |
| **Fly − Shuffled** | +0.066 [+0.000, +0.132] | +0.076 [+0.022, +0.131] | +0.051 [+0.000, +0.102] |
| Fly − Shuffled, robustness probe | +0.020 [−0.032, +0.072] | +0.015 [−0.037, +0.067] | +0.026 [−0.029, +0.082] |

Reading:
- The single-seed `log1p` robustness lead was noise: across 20 seeds the robustness gap is ~0.
- There is a small in-distribution edge of real wiring over the degree-preserving shuffle
  (+5 to +8 points), clearest for `log1p`. (The `binary` CI touches 0 and did not replicate on
  fresh seeds — see below.)
- Still exploratory: three transforms were compared (no multiple-comparison correction), the CI
  is a normal approximation over 20 seeds, and dynamics/readout are placeholders.

## Pre-registered follow-up: replication and cell-type shuffle (H2), seeds 20–39

The analysis plan (fresh seeds 20–39, `log1p` primary, `binary` secondary, decision rules) was
fixed before running. New control **TypeShuffledFly** swaps partners only within the same
hemibrain cell type (370 types): every neuron keeps its in/out degree and the type-to-type edge
counts are preserved; only exact partners inside a type are randomized.

| Paired difference (mean, 95% CI) | `log1p` (primary) | `binary` |
| --- | --- | --- |
| FlyConnectome − ShuffledFly | **+0.061 [+0.017, +0.105]** | −0.001 [−0.039, +0.036] |
| FlyConnectome − TypeShuffledFly | **−0.014 [−0.048, +0.020]** | −0.023 [−0.064, +0.019] |
| FlyConnectome − RandomRecurrent | +0.059 [+0.023, +0.095] | +0.005 [−0.034, +0.044] |
| FlyConnectome − Reactive | +0.305 [+0.273, +0.337] | +0.251 [+0.215, +0.287] |

Reading:
- The edge over the degree-preserving shuffle **replicates** on fresh seeds with `log1p` weights.
- It is **explained by cell-type structure**: rewiring exact partners within types costs nothing
  (TypeShuffled 0.71 vs Fly 0.70). On this task, which *types* connect to which — with
  synapse-count magnitudes — carries the benefit, not the exact neuron-to-neuron map.
- With wiring-only (`binary`) weights there is no edge on fresh seeds; the earlier `binary` result
  did not replicate.
- Exploratory: toy task, placeholder dynamics and readout, all-excitatory weights.

```bash
PYTHONPATH=src python -m neuroflynav.experiments.run_m1_seeds --first-seed 20 --seeds 20 \
    --weights log1p binary --type-shuffle
```

## Next

- A harder task where fine wiring could matter (e.g. path integration with a ring attractor), since type-level structure suffices here.
- Coarser groups (cell-type families) to find how much structure is needed.
- Readout restricted to `PFL` outputs, and real sensory encoding for the ring neurons, are TASK-014 questions.
