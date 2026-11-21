# Pre-registered decision rules

Each comparison reported as "pre-registered" in this repository had its design and decision rule
written down **before** the run, on seeds that had never been used, and its code committed before
the run. The rules were kept in the project's working notes; this page publishes them, unchanged
in substance, with the commit each run used. Results are reported whatever they showed, including
outcomes the rules did not anticipate.

**What this does and does not guarantee.** The commit before each run is public and dated, so the
code and the seeds can be checked. The notes themselves were private at the time, so the claim that
each rule predates its result rests on the project's record-keeping, not on a third-party registry.
Several comparisons are made per experiment without correction for multiple testing; the overall
protocol (dynamics, readout, metrics) is not frozen. All results are exploratory.

Statistics everywhere: per-seed paired differences FlyConnectome − control, mean ± 1.96 standard
errors over 20 seeds (a "CI > 0" reading means the whole interval lies above zero).

## Overview

| # | Experiment | Seeds | Primary rule (in short) | Outcome | Run commit | Details |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Hidden-cue task: replication + cell-type shuffle | 20–39 | Fly − Shuffled CI > 0 → replicates; Fly − TypeShuffled CI ≈ 0 → cell-type structure explains it | replicates; explained by cell types | `70e25ff` | [spike_m1.md](spike_m1.md) |
| 2 | Heading task, tanh reservoir | 0–19 | Fly − Shuffled / − TypeShuffled; Fly vs memory-only oracle | +0.03 over Shuffled; cell types enough; no integration | `2dc8074` | [spike_heading.md](spike_heading.md) |
| 3 | Heading task, signed weights | 20–39 | Signed − Unsigned and − RandomSign CI > 0 → inhibition / its placement helps | signs hurt (−0.03) | `d8ad897` | [spike_heading.md](spike_heading.md) |
| 4 | Heading task, leaky dynamics | 40–59 | Fly − oracle CI > 0 → integration; Fly − Shuffled / − TypeShuffled | integration for all; +0.01 over Shuffled | `cd06c12` | [spike_heading.md](spike_heading.md) |
| 5 | Trained weights, compass core | 0–19 | Fly − RandomMask / − Shuffled / − TypeShuffled CI > 0 | no edge over random or degree-shuffled masks; +0.007 [+0.0004, +0.014] over type-shuffled | `bb0aa7f` | [spike_trained.md](spike_trained.md) |
| 6 | Trained weights, compass core, little data | 20–39 | **one test:** Fly − RandomMask pooled over n = 16/32/64, CI > 0 | opposite: −0.024 | `3595624` | [spike_trained.md](spike_trained.md) |
| 7 | Trained weights, whole CX (AWS) | 40–59 | **one test:** Fly − RandomMask CI > 0 | **yes: +0.028 [+0.021, +0.036]** | `1405eb5` | [spike_trained.md](spike_trained.md) |
| 8 | Homing over central Warsaw (AWS) | 60–79 | **one test:** Fly − RandomMask homing error CI < 0 | **no: −1.2 m [−4.1, +1.7]** | `81967d0` | [spike_homing.md](spike_homing.md) |
| 9 | Network computes the home vector (AWS) | 80–99 | **one test:** Fly − RandomMask homing error CI < 0 | **yes: −14.7 m [−18.8, −10.7]**; vs type-shuffled −2.0 [−5.0, +0.9] | `62f1b95` | [spike_homevector.md](spike_homevector.md) |

## 1. Hidden-cue task: replication and cell-type shuffle (2026-10-21)

- Hemibrain v1.2 CX subgraph (2,966 neurons, 467,579 edges); 7×7 grid, 3 cue steps, 80 training /
  40 evaluation episodes, reservoir spectral radius 1.1, ridge λ = 0.01.
- Primary weights `log1p`; secondary `binary` (reported, decides nothing).
- New control TypeShuffledFly: double-edge swaps only between edges whose targets share a cell type
  (keeps every neuron's degrees and the edge count between every pair of cell types).
- Metric: success rate on the hidden-cue task.

| Comparison | Reading |
| --- | --- |
| Fly − ShuffledFly, CI lower bound > 0 | the edge over degree-only wiring replicates |
| Fly − ShuffledFly, CI includes 0 | the earlier result does not replicate |
| Fly − TypeShuffledFly includes 0 and TypeShuffled ≥ Shuffled | the edge is explained by cell-type structure |
| Fly − TypeShuffledFly, CI lower bound > 0 | exact partners carry information beyond cell types |

## 2. Heading task, tanh reservoir (2026-10-21)

- `log1p` weights; policies FlyConnectome, ShuffledFly, TypeShuffledFly, RandomRecurrent, Reactive.
- Metric: sector accuracy on steps with no heading shown; reference: hold-last-landmark oracle.

| Comparison | Reading |
| --- | --- |
| Fly − Reactive, CI > 0 | the reservoir carries heading between landmarks |
| Fly − ShuffledFly, CI > 0 | real wiring beats degree-only wiring |
| Fly − TypeShuffledFly, CI > 0 / includes 0 | exact partners matter / cell-type structure is enough |
| Fly above the oracle | actual integration, not just memory |

## 3. Heading task, signed weights (2026-10-21)

- Same task and reservoir; Δ7 and ring (ER) neurons inhibitory, everything else excitatory.
- Extra controls: RandomSignFly (same wiring, same number of inhibitory neurons chosen at random),
  UnsignedFly (all positive); in the shuffles signs travel with the presynaptic neuron.

| Comparison | Reading |
| --- | --- |
| Signed − UnsignedFly, CI > 0 | inhibition at Δ7/ER helps |
| Signed − RandomSignFly, CI > 0 | which neurons inhibit matters |
| either CI includes 0 | no evidence for that effect with these dynamics |

## 4. Heading task, leaky dynamics (2026-10-24)

- Primary run unsigned; secondary run with the sign rule of experiment 3.

| Comparison | Reading |
| --- | --- |
| Fly − oracle, CI > 0 | integration: beats perfect memory of the last landmark |
| Fly − ShuffledFly, CI > 0 | real wiring beats degree-only wiring under leaky dynamics |
| Fly − TypeShuffledFly, CI > 0 / includes 0 or below | exact partners matter / cell types are enough |

## 5. Trained recurrent weights, compass core (2026-10-24)

- Compass core (EPG, PEN, PEG, Δ7, ER): 409 neurons, 43,629 edges; leaky rate RNN (α = 0.2); every
  mask edge has a trainable weight; input and readout trained. Budget per model: 300 Adam steps,
  lr 3e-3, batch 32, 256 training and 64 test sequences, cross-entropy on scored steps.
- Variants: FlyConnectome, FrozenFly, ShuffledFly, TypeShuffledFly, RandomMask.

| Comparison | Reading |
| --- | --- |
| Fly − FrozenFly, CI > 0 | training the recurrent weights helps |
| Fly − oracle, CI > 0 | the trained model integrates beyond memory |
| Fly − RandomMask / − ShuffledFly / − TypeShuffledFly, CI > 0 | the real wiring is a better scaffold than that control |
| any of these includes 0 | no evidence for that wiring effect |

## 6. Little training data, compass core (2026-10-25)

- Same model and budget as 5; only the number of training sequences changes: n ∈ {16, 32, 64}.
- **Primary (one test):** Fly − RandomMask averaged per seed over the three sizes, CI > 0 → the real
  mask is a better scaffold when data is scarce. Secondary (no decision alone): each n separately.

## 7. Trained weights, whole CX on AWS Batch (2026-11-16)

- Whole CX subgraph, same model and budget as 5; 100 AWS Batch jobs, one container image.
- **Primary (one test):** Fly − RandomMask, 95% CI > 0 → the real wiring is a better trainable
  scaffold than random wiring of the same size. Secondary: − ShuffledFly, − TypeShuffledFly, − FrozenFly.

## 8. Homing by path integration over central Warsaw (2026-11-17)

- The trained networks of 7 (same training, seeds 60–79), not retrained for homing. Flights: 5 m
  steps, treat detection 15 m, at most 300 steps, landmark glimpse every 8 steps, 8 heading sectors;
  nest in Park Świętokrzyski; treats from `app/public/data/treats.geojson`; 100 flights per seed.
- Metric per (variant, seed): mean homing error (m) after flying back along the home vector.
- **Primary (one test):** Fly − RandomMask of the per-seed mean homing error, 95% CI entirely below 0
  → the real wiring flies home closer. If the interval includes 0 or lies above it, report "no homing
  advantage". Secondary: other controls, heading accuracy on flights, quantisation-only and
  no-integration references.

## 9. The network computes the home vector (2026-11-19)

- Whole CX; the masked RNN reports the home vector (nest relative to the fly, map frame, /100 m) at
  every step, regression readout, mean squared error, 1000 Adam steps, trained on 256 synthetic
  flights of 100 steps; tested on the Warsaw flights of 8 (100 per seed). Seeds 80–99.
- Metric per (variant, seed): mean homing error (m) = distance between reported and true home vector
  at the last outbound step.
- **Primary (one test):** Fly − RandomMask, 95% CI entirely below 0 → the real wiring computes a better
  home vector. Secondary: − TypeShuffledFly, − ShuffledFly, − FrozenFly; error along the flight and on
  synthetic flights; references.
- Recorded before the run: the plan's pilot criterion (Fly clearly below the code-summed last-landmark
  reference, ~32 m) was not met on the Warsaw endpoint (pilot 43 m); the run went ahead with this caveat.

