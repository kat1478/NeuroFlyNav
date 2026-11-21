---
title: "Does fly-brain wiring help an artificial navigation network?"
subtitle: "NeuroFlyNav — technical report, v0.9.0"
author: "Katarzyna Kadyszewska"
date: "November 2026"
---

**Summary.** I tested whether the real wiring of the fruit fly's central complex — the brain region
insects use to keep track of heading and to navigate — gives artificial recurrent networks a useful
starting point compared with fair controls of the same size. Across nine pre-registered comparisons
the answer is mostly *no*: with frozen weights the real wiring has small edges over random wiring that
disappear once cell-type structure is kept. The one exception is the sparse wiring of the whole central
complex (2,966 neurons) with trained recurrent weights: there the real wiring beats every control,
including a within-cell-type shuffle (+1.1 points), and tracks heading **+2.8 percentage points**
better than a random wiring of the same size (95% CI +2.1 to +3.6). That small edge does not
carry over to a harder use of the same estimate: flying home after foraging over a map of Warsaw
(**−1.2 m** [−4.1, +1.7], no advantage) — as long as code sums the heading estimates into a home
vector. When the network has to compute the home vector itself, the real wiring lands the fly
**14.7 m closer** to the nest than a random wiring of the same size (−14.7 m [−18.8, −10.7]); most of
that edge again comes from cell-type structure. Large sweeps ran as containerised AWS Batch jobs defined in
Terraform; the results are browsable in a public web app, where visitors can also drop their own
fly on the map and watch the two trained networks fly it home, computed live in the browser. Code, data provenance and all decision rules
are public. Everything here is exploratory.

Live app: <https://kat1478.github.io/NeuroFlyNav/> · Code: <https://github.com/kat1478/NeuroFlyNav>

# 1. Question

Connectomes — complete wiring diagrams of brains — are now available for the fruit fly. A natural
engineering question is whether such a diagram is a good *inductive bias*: does building a network on
the real wiring make it learn or generalise better than building it on a random or rewired graph?
The central complex (CX) is a good test case: it holds a "compass" whose activity tracks the fly's
heading, and it is thought to support path integration — summing turns and distances to fly home.

Hypotheses: **H1** the real wiring beats the same graph rewired; **H0** no difference once size and
training budget are matched; **H2** if there is an edge, it comes from the exact neuron-to-neuron map
rather than from which cell types connect.

# 2. Data

- **hemibrain v1.2** (Janelia FlyEM; Scheffer et al., 2020; CC BY 4.0), public export of traced
  neurons and synapse counts, downloaded with a SHA-256 check and kept out of Git.
- **CX subgraph:** every neuron with at least one synapse in the six CX regions (AB left and right, EB,
  FB, NO, PB) and every connection between them with weight ≥ 1 — **2,966 neurons, 467,579 directed
  edges (5.3% of all pairs)**. The "compass core" (EPG, PEN, PEG, Δ7, ring neurons) has 409 neurons
  and is much denser (26%). Edge weights `log1p(synapse count)`.
- **City map:** a dated OpenStreetMap extract of central Warsaw (Geofabrik, ODbL), turned into vector
  tiles and 326 "treats" (cafés, bakeries, confectioneries, ice-cream shops).

# 3. Methods

**Models.** (a) A *reservoir*: the wiring as a fixed recurrent weight matrix (spectral radius 1.1,
tanh or leaky rate dynamics), observations entering at the ring neurons, only a linear readout
learned by ridge regression. (b) A *masked RNN*: the same leaky dynamics, but every connectome edge
carries a weight learned by backpropagation through time (PyTorch; Adam, 300 steps, lr 3·10⁻³,
batch 32); connections absent from the graph stay zero.

**Tasks.** (1) *Hidden cue*: reach a goal in a grid world whose direction was shown only briefly.
(2) *Heading*: report the current heading (one of 8 sectors) from self-motion, with a landmark every
8 steps — this requires integrating turns. (3) *Homing*: a simulated fly searches the Warsaw map until
a treat is within 15 m, then flies back along a home vector built from the network's heading
estimates at every step. (4) *Home vector*: the network itself reports where the nest is, in map
coordinates, at every step (regression readout, mean squared error, 1000 Adam steps on synthetic
100-step flights), tested on the same Warsaw flights.

**Controls.** Same size and budget, different wiring: degree-preserving rewiring (`ShuffledFly`),
rewiring only within cell types (`TypeShuffledFly`, which keeps the cell-type-to-cell-type wiring),
random wiring with the same number of edges (`RandomRecurrent` / `RandomMask`), real wiring with frozen
recurrent weights (`FrozenFly`), and no memory (`Reactive`).

**Statistics.** 20 independent seeds per experiment, differences paired within seed, 95% confidence
intervals (±1.96 SE). Every reported comparison had its design and decision rule written down before
the run, on seeds never used before, with the code committed first; the rules and run commits are
published in [docs/preregistration.md](https://github.com/kat1478/NeuroFlyNav/blob/develop/docs/preregistration.md).

# 4. Results

| # | Setting | Key comparison (real wiring − control) | Reading |
| - | ---------- | ------------------------ | ------------ |
| 1 | Hidden cue, frozen reservoir | +0.06 [+0.02, +0.11] vs degree-shuffled; −0.01 [−0.05, +0.02] vs type-shuffled | small edge, explained by cell-type structure |
| 2 | Heading, tanh reservoir | +0.03 vs degree-shuffled; below a memory-only reference | no integration yet |
| 3 | Heading, biologically placed inhibition | −0.03 vs unsigned | signs hurt with these dynamics |
| 4 | Heading, leaky reservoir | beats the memory reference by +0.07 — as do all controls; +0.01 vs degree-shuffled | integration comes from dynamics |
| 5 | Heading, trained weights, compass core | 0.84 vs 0.52 frozen; −0.001 [−0.007, +0.005] vs random | training matters, wiring does not |
| 6 | Same, little training data | −0.024 [−0.029, −0.020] vs random (pooled) | real wiring slightly *worse* |
| 7 | **Heading, trained weights, whole CX** | **+0.028 [+0.021, +0.036] vs random**; +0.037 vs degree-shuffled; +0.011 vs type-shuffled | **real wiring helps on the sparse graph** |
| 8 | **Homing over Warsaw** (networks of 7) | **−1.2 m [−4.1, +1.7]** vs random (27.2 vs 28.4 m) | **no homing advantage** |
| 9 | **Network computes the home vector** | **−14.7 m [−18.8, −10.7]** vs random (47.1 vs 61.9 m); −2.0 [−5.0, +0.9] vs type-shuffled | **real wiring helps; cell-type structure carries it** |

![Real wiring minus each control in the five heading and hidden-cue experiments with full control
sets (95% CI, 20 seeds). Only the trained whole CX beats every control.](figures/results-forest-light.png)

**What the results say.** With frozen weights the real wiring has a small edge that comes from which
cell types connect, not from the exact neuron map. Integration of self-motion comes from the dynamics
and from training, not from the wiring. With trained weights, a dense mask (the compass core) gives no
advantage — almost every connection exists anyway — but the sparse whole CX does: the wiring constrains
learning in a useful way, and most of that advantage survives shuffling within cell types. On the
homing task all trained networks integrate (27–29 m from the nest, versus 32 m without integration),
but heading errors accumulate along a flight; a 1.7-point gap in heading accuracy on these flights
(0.823 vs 0.806) is too small to bring the fly measurably closer to home. When the network must sum
the vector itself, the gap opens: the real wiring (and a within-cell-type shuffle) learns the
integration clearly better than random or degree-shuffled wiring — consistent with the fan-shaped
body's structure being a useful scaffold for vector computation. The network's own vector (47 m) is
still less accurate than code summing its heading (27 m): the test flights run up to three times the
training length (on 100-step flights the error is 18 m). A pilot criterion of this experiment (land
clearly below the ~32 m code-summed reference) was not met; this was recorded before the main run.

# 5. Engineering

- **Code:** typed Python package (mypy strict), one `Graph` interface shared by every data source, task
  and model; 89 Python and 19 TypeScript tests; ruff; CI on every pull request; semantic-versioned
  releases with a changelog; a provenance command captures commit, package versions and environment
  hash, cloud images are tagged with the git commit, and the run commits are listed with the
  pre-registered rules.
- **Cloud:** each (variant, seed) is an independent job. A Docker image (base pinned by digest, non-root)
  tagged with the git commit runs as an AWS Batch array job on Fargate Spot; jobs download the public
  connectome, check its SHA-256 and write one JSON to a private S3 bucket. Infrastructure (ECR, S3, IAM
  roles with least privilege, Batch, CloudWatch, cost budget) is defined in Terraform. A 100-job sweep
  took 7.8 job-hours in 3.3 h wall time instead of ~30 h on a laptop; the day of the whole-CX run cost
  0.63 USD in AWS Cost Explorer. A pilot job matched the local result exactly. The job image is built by
GitHub Actions and published to GitHub Container Registry with the built-in token (no stored keys;
OIDC to AWS is blocked by the account's organization policy); the home-vector sweep (76 job-hours)
ran from that image.
- **Web app:** static, no server: Python precomputes small JSON files; Vite + TypeScript, Observable Plot
  for charts, MapLibre with a self-hosted PMTiles map; light/dark, WCAG AA contrast, ARIA, reduced
  motion; independent UI/UX audits before release; GitHub Pages plus an S3 + CloudFront mirror.
- **Model in the browser:** the two trained whole-CX networks are exported to ONNX as a one-step cell
  that keeps the recurrence sparse (gather the state at each edge's source, multiply by the weight,
  scatter-add at the target; int32 indices, 5.8 MB per model instead of a dense 2,966 × 2,966
  matrix). The exported cells agree with PyTorch at every step (agreement 1.0; a golden test checks
  the browser code path in CI) and run with ONNX Runtime Web in WebAssembly, single-threaded, about
  10 ms per step for both networks, loaded only when a visitor drops a fly (~9 MB compressed). The
  models carry neuron-level connectivity, so they are published as a checksummed release asset,
  not committed to Git.

# 6. Limitations

One family of rate-based models; small synthetic tasks; heading reported in 8 sectors; several
comparisons per experiment without correction for multiple testing; the overall protocol (dynamics,
readout, metrics) is not frozen, and no held-out confirmation has been run. The homing flight model
ignores streets and obstacles and keeps speed constant; the home-vector networks were trained on
flights shorter than the test flights. The decision
rules were recorded in project notes rather than an external registry.

# 7. Next steps

Train the home-vector network on longer and variable-speed flights and ask which fan-shaped-body cell
types (PFN, hΔ) carry the advantage; test a random mask with matched degree distribution (are hub neurons what limits learning?);
freeze a protocol and run a one-shot held-out confirmation.

# References

Scheffer, L. K. et al. (2020). A connectome and analysis of the adult *Drosophila* central brain.
*eLife* 9:e57443. · Hulse, B. K. et al. (2021). A connectome of the *Drosophila* central complex reveals
network motifs suitable for flexible navigation and context-dependent action selection. *eLife*
10:e66039. · Map data © OpenStreetMap contributors (ODbL).
