# Exploratory spike: heading integration

> **Exploratory.** Placeholder dynamics (tanh reservoir, all-excitatory weights), full-state
> linear readout. Numbers are a first look on one task configuration.

## Task

`neuroflynav.tasks.heading`: the agent turns with a smoothly varying angular velocity
(AR(1), std 0.15 rad/step). Its absolute heading is shown for the first 3 steps and at a
landmark every 8 steps; in between it gets only the angular velocity. At every step without a
landmark it must report its heading as one of 8 sectors (chance 0.125). This is what the fly's
central-complex compass does: combine visual landmarks with self-motion.

A **hold-last-landmark oracle** (perfect memory of the last landmark, no integration) gives the
level that memory alone can reach.

## Run

```bash
PYTHONPATH=src python -m neuroflynav.experiments.run_heading --graph hemibrain --seeds 20
```

Defaults (landmark every 8 steps, 1 reservoir micro-step per task step, ridge λ 0.01, turn std
0.15) were calibrated on pilot sequences: with 3 micro-steps and no landmarks every policy was at
chance, because the reservoir's memory decays to nothing within ~37 steps.

## Pre-registered result (seeds 0–19, `log1p` weights)

| Policy | Sector accuracy | Fly − policy (paired) |
| --- | --- | --- |
| FlyConnectome | 0.389 [0.375, 0.404] | — |
| TypeShuffledFly | 0.409 [0.397, 0.420] | −0.019 [−0.037, −0.001] |
| ShuffledFly | 0.359 [0.352, 0.366] | +0.031 [+0.017, +0.044] |
| RandomRecurrent | 0.356 [0.349, 0.364] | +0.033 [+0.018, +0.048] |
| Reactive | 0.115 [0.107, 0.123] | +0.275 [+0.257, +0.293] |
| hold-last-landmark oracle | 0.567 [0.555, 0.578] | — |

Reading:
- The real wiring again beats the degree-preserving shuffle and the random graph (+0.03).
- Rewiring within cell types does not hurt — it is even slightly better. As on the hidden-cue
  task, cell-type structure carries the benefit; exact partners add nothing here.
- No policy reaches the memory-only oracle, so these reservoirs act as a fading memory of the
  last landmark, not as an integrator. The task does not yet test integration.
- The likely missing piece is **inhibition**: the fly compass is a ring attractor built with
  inhibitory Δ7 neurons, while all weights here are positive. Signed weights are the next test.

## Signed weights (pre-registered, seeds 20–39)

Exploratory sign rule: outgoing edges of Δ7 (glutamatergic) and ring `ER*` (GABAergic) neurons
are inhibitory, all others excitatory (9.5% of edges negative). Controls: **RandomSignFly** (same
wiring, the same number of inhibitory neurons chosen at random) and **UnsignedFly** (all positive).

```bash
PYTHONPATH=src python -m neuroflynav.experiments.run_heading --graph hemibrain --signs types \
    --first-seed 20 --seeds 20
```

| Paired difference | Accuracy |
| --- | --- |
| SignedFly − UnsignedFly | −0.027 [−0.034, −0.021] |
| SignedFly − RandomSignFly | −0.040 [−0.056, −0.025] |
| SignedFly − ShuffledFly (signed) | −0.013 [−0.024, −0.001] |

SignedFly 0.34, hold-last-landmark oracle 0.55.

Reading: biologically placed inhibition makes this reservoir *worse*, worse even than randomly
placed inhibition, and still nothing integrates. A ring attractor needs the right dynamics, not
just the right signs; with a generic tanh echo-state reservoir, dynamics are the bottleneck.
Next: model dynamics (trained recurrent gains on the CX mask, or a tuned rate model).

## Leaky rate dynamics (pre-registered, seeds 40–59)

The reservoir update becomes `x ← (1−α)·x + α·tanh(W·x + u)` (`--leak`). Calibrated on pilot
sequences only: α = 0.2 (pilot Fly 0.585 vs oracle 0.525); fixed before the run.

```bash
PYTHONPATH=src python -m neuroflynav.experiments.run_heading --graph hemibrain --leak 0.2 \
    --first-seed 40 --seeds 20                 # add --signs types for the signed variant
```

| Paired difference (accuracy) | unsigned | signed (Δ7 + ER) |
| --- | --- | --- |
| **FlyConnectome − oracle** | **+0.069 [+0.056, +0.081]** | +0.069 [+0.056, +0.082] |
| ShuffledFly − oracle | +0.059 [+0.047, +0.072] | +0.059 [+0.045, +0.072] |
| FlyConnectome − ShuffledFly | +0.009 [+0.004, +0.015] | +0.010 [+0.006, +0.014] |
| FlyConnectome − TypeShuffledFly | −0.010 [−0.015, −0.006] | −0.012 [−0.017, −0.008] |
| SignedFly − UnsignedFly | — | +0.000 [−0.003, +0.003] |
| SignedFly − RandomSignFly | — | −0.001 [−0.005, +0.003] |

Accuracies: FlyConnectome 0.623, TypeShuffledFly 0.633, ShuffledFly 0.613, RandomRecurrent 0.614,
Reactive 0.116, hold-last-landmark oracle 0.554.

Reading:
- **Integration works**: with leaky dynamics the reservoir beats perfect memory of the last
  landmark — it adds up self-motion. But every recurrent control does too, so integration
  comes from the dynamics, not from the CX wiring.
- Real wiring keeps a small edge over degree-preserving and random graphs (~+0.01); cell-type
  structure is again enough (type-shuffled slightly better).
- Signs no longer matter: the earlier harm from Δ7/ER inhibition disappears with leaky dynamics.
- Next: train recurrent gains on the CX mask, the first setup where exact wiring could be used.
