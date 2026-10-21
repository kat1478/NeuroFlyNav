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
