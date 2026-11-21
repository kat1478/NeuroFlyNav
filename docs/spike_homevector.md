# The network computes the home vector itself (exploratory, pre-registered)

In the homing test ([spike_homing.md](spike_homing.md)) the trained network reported only its
heading, and code summed the steps into the home vector; the real wiring gave no homing advantage.
Insects are thought to compute the home vector inside the central complex, in the fan-shaped body
(PFN and hΔ neurons). Here the network has to do the summing itself: does the real wiring help?

## Task

`neuroflynav.tasks.homevector`, run by `neuroflynav.experiments.run_homevector`:

- At every step the network reports **where the nest is relative to the fly**, in map coordinates
  (x east, y north, divided by 100 m) — a regression readout trained with mean squared error.
  Inputs are the heading task's: self-motion (angular velocity) and the heading at a landmark glimpse
  every 8 steps. Speed is constant (5 m per step), so distance is the number of steps.
- Training: 256 synthetic flights of 100 steps (same turning statistics as the homing task, no map),
  whole CX (2,966 neurons), 1000 Adam steps (the heading task used 300; a compass-core smoke test
  showed the regression needs more), otherwise the model and budget of
  [spike_trained.md](spike_trained.md).
- Test: the **same Warsaw foraging flights** as the homing test (nest in Park Świętokrzyski, 326 OSM
  treats, stop within 15 m of a treat, at most 300 steps; 100 flights per seed). The **homing error**
  is the distance between the network's home vector at the last outbound step and the true one —
  where the fly lands flying back along its own estimate.
- References: staying put; and the homing test's code-summed vector from the true heading sectors
  (quantisation only) and from the last landmark's sector (no heading integration).

```bash
PYTHONPATH=src python -m neuroflynav.experiments.run_homevector --variant FlyConnectome --seed 100 \
    --steps 1000 --output results/homevector/job.json
PYTHONPATH=src python -m neuroflynav.experiments.aggregate_homevector results/homevector/jobs
```

## Pilot and an unmet criterion (stated openly)

A pilot on seeds 100–102 (15 jobs, not part of the test) checked that the task separates trained
from untrained networks (frozen 131 m vs trained 43–59 m; staying put 184 m). The plan had also asked
for the real wiring to land clearly below the code-summed last-landmark reference (~32 m); on the
Warsaw endpoint it did not (43 m). That reference sums steps in code with perfect distance, so it is
not a network without integration; we went ahead with this caveat recorded before the main run.

## Pre-registered result (whole CX, seeds 80–99, 100 flights per seed)

Run as 100 AWS Batch jobs from the image built by GitHub Actions for the run commit
([cloud.md](cloud.md)): 100/100 succeeded, 76.4 job-hours. Rule fixed before the run: Fly − RandomMask
of the per-seed mean homing error, paired over seeds, 95% CI entirely below 0.

| Variant | Homing error, Warsaw (m) | Mean error along the flight (m) | Synthetic 100-step flights (m) | Fly − variant (m, paired) |
| --- | --- | --- | --- | --- |
| FlyConnectome (real wiring) | 47.1 [43.0, 51.2] | 19.9 | 17.7 | — |
| TypeShuffledFly | 49.2 [45.1, 53.2] | 20.6 | 18.0 | −2.0 [−5.0, +0.9] |
| RandomMask (same size) | 61.9 [56.0, 67.8] | 29.7 | 28.7 | **−14.7 [−18.8, −10.7]** |
| ShuffledFly (degree-preserving) | 61.5 [56.9, 66.1] | 30.1 | 29.6 | −14.4 [−17.7, −11.1] |
| FrozenFly (recurrent weights frozen) | 131.8 [127.7, 136.0] | 97.6 | 134.0 | −84.7 [−88.2, −81.3] |
| staying put | 187.6 [179.9, 195.2] | — | — | — |
| code-summed, true sectors | 12.4 [11.9, 12.9] | — | — | — |
| code-summed, last-landmark sectors | 32.5 [31.4, 33.6] | — | — | — |

**The real wiring computes a better home vector.** It lands 14.7 m closer to the nest than a
same-size random wiring (about a quarter less error), and as much closer than a degree-preserving
rewiring; both intervals lie well below zero. Against a rewiring that keeps which cell types connect,
the difference shrinks to 2.0 m and the interval includes zero — as on the heading task, most of the
advantage comes from cell-type structure rather than the exact neuron-to-neuron map.

Caveats: the network's own vector is still less accurate than code summing its heading estimates
(27.2 m in the homing test) — the Warsaw flights run up to 300 steps, three times the training
length, and on in-distribution 100-step flights the error is 17.7 m. One model family, constant speed,
a flight model that ignores streets. Exploratory.
