# Homing by path integration over central Warsaw (exploratory, pre-registered)

Does the real central-complex wiring's heading advantage (+2.8 points on the heading task,
[spike_trained.md](spike_trained.md)) help a fly **fly home**? Insects that forage keep a home
vector — the running sum of every step — and fly back along it. Heading errors along the way add
up into the home vector, so homing is a demanding use of a heading estimate.

## Task

`neuroflynav.tasks.homing`, run by `neuroflynav.experiments.run_homing`:

- The nest is in Park Świętokrzyski, central Warsaw. Treats are the 326 cafés, bakeries,
  confectioneries and ice-cream shops of OpenStreetMap in a 2.7 × 2.6 km box
  (`app/public/data/treats.geojson`, built by `geo/build_map.py`; © OpenStreetMap contributors, ODbL).
- The fly flies 5 m steps with the heading task's turning statistics (AR(1), turn_std 0.15) and a
  landmark glimpse every 8 steps, until a treat is within 15 m (or 300 steps). Median flight: 43 steps.
- At every step the trained network reports its heading as one of 8 sectors; the home vector is the
  sum of steps along the sector centres. The fly flies back along it; the **homing error** is how far
  from the nest it lands.
- The network is the masked RNN trained on the 40-step heading task exactly as in the whole-CX run
  (300 Adam steps, 256 sequences); it is not retrained for homing. Pilot: heading accuracy on the
  long flights stays within 0.01 of the 40-step task.
- References: decoding the **true** sectors (quantisation only — the floor for 8 sectors) and keeping
  the sector of the last landmark (no integration).

```bash
PYTHONPATH=src python -m neuroflynav.experiments.run_homing --variant FlyConnectome --seed 100 \
    --flights 100 --output results/homing/job.json
PYTHONPATH=src python -m neuroflynav.experiments.aggregate_homing results/homing/jobs
```

## Pre-registered result (whole CX, seeds 60–79, 100 flights per seed)

Run as 100 AWS Batch jobs ([cloud.md](cloud.md); `MODULE=run_homing`): 100/100 succeeded,
7.7 job-hours, 3.3 h wall time. Rule fixed before the run: Fly − RandomMask of the per-seed mean
homing error, paired over seeds, 95% CI below 0.

| Variant | Mean homing error (m) | Heading accuracy on flights | Fly − variant (m, paired) |
| --- | --- | --- | --- |
| FlyConnectome (real wiring) | 27.2 [24.4, 30.1] | 0.823 | — |
| TypeShuffledFly | 27.1 [24.5, 29.7] | 0.820 | +0.1 [−2.2, +2.4] |
| RandomMask (same size) | 28.4 [26.0, 30.9] | 0.806 | −1.2 [−4.1, +1.7] |
| ShuffledFly (degree-preserving) | 29.2 [26.8, 31.7] | 0.797 | −2.0 [−5.0, +0.9] |
| FrozenFly (not trained) | 166.4 [151.5, 181.2] | 0.399 | −139.1 [−154.5, −123.8] |
| true sectors (quantisation only) | 12.1 [11.7, 12.5] | — | — |
| last landmark (no integration) | 32.0 [31.2, 32.7] | — | — |

**No homing advantage.** The real wiring lands 1.2 m closer to the nest than a same-size random
wiring on average, but the interval includes zero. All trained networks integrate (below the
no-integration reference) yet stay well above the quantisation floor; training matters far more
than wiring. Exploratory: one task family, one model, 8 heading sectors, a flight model that ignores
streets and obstacles.

The interactive City view of the app animates six pilot flights (seed 100, not hand-picked) and
shows this result ([app.md](app.md)).
