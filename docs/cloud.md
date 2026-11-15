# Cloud runs (M3)

The whole-CX trained-mask sweep (2,966 neurons, 20 seeds × 5 variants = 100 jobs) is too slow
for a laptop (~90 min per seed). It runs as an AWS Batch array job: one container per
(variant, seed). This page grows with each M3 step; infrastructure code comes next.

## Container

```bash
docker build -t neuroflynav:dev .
# one job by name
docker run --rm -e VARIANT=FlyConnectome -e SEED=100 \
    -e EXTRA_ARGS="--core --steps 5 --train 8 --test 4" neuroflynav:dev
# one job by array index, as AWS Batch runs it (index 7, first seed 40 -> ShuffledFly, seed 41)
docker run --rm -e AWS_BATCH_JOB_ARRAY_INDEX=7 -e FIRST_SEED=40 \
    -e EXTRA_ARGS="--core --steps 2" neuroflynav:dev
```

- Base image `mambaorg/micromamba:2.0.5`, pinned by digest; runs as the non-root `mambauser`.
- The Python environment comes from the same `environment.yml` (conda-forge) as development,
  plus `awscli` and `curl`.
- The hemibrain v1.2 export is **not** in the image. Each job downloads it from the public
  Janelia bucket and checks its SHA-256 before use (see [spike_m1.md](spike_m1.md)).
- With `BUCKET` and `RUN_ID` set, the result JSON is uploaded to
  `s3://$BUCKET/runs/$RUN_ID/<job>.json`; otherwise it is printed.
- Merge downloaded results with `aggregate_trained` (see [spike_trained.md](spike_trained.md)).

Smoke check (2026-11-15): the container result for `FlyConnectome`, seed 100, core, 5 steps
equals the local run exactly. Image size 3.4 GB (PyTorch CPU + dev tools); first build
~12 min, later builds reuse the cached environment layer.
