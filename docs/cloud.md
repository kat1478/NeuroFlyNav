# Cloud runs (M3)

The whole-CX trained-mask sweep (2,966 neurons, 20 seeds × 5 variants = 100 jobs) is too slow
for a laptop (~90 min per seed). It runs as an AWS Batch array job: one container per
(variant, seed). Infrastructure is code in [infra/](../infra/README.md).

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

## Running a sweep on AWS Batch

Prerequisites: the Terraform stack in `infra/` is applied, and the AWS CLI is logged in
(`aws login --profile neuroflynav --region eu-north-1`).

```bash
export AWS_PROFILE=neuroflynav AWS_REGION=eu-north-1
SHA=$(git rev-parse --short=12 HEAD)          # image tag = the commit it is built from
REPO=$(terraform -chdir=infra output -raw ecr_repository_url)
aws ecr get-login-password | docker login -u AWS --password-stdin "${REPO%%/*}"
docker build -t "$REPO:$SHA" . && docker push "$REPO:$SHA"
eval "$(aws configure export-credentials --format env)"
terraform -chdir=infra apply -var image_tag="$SHA"   # points the job definition at the image

# 100 jobs: array index i -> variant i % 5, seed FIRST_SEED + i // 5
aws batch submit-job --job-name whole-cx --job-queue neuroflynav \
    --job-definition neuroflynav-trained --array-properties size=100 \
    --container-overrides '{"environment":[{"name":"RUN_ID","value":"my-run"},
        {"name":"FIRST_SEED","value":"40"},{"name":"EXTRA_ARGS","value":""}]}'

# when all jobs have succeeded
aws s3 sync "s3://$(terraform -chdir=infra output -raw bucket)/runs/my-run" results/trained/jobs
PYTHONPATH=src python -m neuroflynav.experiments.aggregate_trained results/trained/jobs \
    --output results/trained/my-run.json
```

Every `terraform apply` must pass the same `image_tag`; without it the job definition falls
back to `latest`, which is never pushed. `EXTRA_ARGS` passes options to `run_trained`
(for example `--core` or `--train 64`).

Measured (pilot, 2026-11-16, Fargate Spot, 4 vCPU / 8 GB): one whole-CX job trains in ~4 min;
start-up (capacity + 1 GB image pull) adds ~1-2 min. The account quota of 12 Fargate Spot vCPU
allows 3 jobs at once.
