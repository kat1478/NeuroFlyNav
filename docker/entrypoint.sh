#!/usr/bin/env bash
# Run one trained-mask job inside the container.
#
# Job choice (one of):
#   AWS_BATCH_JOB_ARRAY_INDEX (set by AWS Batch for array jobs) + FIRST_SEED
#   VARIANT + SEED
# Optional: EXTRA_ARGS (e.g. "--core --steps 5"), BUCKET + RUN_ID (upload result to S3).
set -euo pipefail

DATA_URL="https://storage.googleapis.com/hemibrain/v1.2/exported-traced-adjacencies-v1.2.tar.gz"
DATA_SHA256="07d8946eb0c4e3a5cb23d5769c9817847494f9fcadbc0ca239eed7bbd5555cf7"
DATA_DIR="data/hemibrain/v1.2"

mkdir -p "$DATA_DIR" results
if [[ ! -d "$DATA_DIR/exported-traced-adjacencies-v1.2" ]]; then
    curl -fsSL --retry 3 -o "$DATA_DIR/hemibrain.tar.gz" "$DATA_URL"
    echo "$DATA_SHA256  $DATA_DIR/hemibrain.tar.gz" | sha256sum -c -
    tar -xzf "$DATA_DIR/hemibrain.tar.gz" -C "$DATA_DIR"
fi

if [[ -n "${AWS_BATCH_JOB_ARRAY_INDEX:-}" ]]; then
    job_args=(--array-index "$AWS_BATCH_JOB_ARRAY_INDEX" --first-seed "${FIRST_SEED:?FIRST_SEED is required}")
    name="index-${AWS_BATCH_JOB_ARRAY_INDEX}"
else
    job_args=(--variant "${VARIANT:?VARIANT or AWS_BATCH_JOB_ARRAY_INDEX is required}" --seed "${SEED:?SEED is required}")
    name="${VARIANT}-${SEED}"
fi
output="results/${name}.json"

# shellcheck disable=SC2086  # EXTRA_ARGS is intentionally split into words
python -m neuroflynav.experiments.run_trained "${job_args[@]}" ${EXTRA_ARGS:-} --output "$output" > /dev/null

if [[ -n "${BUCKET:-}" ]]; then
    aws s3 cp "$output" "s3://${BUCKET}/runs/${RUN_ID:?RUN_ID is required with BUCKET}/${name}.json"
else
    cat "$output"
fi
