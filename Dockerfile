# Container for one trained-mask job (M3: AWS Batch array task). CPU only.
# Base pinned by digest: mambaorg/micromamba:2.0.5 (Debian 12, non-root user "mambauser").
FROM mambaorg/micromamba:2.0.5@sha256:8d0d4cad074afe3ac93b58dce88bd2659153d3fa6de3a6968f501710e8d3c832

# Same conda-forge environment as development, plus the tools the entrypoint needs.
COPY --chown=mambauser:mambauser environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml \
    && micromamba install -y -n base -c conda-forge awscli curl \
    && micromamba clean -a -y

WORKDIR /app
COPY --chown=mambauser:mambauser pyproject.toml ./
COPY --chown=mambauser:mambauser src ./src
COPY --chown=mambauser:mambauser docker/entrypoint.sh ./entrypoint.sh

ENV PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1

# The micromamba entrypoint activates the base env, then runs our script.
CMD ["bash", "/app/entrypoint.sh"]
