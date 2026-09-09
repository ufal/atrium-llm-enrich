# syntax=docker/dockerfile:1.7
FROM python:3.11-slim AS base

ARG ATRIUM_RUNNER_IMAGE=""
ARG ATRIUM_RUNNER_REPO="https://github.com/ufal/atrium-llm-enrich"
ARG ATRIUM_RUNNER_REF=""

ENV ATRIUM_RUNNER_IMAGE=${ATRIUM_RUNNER_IMAGE} \
    ATRIUM_RUNNER_REPO=${ATRIUM_RUNNER_REPO} \
    ATRIUM_RUNNER_REF=${ATRIUM_RUNNER_REF} \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Backend-agnostic base deps only (pydantic/requests/tqdm) — see requirements.txt.
# Heavy (requirements_llm.txt) and light-remote (requirements_remote.txt) deps are
# layered on in the two stages below, so neither pulls in the other's footprint.
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

RUN useradd --create-home --uid 10001 atrium \
    && mkdir -p /cache/huggingface /data \
    && chown -R atrium:atrium /app /cache /data

USER atrium


# ---------------------------------------------------------------------------
# Remote / lightweight-local variant — published as :<version>-remote
# For openrouter_client.py and ollama_client.py: no torch/transformers/vllm/
# bitsandbytes (see requirements_remote.txt). No single default script — pass
# one of the two client modules (+ its args) as the container command.
# ---------------------------------------------------------------------------
FROM base AS remote

USER root
COPY requirements_remote.txt ./
RUN pip install -r requirements_remote.txt
RUN chown -R atrium:atrium /app
USER atrium

ENTRYPOINT ["python"]
CMD ["openrouter_client.py", "--help"]


# ---------------------------------------------------------------------------
# API service variant — published as :<version>-api
# FastAPI wrapper over the remote clients (service/api.py); torch-free, same
# python:3.11-slim non-root pattern as the sibling services, port 8000.
# ---------------------------------------------------------------------------
FROM remote AS api

USER root
COPY service/requirements.txt ./service-requirements.txt
RUN pip install -r service-requirements.txt
RUN chown -R atrium:atrium /app
USER atrium

EXPOSE 8000
# Issue #55 container contract, restored 2026-09-09 to match the default branch.
# STOPSIGNAL is already SIGTERM by default; declared so a later edit cannot change it
# silently — service/api.py's lifespan chains to uvicorn's own handler for it via
# serve_lifecycle (service/atrium_service.py). --timeout-graceful-shutdown bounds
# uvicorn's wait for in-flight requests, and --start-period on HEALTHCHECK covers the
# first-run model download.
#
# This branch had shipped service/healthcheck.py with nothing referencing it and no
# HEALTHCHECK at all, while service/README.md documented the behaviour as live.
STOPSIGNAL SIGTERM
ENTRYPOINT ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000", "--timeout-graceful-shutdown", "20"]
CMD []
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD ["python", "/app/service/healthcheck.py"]

# The local multi-GPU (transformers/vLLM) image lives on the development branch;
# this agent-skill branch is torch-free and serves the API only.