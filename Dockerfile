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

# Fail the BUILD, not the pipeline, if the vocabulary stops being packaged.
# data_samples/ is excluded in .dockerignore with two `!` exceptions, so a later
# edit to either file's path — a rebuilt vocabulary under a new name, a tightened
# ignore rule — silently produces an image whose own llm_config.txt points at
# nothing. That shipped once already and only surfaced as a cross-repo e2e failure
# in another repository (atrium-project run 34032532443). Two stat calls here turn
# a packaging regression back into a build error.
RUN test -f data_samples/vocab/union_nested.json \
    && test -f data_samples/taxonomy_config.json \
    || (echo "ERROR: runtime vocabulary missing from the image - check .dockerignore" >&2; exit 1)

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
# Digital-born converter — published as :<version>-digital  (W8)
#
# api_util/digital_to_json.py turns a born-digital PDF/DOCX directly into an
# atrium_document record. It is an ORIGINATOR, like page-classification's scan
# path: it takes no --document-json baseline, it creates the record.
#
# Why a separate stage rather than folding it into `remote`: the two have disjoint
# dependency sets and disjoint reasons to exist. `remote` talks to OpenRouter and
# needs no document parsing; this needs pdfplumber/python-docx and no network at
# all. Merging them would put a PDF parser in the image whose whole selling point
# is being the torch-free API client.
#
# NOTE ON THE MANIFEST: requirements_digital.txt currently also declares `docling`
# and `docx2python`, which NOTHING SHIPPED IMPORTS — digital_to_json.py imports
# pdfplumber (line ~427) and docx (line ~496) lazily, and jsonschema arrives via
# atrium_document.validate_document(). They are left in the manifest because the
# licence posture documented there is load-bearing (accretion rule 5 merges
# component licences into provenance.license for every digital-born document, so
# the MIT-only stack is a deliberate constraint, not a preference) and dropping a
# name from that file without also dropping its para_config.txt [components] row
# would make ParadataLogger record it as UNKNOWN — which para_licenses treats as
# maximally restrictive. Splitting a runtime subset out of the manifest is the
# right fix and is a licence-review change, not a Dockerfile one; until then this
# stage installs the declared manifest so the image matches what para_config.txt
# claims is in it.
# ---------------------------------------------------------------------------
FROM base AS digital

USER root
COPY requirements_digital.txt ./
RUN pip install -r requirements_digital.txt
RUN chown -R atrium:atrium /app
USER atrium

ENTRYPOINT ["python", "api_util/digital_to_json.py"]
CMD ["--help"]


# ---------------------------------------------------------------------------
# Local multi-GPU variant — published as :<version>-llm
# ---------------------------------------------------------------------------
FROM base AS llm

USER root
COPY requirements_llm.txt ./
RUN pip install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements_llm.txt

RUN chown -R atrium:atrium /app
USER atrium

ENTRYPOINT ["python", "llm_run.py"]
CMD ["llm_config.txt"]


# ---------------------------------------------------------------------------
# API service variant — published as :<version>-api
# FastAPI meta-contract service (strategy §4) wrapping the torch-free remote /
# lightweight-local enrichment engine. Built on the remote stack + web server.
# ---------------------------------------------------------------------------
FROM remote AS api

USER root
COPY service/requirements.txt ./service/requirements.txt
RUN pip install -r service/requirements.txt
RUN chown -R atrium:atrium /app
USER atrium

EXPOSE 8000
ENTRYPOINT ["uvicorn", "service.api:app", "--host", "0.0.0.0", "--port", "8000"]
CMD []