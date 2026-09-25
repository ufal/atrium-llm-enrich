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

# ── Distro security patches, applied at build time ───────────────────────────
# `python:3.11-slim` is a floating TAG, and nothing in this ecosystem bumps it:
# no repo declares a `docker` dependabot ecosystem (docker_gha_roadmap.md, H6),
# so the base layer is whatever Docker Hub last rebuilt. On 2026-09-13 that layer
# carried perl-base 5.40.1-6 with three FIXABLE CRITICAL CVEs — CVE-2026-13221,
# CVE-2026-42496 and CVE-2026-8376, all fixed in 5.40.1-6+deb13u1. The release
# gate in atrium-project's docker-tool.reusable.yml ("Fail the release on fixable
# CRITICAL vulnerabilities") blocks on exactly that class, and because the
# promotion step is `if: success()`, a blocked release publishes by DIGEST ONLY —
# the `:<version>` and `:latest` tags are never applied.
#
# It has already cost two releases: translator v1.0.0-beta (2026-09-13, both
# targets) and nlp-enrich v0.20.2 (2026-09-15, run 34970419474, all three
# targets). THIS repo had not been tagged since, which is the only reason it had
# not happened here too — the gate is `if: startsWith(github.ref, 'refs/tags/')`,
# so day-to-day `test` pushes never surface it. (atrium-project#53)
#
# `upgrade` rather than `install --only-upgrade perl-base`, deliberately. The gate
# blocks on *fixable* CRITICALs — precisely those the distro already ships a patch
# for — so the fix that matches the gate's own definition is "apply the distro's
# available patches", not a package name that has to be edited by hand the next
# time a different one is announced.
#
# CACHE INTERACTION, which is what makes this hold rather than run once: the build
# uses `cache-from: type=gha`, so an apt layer high in the file would be served
# from cache forever and silently stop patching. It sits HERE, immediately after
# the ENV block that embeds ATRIUM_RUNNER_REF, because CI passes that as
# `github.ref_name` — a value unique to each release tag. The ENV layer therefore
# changes on every release, busting this layer with it, so every released image is
# scanned against a freshly patched base while day-to-day `test` pushes still hit
# the cache. Do not move this above the ENV block.
#
# One apt layer, not two: the upgrade and the install share a single `apt-get
# update`, so the package lists are fetched once and removed once.
# Guarded by tests/test_dockerfile_security_layer.py (atrium-project#53).
RUN apt-get update \
    && apt-get upgrade -y --no-install-recommends \
    && apt-get install -y --no-install-recommends \
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
# The image carries the LIGHT engine only — requirements_digital.txt: pdfplumber,
# pypdfium2, python-docx, jsonschema, all permissive, no models, no network. Until
# 2026-09-25 it also installed `docling` and `docx2python`, which nothing shipped
# imports; docling alone brings torch and the CUDA libraries, several GB, into an image
# whose converter never loads them. The opt-in heavy engine is its own stage below.
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
# Digital-born converter, heavy engine — built on demand, NOT published
#
#   docker build --target digital-docling -t atrium-llm-enrich-digital-docling .
#   docker run --rm -v "$PWD:/data" atrium-llm-enrich-digital-docling \
#       /data/report.pdf --engine docling --document-json-out /data/report.document.json
#
# `--engine docling` (api_util/digital_docling.py): Docling's layout and table models
# decide reading order, headings, furniture and tables on complex PDFs; the light
# engine's lines keep their geometry. Not in docker.yml's build-targets on purpose: a
# torch image of several GB on every push, for an opt-in engine.
#
# The model weights are downloaded HERE, at build time — the build needs the Hugging
# Face Hub, the running container does not — into /opt/docling-models, which
# DOCLING_ARTIFACTS_PATH names. TableFormer's weights are CDLA-Permissive-2.0: see
# requirements_digital_docling.txt for what that does to provenance.license.
#
# Declared before `llm` so that an untargeted `docker build` still builds the last
# stage, `api`.
# ---------------------------------------------------------------------------
FROM digital AS digital-docling

USER root
COPY requirements_digital_docling.txt ./
RUN pip install -r requirements_digital_docling.txt \
    && docling-tools models download layout tableformer -o /opt/docling-models \
    && chown -R atrium:atrium /opt/docling-models
ENV DOCLING_ARTIFACTS_PATH=/opt/docling-models
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

# EXPOSE tracks the DEFAULT port: it is image metadata and cannot read $PORT at
# runtime. Set PORT to move the listener, and publish with `-p <port>:<port>` to
# match. (issue #58)
EXPOSE 8000

# STOPSIGNAL is the default (SIGTERM) — declared explicitly so a future edit cannot
# change it silently; service/api.py's lifespan chains to uvicorn's own handler for it
# via serve_lifecycle (service/atrium_service.py, issue #55).
STOPSIGNAL SIGTERM

# PORT and HOST are read by service/api.py's __main__ block; PORT is also the port
# service/healthcheck.py probes, which is why setting it used to make the container
# permanently unhealthy — the probe moved and the listener did not. Declared here so
# `docker inspect` is self-documenting and so the probe still has a value if the code
# default ever drifts. (issue #58)
#
# GRACEFUL_SHUTDOWN_S carries the `--timeout-graceful-shutdown 20` that used to sit on
# the ENTRYPOINT line. It bounds uvicorn's wait for in-flight HTTP
# requests. Note llm-enrich's slow work happens INSIDE the request (one remote LLM call
# per line, each up to LLM_TIMEOUT), so a large document can legitimately outlive this
# budget and be cut short — raise this together with the deployment's grace period for
# such a workload (docs/k8s_deployment.md, "Known limits").
ENV PORT=8000 GRACEFUL_SHUTDOWN_S=20

# `python -m service.api`, NOT `python service/api.py`: a script launch puts
# sys.path[0] at /app/service with no package context, so `from .atrium_service import ...`
# raises "attempted relative import with no known parent package" before the app is
# built. `-m` keeps sys.path[0] at /app — byte for byte the environment the old
# `uvicorn service.api:app` entrypoint ran in, so every repo-root import still
# resolves. (issue #58)
ENTRYPOINT ["python", "-m", "service.api"]
CMD []
HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD ["python", "/app/service/healthcheck.py"]