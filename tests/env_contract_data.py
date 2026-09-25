"""Repo-local declarations for tests/test_env_contract.py (atrium-project#60).

Never vendored, never in para-drift, never in docs/templates/ruff.toml's [format]
exclude — unlike test_env_contract.py itself, this file's SHAPE is per-repo by
design: what one repo deliberately withholds from its ledger is not the same set as
what another does. See the canonical test's module docstring for the full rationale.
"""

from __future__ import annotations

# Read by shipped code but deliberately absent from .env.example, each with a reason.
NOT_PUBLISHED: dict[str, str] = {
    "HF_TOKEN": "read only by llm_run.py, the batch CLI entrypoint; service/api.py never imports it",
    "PROMPT_TEMPLATE": "read only by prompt_template.py via llm_run.py, the batch CLI; not reachable from the service entrypoint",
    "PROMPT_GEO_GUARDRAIL": "read only by prompt_template.py via llm_run.py, the batch CLI; not reachable from the service entrypoint",
    "PROMPT_VOCAB_GROUPING": "read only by prompt_template.py via llm_run.py, the batch CLI; not reachable from the service entrypoint",
    "DIGITAL_ENGINE": "read only by llm_client_shared.prepare_document_input, the batch clients' PDF/DOCX auto-convert (openrouter_client.py / ollama_client.py); service/api.py never converts documents",
    "DOCLING_ARTIFACTS_PATH": "read only by api_util/digital_docling.py (`--engine docling`), set by the Dockerfile's digital-docling stage; the api image carries no Docling",
}

# In .env.example but read by no Python in this repo — each with a reason.
CONSUMED_ELSEWHERE: dict[str, str] = {
    "ATRIUM_VERSION": "read only by docker-compose.yaml to pick the image tag; no Python here reads it",
    "HF_HOME": "read by huggingface_hub itself, set by the Dockerfile and docker-compose.yaml",
}

# service/README.md or .env.example cells whose value is prose rather than a literal
# the code-default resolver can compare against.
PROSE_DEFAULTS: dict[str, str] = {}
