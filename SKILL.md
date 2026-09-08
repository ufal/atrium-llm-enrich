---
name: atrium-llm-enrich
description: Extracts vocabulary-guided archaeological keywords from digitized document text using a server-configured LLM backend (OpenRouter remote API or local Ollama) - line-level (CSV/TEITOK) or whole-document (Markdown/plain text) records each get Czech + English key terms, a TEATER/AMCR thematic category constrained to the controlled vocabulary, and a confidence score. Use this skill for semantic enrichment of archival text after OCR and quality filtering, when NLP-statistical keywords are not enough and vocabulary-grounded semantic categories are needed.
---

# ATRIUM LLM Enrichment Skill 🗝️

This skill provides agent access to the **ATRIUM LLM Enrichment** service -
vocabulary-constrained keyword extraction over an LLM. It follows a
**server-client** design: a FastAPI server (in `service/`) drives the LLM
backends and the TEATER/AMCR vocabulary prompting, and a zero-dependency
client script (`scripts/atrium_keywords.py`) is the only thing the agent
calls directly.

## Operational Requirements ⚙️

- **Server**: a running instance is required. Default `http://localhost:8000`;
  override with `--base-url` or the `ATRIUM_LE_URL` environment variable.
- **Client dependencies**: none - `scripts/atrium_keywords.py` uses only the
  Python 3 standard library.
- **Server dependencies**: Docker (recommended, compose `api` profile) or a
  Python venv with `requirements_remote.txt` + `service/requirements.txt`
  (torch-free). A backend must be configured: `OPENROUTER_API_KEY` (+
  `OPENROUTER_MODEL` in `llm_config.txt`) for `openrouter`, or a reachable
  Ollama server (`OLLAMA_HOST`, `OLLAMA_MODEL`) for `ollama`.
- **First launch**: the TEATER/AMCR vocabulary auto-syncs from the AMCR
  OAI-PMH API when its cache file is missing - minutes, network-bound. Do
  **not** treat a slow first start as failure; `/health?deep=true` reports
  readiness.
- **Limits**: 10 MB per upload. **This is the slowest ATRIUM service** - a full
  document takes minutes; there is no per-request line cap, so split very large
  inputs yourself if a call is taking too long.
- **Readiness**: `GET /health` is liveness (stays 200 while draining); `GET /ready` is
  the orchestrator-facing readiness probe — 503 while warming up or shutting down.
  `GET /info` reports `ready: false` (rather than failing to start) when the configured
  backend is misconfigured (missing API key / model) — check `/info` before assuming
  the server is up.

## Backends & vocabulary 🗝️

| Backend      | Where it runs           | Requirements                            |
|--------------|--------------------------|------------------------------------------|
| `openrouter` | remote LLM-as-a-service | `OPENROUTER_API_KEY`, `OPENROUTER_MODEL` |
| `ollama`     | local Ollama server      | `OLLAMA_HOST` reachable, `OLLAMA_MODEL` |

**The backend and model are chosen server-side, once, at server startup** via the
`LLM_BACKEND` env var (`openrouter` default, or `ollama`) plus the matching
`OPENROUTER_MODEL`/`OLLAMA_MODEL` — there is no per-request backend override. Use
`--info` to see which backend/model the running server warmed up, and whether it is
`ready`. (`local` in-process transformers/vLLM is CLI-only, outside the API.)

Two extraction modes, selected by input file suffix (or, for raw stdin text, always
document-level):

| Mode            | Input suffix                | Granularity                                                      |
|-----------------|------------------------------|-------------------------------------------------------------------|
| line-level      | `.csv`, `.teitok.xml`        | one record per qualifying line                                    |
| document-level  | `.md`, `.txt`, or stdin text | one chat call over the whole document, one record per located passage |

Each record's `enrichment` block picks the single most relevant **TEATER/AMCR
category** from the controlled vocabulary (it cannot invent categories), extracts
Czech key terms found in the passage itself, translates them to English, and reports a
calibrated confidence score. Meta-text is categorized `Nerelevantní (meta-text)` with
no keywords.

## Workflows 🪄

### 1. Ensure the server is running

```bash
bash scripts/server.sh          # Docker Compose api profile (or local fallback)
bash scripts/server.sh --local  # force local uvicorn (no Docker)
```

Idempotent: exits immediately if GET /info already answers; waits for
first-run vocabulary sync.

### 2. Extract keywords

```bash
# CSV with text[,page_num,line_num] columns - line-level
python3 scripts/atrium_keywords.py small_data_samples/lines_sample.csv

# TEITOK XML page - line-level, full JSON envelope
python3 scripts/atrium_keywords.py page.teitok.xml --format json

# Markdown/plain-text document - document-level (one call, several passages)
python3 scripts/atrium_keywords.py notes.md

# Inline document text from stdin (no file needed) - document-level
printf 'V sondě S3 byly odkryty základy kostela.\n' | python3 scripts/atrium_keywords.py -

# Discover the configured backend/model, readiness, and limits
python3 scripts/atrium_keywords.py --info
```

### 3. ATRIUM Document JSON accretion (optional)

Accrete this tool's `enrichment` block onto an existing baseline record (accretion
contract, `docs/document_schema.md` in the hub repo; single-file only):

```bash
python3 scripts/atrium_keywords.py page.teitok.xml --document-json in.document.json \
    --document-json-out-file out.document.json
```

Every other tool's block (`pages`, `lines`, `entities`, `translations`, ...) passes
through unchanged. An invalid baseline is still accepted (rule 6), and the response
then also carries `document_json_schema_error` — the client prints this as a warning.

### 4. Interpret output

- `table` (default): `DOC, LOCATOR, CATEGORY, CONF, KEYWORDS_CS` rows (`LOCATOR` is
  `p<page>/l<line>` for line-level records, `p<page>:<locator>` for document-level ones)
  plus a one-line run summary (`mode`, backend, model, processed/filtered/error counts)
  on stderr.
- `csv`: adds `KEYWORDS_EN`, complete keyword lists for downstream tabular use.
- `json`: the full envelope — `doc_id`, `mode`, `backend`, `model`, `stats`, and
  `results[]` with each record's `enrichment` (`extracted_keywords_cs/en`,
  `teater_category`, `confidence_score`).

## Agent Guidelines 🤖

1. **Backend is server-configured, not client-chosen**: there is no `--backend` flag —
   check `--info` for which backend/model the running server warmed up, and whether it
   reports `ready: true`. If the wrong backend is configured, that is a server restart
   with a different `LLM_BACKEND`, not a client-side switch.
2. **Mode follows the file**: `.csv`/`.teitok.xml` get line-level records; `.md`/`.txt`
   and stdin text get one document-level call. Convert other formats (e.g. ALTO XML)
   with `api_util/xml_to_md.py` first if you need document-level extraction from them.
3. Confidence discipline: treat `confidence_score < 0.7` as tentative - surface
   the category with its score rather than asserting it; downstream filters
   commonly threshold on this field.
4. Prefer `--format json` (or `csv`) when the result feeds further
   processing; the table truncates keyword lists for readability.
5. For full request/response schemas, fetch `GET /openapi.json` from the
   running server (Swagger UI at `/docs`).
6. Exit code `2` (unreachable): start the server (`bash scripts/server.sh`)
   and retry once. Exit code `3` (server error, including HTTP 503 for a
   not-yet-ready or misconfigured backend): the client already retried
   502/503/504 three times - check `GET /health?deep=true` and `--info`'s
   `ready` field, and server logs; do not loop.
7. **Budget**: this is the slowest ATRIUM service (minutes per document) and has no
   built-in per-request line cap - run the quality filter (atrium-alto-postprocess)
   first so only meaningful lines reach the LLM, split very large documents yourself,
   and tell the user what you did.
8. Do not bypass the API by importing the LLM client code directly - the
   server is the supported entry point and enforces the vocabulary contract.

## Acknowledgements & Citations 🙏

The models and dataset are developed within the [ATRIUM](https://atrium-research.eu/)
project at ÚFAL, Charles University, with data hosted on
[LINDAT/CLARIAH-CZ](https://lindat.cz). If you use this service for research, cite the
repository's `CITATION.cff` and the LINDAT dataset record
(http://hdl.handle.net/20.500.12800/1-6184).
