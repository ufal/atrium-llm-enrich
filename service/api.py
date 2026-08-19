"""service/api.py — FastAPI surface for atrium-llm-enrich (strategy §4.2).

Brings llm-enrich into API parity with the rest of the ATRIUM pipeline. It wraps the
existing **remote / lightweight-local** enrichment engine (``llm_client_shared`` +
``openrouter_client`` / ``ollama_client``) — deliberately the torch-free path, so the
service stays in the no-model fast lane and never needs the GPU stack.

Backend is selected with ``LLM_BACKEND`` (``openrouter`` default, or ``ollama``); the
engine is warmed once on startup. A misconfigured backend (missing API key / model) does
**not** crash the app: ``/info`` and ``/health`` stay up and report ``ready: false`` while
the extraction endpoints answer 503 until configured.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, File, HTTPException, UploadFile

from atrium_document import FILE_SUFFIX, canonical_doc_id
from atrium_paradata import ParadataLogger

# Shared ATRIUM meta-contract helpers (§4). Byte-identical across every service,
# enforced by para-drift.reusable.yml.
from .atrium_service import (
    add_cors,
    attach_health,
    build_info,
    read_tool_version,
    resolve_max_upload_mb,
)

logger = logging.getLogger(__name__)

# Canonical upload limit (§4.5).
MAX_UPLOAD_MB = resolve_max_upload_mb(10)
MAX_UPLOAD_BYTES = int(MAX_UPLOAD_MB * 1024 * 1024)

# Reserved output-token budget subtracted from the context window when truncating the
# vocabulary prompt. Mirrors {openrouter,ollama}_client: MAX_NEW_TOKENS (2048) + 512,
# which those modules keep in sync with llm_utils by hand.
_CONTEXT_RESERVED = 2048 + 512

_LINE_SUFFIXES = (".csv", ".teitok.xml")
_DOC_SUFFIXES = (".md", ".txt")

# Warmed engine state (or an "error" key when the backend is unavailable).
_engine: Dict[str, Any] = {}


def _load_engine() -> Dict[str, Any]:
    """Build the enrichment engine for the configured backend (blocking).

    Replicates the setup sequence of ``openrouter_client.main`` / ``ollama_client.main``:
    load config + vocabulary, build the archaeological schema and system prompt, and bind
    a ``chat_fn`` to the chosen remote/local backend. Heavy-ish imports are kept local so
    importing this module (for contract tests) never requires ``requests``/vocab data.
    """
    import os

    import requests

    from llm_client_shared import (
        build_document_schema,
        build_document_system_prompt,
        build_schema,
        build_system_prompt,
        excluded_prompt_themes,
        load_config,
    )
    from vocab_manager import VocabularyManager

    backend = os.getenv("LLM_BACKEND", "openrouter").lower()
    config_path = os.getenv("LLM_CONFIG", "llm_config.txt")
    config = load_config(config_path) if Path(config_path).exists() else {}

    vocab_path = os.getenv("VOCAB_PATH") or config.get(
        "VOCAB_PATH", "data_samples/vocab/union_nested.json"
    )
    context_window = int(os.getenv("LLM_CONTEXT_WINDOW", config.get("CONTEXT_WINDOW", "32000")))
    max_retries = int(os.getenv("LLM_MAX_RETRIES", "3"))
    timeout = int(os.getenv("LLM_TIMEOUT", "300"))
    max_input_tokens = context_window - _CONTEXT_RESERVED

    filter_params = {
        "include_non_text": config.get("INCLUDE_NON_TEXT", "true").lower() == "true",
        "min_char_count": int(config.get("MIN_CHAR_COUNT", "3")),
        "min_char_non_text": int(config.get("MIN_CHAR_NON_TEXT", "8")),
        "min_alpha_ratio_non_text": float(config.get("MIN_ALPHA_RATIO_NON_TEXT", "0.40")),
    }

    vocab_mgr = VocabularyManager(vocab_path=vocab_path)
    # auto_sync=False: never harvest inside a pipeline run. Besides the
    # multi-minute OAI-PMH round trip, the sync path can no longer build a
    # usable vocabulary — fetch_amcr_vocab() emits bare {"cs", "en"} pairs
    # with no "source"/"scheme", so assign_theme() drops every term into
    # "Other", which excluded_prompt_themes() withholds from the prompt. The
    # result is an enum holding only "Nerelevantní (meta-text)", which then
    # rejects every correct answer the model gives as a validation error.
    # A missing vocabulary is a configuration fault: say so and stop.
    vocab_data = vocab_mgr.load(auto_sync=False)
    # Which themes reach the model is a taxonomy_config decision (in_prompt), not a
    # literal in the prompt builder — see excluded_prompt_themes().
    excluded_themes = excluded_prompt_themes(vocab_mgr)
    line_prompt, line_terms = build_system_prompt(
        vocab_data, max_tokens=max_input_tokens, excluded_themes=excluded_themes
    )
    doc_prompt, doc_terms = build_document_system_prompt(
        vocab_data, max_tokens=max_input_tokens, excluded_themes=excluded_themes
    )
    line_model = build_schema(line_terms)
    doc_model = build_document_schema(doc_terms)
    session = requests.Session()

    if backend == "openrouter":
        from openrouter_client import _build_headers, make_chat_fn

        api_key = os.getenv("OPENROUTER_API_KEY") or config.get("OPENROUTER_API_KEY")
        model = os.getenv("OPENROUTER_MODEL") or config.get("OPENROUTER_MODEL")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        if not model:
            raise RuntimeError("OPENROUTER_MODEL is not set")
        headers = _build_headers(
            api_key,
            os.getenv("OPENROUTER_SITE_URL"),
            os.getenv("OPENROUTER_APP_NAME", "atrium-llm-enrich"),
        )
        line_chat_fn = make_chat_fn(
            session, headers, model, line_model.model_json_schema(), max_retries, timeout, None
        )
        doc_chat_fn = make_chat_fn(
            session, headers, model, doc_model.model_json_schema(), max_retries, timeout, None
        )
        model_id = model
    elif backend == "ollama":
        from ollama_client import DEFAULT_OLLAMA_HOST, make_chat_fn

        host = os.getenv("OLLAMA_HOST") or config.get("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
        model = os.getenv("OLLAMA_MODEL") or config.get("OLLAMA_MODEL")
        if not model:
            raise RuntimeError("OLLAMA_MODEL is not set")
        line_chat_fn = make_chat_fn(
            session, host, model, line_model.model_json_schema(), max_retries, timeout
        )
        doc_chat_fn = make_chat_fn(
            session, host, model, doc_model.model_json_schema(), max_retries, timeout
        )
        model_id = f"{model}@{host}"
    else:
        raise RuntimeError(f"Unknown LLM_BACKEND '{backend}' (expected 'openrouter' or 'ollama')")

    return {
        "backend": backend,
        "model": model_id,
        "line_prompt": line_prompt,
        "line_model": line_model,
        "line_chat_fn": line_chat_fn,
        "doc_prompt": doc_prompt,
        "doc_model": doc_model,
        "doc_chat_fn": doc_chat_fn,
        "filter_params": filter_params,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the backend once; a misconfigured backend is recorded, not fatal.
    loop = asyncio.get_event_loop()
    try:
        _engine.update(await loop.run_in_executor(None, _load_engine))
        logger.info("llm-enrich engine ready (backend=%s)", _engine.get("backend"))
    except Exception as exc:
        _engine.clear()
        _engine["error"] = str(exc)
        logger.warning("llm-enrich engine warmup failed: %s", exc)
    yield
    _engine.clear()


app = FastAPI(
    title="ATRIUM llm-enrich API",
    version=read_tool_version(Path(__file__).resolve().parent),
    description="LLM-based archaeological keyword extraction over text lines / documents.",
    lifespan=lifespan,
)

# CORS — standard §4.5 configuration (ALLOWED_ORIGINS CSV, default "*").
add_cors(app, methods=["GET", "POST"])


def _deep_health() -> str | None:
    """Deep readiness (§4.1): the LLM backend warmed up and a chat_fn is bound."""
    if _engine.get("error"):
        return f"backend not configured: {_engine['error']}"
    if not _engine.get("line_chat_fn"):
        return "engine not initialized"
    return None


attach_health(app, deep_check=_deep_health)


def _require_engine() -> Dict[str, Any]:
    """Return the warmed engine, or 503 if the backend is not ready (§4.4 → client retries)."""
    if _engine.get("error"):
        raise HTTPException(503, f"LLM backend not ready: {_engine['error']}") from None
    if not _engine.get("line_chat_fn"):
        raise HTTPException(503, "LLM backend not initialized.") from None
    return _engine


def _doc_id(filename: str) -> str:
    """The uploaded document's identity, derived the one canonical way.

    Delegates to ``atrium_document.canonical_doc_id()`` (atrium-project#10, D3): this was a
    THIRD independent suffix-stripper in this repo, alongside
    ``api_util/teitok_read.doc_id_from_path`` and the two batch clients' ``Path.stem``. The
    accretion contract is keyed on ``doc_id``, so a service that re-keys the record it was
    handed silently orphans the caller's baseline — which is exactly what D1/D2 did in the
    batch clients and in alto's ``/process``.
    """
    return canonical_doc_id(filename)


def _run_extraction(
    tmp_path: str,
    filename: str,
    engine: Dict[str, Any],
    doc_id: str,
    document_record_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Blocking enrichment call, dispatched by file extension (line vs document level).

    When ``document_record_dir`` is given, also folds the results into this document's
    paired ATRIUM record (accretion model, docs/document_schema.md / issue #13): a baseline
    at ``<document_record_dir>/<doc_id>.document.json``, if the caller placed one there, is
    read and written back with only llm-enrich's ``enrichment`` block updated — every other
    tool's block passes through untouched (rule 2). With no baseline present the record holds
    just llm-enrich's own part (rule 3), mirroring ``llm_run.py``/``openrouter_client.py``/
    ``ollama_client.py``'s ``write_document_record`` call. No results → nothing is written,
    same as those batch entry points.

    The Layer D schema gate (atrium-project#10, D4) lives in ``write_document_record()`` —
    the repo's single write chokepoint — so an invalid record is never written here either.
    What this function adds is the gate on the record it hands BACK: see below.
    """
    from llm_client_shared import (
        run_document_level,
        run_line_level,
        schema_gate,
        write_document_record,
    )

    name = filename.lower()
    path = Path(tmp_path)
    if name.endswith(_LINE_SUFFIXES):
        records, stats = run_line_level(
            path,
            engine["line_chat_fn"],
            engine["line_prompt"],
            engine["line_model"],
            **engine["filter_params"],
        )
        mode = "line"
    else:  # validated to be a _DOC_SUFFIXES file by the caller
        records, stats = run_document_level(
            path, engine["doc_chat_fn"], engine["doc_prompt"], engine["doc_model"]
        )
        mode = "document"

    result: Dict[str, Any] = {"mode": mode, "results": records, "stats": stats}

    if document_record_dir is not None and records:
        from atrium_document import load_document

        with ParadataLogger(
            program="llm-enrich-api",
            config={"mode": mode, "backend": engine["backend"], "model": engine["model"]},
            paradata_dir=str(Path(document_record_dir) / "paradata"),
            output_types=["json"],
        ) as para_logger:
            try:
                record_path = write_document_record(
                    doc_id,
                    records,
                    document_record_dir,
                    run_id=para_logger.run_id,
                    paradata_ref=os.path.join(
                        para_logger.paradata_dir,
                        f"{para_logger.run_id}_{para_logger.program}.json",
                    ),
                    used_markdown_input=(mode == "document"),
                    license_detail=para_logger.get_license_block(),
                )
            except RuntimeError as exc:
                # The Layer D refusal (D4). Translated here rather than left to
                # _extract_from_path's blanket `RuntimeError -> 502 LLM backend error`,
                # which would blame the upstream provider for a record WE built wrong —
                # and 502 invites a retry that would fail identically. A record llm-enrich
                # cannot emit is a defect on this side, so it is a 500, named as such.
                raise HTTPException(
                    500, f"Document record rejected by its own schema: {exc}"
                ) from exc
            if record_path is not None:
                para_logger.log_success("json")
                para_logger.log_document_success()

        if record_path is not None:
            record = load_document(str(record_path))
            # Layer D on the way OUT (atrium-project#10, D4). write_document_record() has
            # already refused to emit a record whose invalidity was ours, so anything caught
            # here is a record it deliberately let through because the caller's own uploaded
            # baseline did not validate. Re-raising would contradict that decision (and 500
            # on somebody else's bad data), and returning it in silence is what D4 is about —
            # so the response says so, in a field an automated caller can test instead of
            # grepping the service log.
            schema_error = schema_gate(record, f"{doc_id}{FILE_SUFFIX}")
            if schema_error:
                logger.warning(
                    "returned document record for %s does not validate against the ATRIUM "
                    "document schema: %s",
                    doc_id,
                    schema_error,
                )
                result["document_json_schema_error"] = schema_error
            result["document_json"] = record

    return result


def _envelope(engine: Dict[str, Any], doc_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "service": "atrium-llm-enrich",
        "doc_id": doc_id,
        "backend": engine["backend"],
        "model": engine["model"],
        **payload,
    }


async def _extract_from_path(
    tmp_path: str,
    filename: str,
    engine: Dict[str, Any],
    doc_id: str,
    document_record_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    loop = asyncio.get_event_loop()
    try:
        return await loop.run_in_executor(
            None, _run_extraction, tmp_path, filename, engine, doc_id, document_record_dir
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    except RuntimeError as exc:
        # chat_fn exhausted its retries against the upstream LLM — retryable (§4.4).
        raise HTTPException(502, f"LLM backend error: {exc}") from exc


@app.get("/info")
async def info() -> Dict[str, Any]:
    """Service identity and capabilities (§4.1)."""
    return build_info(
        app,
        service="atrium-llm-enrich",
        limits={"max_upload_mb": MAX_UPLOAD_MB},
        backend=_engine.get("backend"),
        model=_engine.get("model"),
        ready=not _engine.get("error") and bool(_engine.get("line_chat_fn")),
        supported_inputs=[*_LINE_SUFFIXES, *_DOC_SUFFIXES],
        languages=["cs", "en"],
    )


@app.post("/extract_keywords")
async def extract_keywords(
    file: UploadFile = File(...),  # noqa: B008
    document_json: UploadFile = File(  # noqa: B008
        None,
        description=(
            "Optional baseline ATRIUM Document JSON (accretion model, docs/document_schema.md "
            "/ issue #13). When given, the response's `document_json` carries the record back "
            "with only llm-enrich's `enrichment` block updated — every other tool's block "
            "(pages, lines, entities, translations, ...) passes through untouched. A baseline "
            "that does not validate against atrium_document.schema.json is still accepted "
            "(rule 6), but the response then also carries `document_json_schema_error`."
        ),
    ),
):
    """Extract archaeological keywords from an uploaded document (§4.2).

    ``.csv`` / ``*.teitok.xml`` → line-level (one record per qualifying line);
    ``.md`` / ``.txt`` → document-level (one record set per document). Each record's
    ``enrichment`` carries ``extracted_keywords_cs`` / ``extracted_keywords_en``.
    """
    engine = _require_engine()
    if not file.filename:
        raise HTTPException(422, "Filename is missing from the upload.") from None
    name = file.filename.lower()
    if not name.endswith(_LINE_SUFFIXES + _DOC_SUFFIXES):
        raise HTTPException(
            422, f"Unsupported file type. Accepted: {', '.join(_LINE_SUFFIXES + _DOC_SUFFIXES)}."
        ) from None

    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"File too large. Maximum size is {MAX_UPLOAD_MB} MB.") from None

    doc_id = _doc_id(file.filename)
    suffix = ".teitok.xml" if name.endswith(".teitok.xml") else Path(name).suffix

    with tempfile.TemporaryDirectory() as tmp_dir:
        work_dir = Path(tmp_dir)
        tmp_path = work_dir / f"input{suffix}"
        tmp_path.write_bytes(data)

        document_record_dir: Optional[Path] = None
        if document_json is not None:
            baseline_bytes = await document_json.read()
            (work_dir / f"{doc_id}{FILE_SUFFIX}").write_bytes(baseline_bytes)
            document_record_dir = work_dir

        result = await _extract_from_path(
            str(tmp_path), file.filename, engine, doc_id, document_record_dir
        )

    return _envelope(engine, doc_id, result)


@app.post("/extract_keywords_text")
async def extract_keywords_text(
    text: str = Body(
        ..., description="Raw text for document-level archaeological keyword extraction."
    ),
    document_json: dict = Body(
        None,
        description="Optional baseline ATRIUM Document JSON. When given, the response's `document_json` carries the record back with only llm-enrich's `enrichment` block updated. An invalid baseline is still accepted, and the response then also carries `document_json_schema_error`.",
    ),
):
    """Extract archaeological keywords from inline text (§4.2)."""
    engine = _require_engine()

    # Assign a dummy doc_id for inline text (or generate a UUID if preferred)
    doc_id = "inline_text"

    with tempfile.TemporaryDirectory() as tmp_dir:
        work_dir = Path(tmp_dir)
        tmp_path = work_dir / "input.txt"
        tmp_path.write_text(text, encoding="utf-8")

        document_record_dir: Optional[Path] = None
        if document_json is not None:
            baseline_path = work_dir / f"{doc_id}{FILE_SUFFIX}"
            baseline_path.write_text(json.dumps(document_json), encoding="utf-8")
            document_record_dir = work_dir

        result = await _extract_from_path(
            str(tmp_path), "input.txt", engine, doc_id, document_record_dir
        )

    return _envelope(engine, doc_id, result)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("service.api:app", host="0.0.0.0", port=8000)
