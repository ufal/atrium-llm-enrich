"""
llm_client_shared.py — Lightweight shared front-end for the remote and
local-lightweight LLM backends (openrouter_client.py, ollama_client.py).

Why this module exists (rather than importing llm_utils.py / llm_run.py):
  llm_utils.py unconditionally imports torch and sets PYTORCH_CUDA_ALLOC_CONF
  as a side effect of import (see its module docstring); llm_run.py imports
  llm_utils.py. Remote/local-lightweight users install requirements_remote.txt
  and should never need the GPU stack. This module therefore DUPLICATES the
  small, pure-Python pieces those two files provide — config loading,
  CSV/TEITOK row reading, line-quality filtering, context-window building,
  the archaeological system prompt + Pydantic schema, and lenient JSON
  validation — instead of importing them.

  This is the deliberate "some duplication to reconcile later via a shared
  package" tradeoff called out in README.md, kept to ONE place so
  openrouter_client.py and ollama_client.py don't duplicate it a second time
  between themselves.

Kept in sync BY HAND with llm_utils.py / llm_run.py. If you change the
quality filter, the context-window builder, or the archaeological system
prompt over there, mirror the change here.
"""

import csv
import enum
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field, ValidationError

from api_util import teitok_read
from api_util.teitok_read import doc_id_from_path  # noqa: F401  (re-exported for clients)

# ---------------------------------------------------------------------------
# 1. Config loader — duplicated from llm_utils.load_config
# ---------------------------------------------------------------------------


def load_config(config_path: str = "llm_config.txt") -> Dict[str, str]:
    """Parse a KEY=VALUE config file, ignoring blank lines and # comments."""
    config: Dict[str, str] = {}
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}") from None
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                config[key.strip()] = value.strip()
    return config


# ---------------------------------------------------------------------------
# 2. Token-count approximation — no tokenizer/torch dependency
# ---------------------------------------------------------------------------

# Rough, tokenizer-free estimate. Czech/English archival text averages
# roughly 4 characters per token across the model families this repo has
# targeted so far (Qwen/Gemma/Llama tokenizers). Good enough for vocabulary-
# truncation decisions; NOT precise enough for exact context-limit or
# billing arithmetic — callers that need that should use the provider's own
# token-counting endpoint if one exists.
_CHARS_PER_TOKEN_ESTIMATE = 4


def approx_token_count(text: str) -> int:
    """Character-based token estimate. See _CHARS_PER_TOKEN_ESTIMATE."""
    return max(1, len(text) // _CHARS_PER_TOKEN_ESTIMATE)


# Chat callable both clients implement: takes the [system, user] message
# list, returns the raw text of the model's reply (expected to be JSON, but
# validate_llm_output() tolerates near-miss JSON — see there).
ChatFn = Callable[[List[Dict[str, str]]], str]


# ---------------------------------------------------------------------------
# 3. Line-quality filter — duplicated from llm_utils._should_process_line
# ---------------------------------------------------------------------------

_ALWAYS_SKIP_CATEG = {"Empty", "Trash"}
_NOISE_CATEG = {"Empty", "Trash", "Non-text"}


def should_process_line(
    text: str,
    categ: str,
    quality_score: float,
    include_non_text: bool,
    min_char_count: int,
    min_char_non_text: int,
    min_alpha_ratio_non_text: float,
) -> Tuple[bool, str]:
    if quality_score < 0.40:
        categ = "Trash"
    elif quality_score < 0.70 and categ != "Trash":
        categ = "Noisy"

    if not text:
        return False, "empty text"

    if categ in _ALWAYS_SKIP_CATEG:
        return False, f"categ={categ!r} (quality={quality_score})"

    if categ == "Non-text":
        if not include_non_text:
            return False, "Non-text excluded by config"
        char_count = len(text)
        if char_count < min_char_non_text:
            return False, f"Non-text too short ({char_count} < {min_char_non_text} chars)"
        alpha_count = sum(c.isalpha() for c in text)
        alpha_ratio = alpha_count / char_count if char_count else 0.0
        if alpha_ratio < min_alpha_ratio_non_text:
            return False, f"Non-text alpha ratio too low ({alpha_ratio:.2f})"
        return True, ""

    if not categ:
        if len(text) < min_char_count:
            return False, f"text too short ({len(text)} < {min_char_count} chars) [unknown categ]"
        return True, ""

    if len(text) < min_char_count:
        return False, f"text too short ({len(text)} < {min_char_count} chars)"

    return True, ""


# ---------------------------------------------------------------------------
# 4. Row reading — duplicated from llm_utils.read_input_rows
# ---------------------------------------------------------------------------


def read_input_rows(input_path: Path) -> List[dict]:
    """Reads rows from a CSV or synthesizes lines from a TEITOK XML document."""
    if input_path.name.lower().endswith(".teitok.xml"):
        return [
            {
                "text": r["text"],
                "page_num": str(r.get("page_num", "")),
                "line_num": str(r.get("line_num", "")),
                "categ": "",  # Falls back to plain text handling
                "quality_score": 0.0,
            }
            for r in teitok_read.read_teitok_rows(str(input_path))
        ]
    with open(input_path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# 5. Context-window builder — duplicated from llm_utils.get_context_window
# ---------------------------------------------------------------------------


def get_context_window(rows: List[dict], center_idx: int, window: int = 2) -> str:
    """Build a text snippet around ``rows[center_idx]`` for the LLM user prompt.
    See llm_utils.get_context_window — identical logic, duplicated here."""
    center_row = rows[center_idx]
    center_page = center_row.get("page_num", center_row.get("page", None))
    start = max(0, center_idx - window)
    end = min(len(rows), center_idx + window + 1)

    parts: List[str] = []

    if center_idx > window + 2:
        parts.append("--- GLOBAL DOCUMENT HEADER ---")
        added = 0
        for row in rows:
            if row.get("categ", "").strip() not in _NOISE_CATEG:
                pg = row.get("page_num", row.get("page", 0))
                ln = row.get("line_num", row.get("line", 0))
                parts.append(f"    [P{pg} L{ln}] {row.get('text', '').strip()}")
                added += 1
                if added >= 2:
                    break

    current_section = "Unknown Section"
    for i in range(center_idx - 1, -1, -1):
        if rows[i].get("categ", "").strip() in {"Header", "Heading"}:
            current_section = rows[i].get("text", "").strip()
            break

    parts.append(f"--- CURRENT SECTION: {current_section} ---")
    parts.append("--- LOCAL CONTEXT WINDOW ---")

    for i in range(start, end):
        row = rows[i]
        row_page = row.get("page_num", row.get("page", None))
        categ = row.get("categ", "").strip()

        if row_page != center_page and i != center_idx:
            continue
        if i != center_idx and categ in _NOISE_CATEG:
            continue

        text = row.get("text", "").strip()
        pg = row_page
        ln = row.get("line_num", row.get("line", 0))

        if i == center_idx:
            parts.append(f"<target_line> >>> [P{pg} L{ln}] {text} </target_line>")
        else:
            parts.append(f"    [P{pg} L{ln}] {text}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 6. Lenient JSON validation — duplicated from llm_utils.validate_llm_output
# ---------------------------------------------------------------------------


def validate_llm_output(
    result_json: str, EnrichmentModel: type, file_id: str, page_num: int, line_num: int
) -> dict:
    """Validate and sanitize LLM JSON output against a Pydantic model."""
    try:
        semantic_data = EnrichmentModel.model_validate_json(result_json)
    except ValidationError:
        try:
            raw_dict = json.loads(result_json, strict=False)
            if "confidence_score" in raw_dict:
                try:
                    val = float(raw_dict["confidence_score"])
                    raw_dict["confidence_score"] = min(1.0, max(0.0, val))
                except (ValueError, TypeError):
                    pass
            semantic_data = EnrichmentModel.model_validate(raw_dict)
        except (json.JSONDecodeError, ValidationError) as exc:
            raise ValueError(
                f"[{file_id}] Persistent validation error P{page_num} L{line_num}: {exc}"
            ) from exc

    dump_data = semantic_data.model_dump()

    if hasattr(semantic_data, "category_name"):
        dump_data["teater_category"] = semantic_data.category_name()
    else:
        dump_data["teater_category"] = dump_data.get("teater_category", "")

    if dump_data.get("teater_category") == "Nerelevantní (meta-text)":
        dump_data["extracted_keywords_cs"] = []
        dump_data["extracted_keywords_en"] = []

    return dump_data


# ---------------------------------------------------------------------------
# 7. Schema + system prompt — duplicated/adapted from llm_run.py
#    (build_schema, build_system_prompt); token counting swapped from
#    tokenizer-based count_tokens() to approx_token_count() above.
# ---------------------------------------------------------------------------

_EXAMPLES_FOOTER = (
    "\nEXAMPLES:\n\n"
    'Input line: "Výzkum odhalil základy gotického kostela ze 14. '
    'století."\n'
    "Correct output:\n"
    "{\n"
    '  "extracted_keywords_cs": ["základy", "gotický kostel"],\n'
    '  "extracted_keywords_en": ["foundations", "Gothic church"],\n'
    '  "teater_category": "kostel",\n'
    '  "confidence_score": 0.92\n'
    "}\n\n"
    'Input line: "Praha, dne 6. října 1956, Dr. Solle"\n'
    "Correct output:\n"
    "{\n"
    '  "extracted_keywords_cs": [],\n'
    '  "extracted_keywords_en": [],\n'
    '  "teater_category": "Nerelevantní (meta-text)",\n'
    '  "confidence_score": 1.0\n'
    "}\n"
)

_SYSTEM_HEADER = (
    "You are an expert archaeological data extractor. "
    "Analyze the MARKED LINE enclosed in <target_line> ... </target_line> "
    "within its surrounding document context.\n"
    "1. Extract ONLY archaeological entities, features, periods, or materials "
    "from the marked line. "
    "Do NOT extract names of researchers, dates, conjunctions, or "
    "administrative words.\n"
    "2. Select the SINGLE most relevant category from the thematic vocabulary "
    "list below.\n"
    "CRITICAL: If the marked line is purely administrative, a table of contents, "
    "a generic heading (e.g. page numbers, titles, author names, 'Práce:', "
    "'Obsah:', literature references) or lacks direct archaeological context, "
    "you MUST select 'Nerelevantní (meta-text)'.\n"
    "NEVER select a country name, language name, or geographic region name "
    "as the teater_category for any line — including administrative lines. "
    "For any line that lacks direct archaeological significance, "
    "you MUST use 'Nerelevantní (meta-text)'.\n"
    "When extracting keywords, normalize obvious OCR artifacts and typos to "
    "their correct Czech forms. "
    "Do NOT include garbled tokens or split words as keywords. "
    "Prefer the normalized phrase over the raw OCR text.\n"
    "You MUST use the exact Czech term as written in the vocabulary.\n"
    "You MUST respond ONLY with a valid JSON object matching the requested "
    "schema.\n\n"
    "THEMATIC VOCABULARY:\n"
)


def build_schema(term_names: List[str]) -> type:
    if not term_names:
        raise ValueError("term_names is empty — vocabulary failed to load or was fully truncated.")

    TermEnum = enum.Enum("TermEnum", {f"term_{i}": name for i, name in enumerate(term_names)})

    class ConstrainedEnrichment(BaseModel):
        extracted_keywords_cs: List[str] = Field(
            ...,
            description=(
                "Key Czech archaeological terms, methods, or objects found ONLY in "
                "the text marked with (>>>). "
                "DO NOT copy terms from the THEMATIC VOCABULARY list. "
                "If no relevant archaeological terms appear in the target line, "
                "return []. "
                "If teater_category is 'Nerelevantní (meta-text)', MUST be []. "
                "Do not extract names of researchers or administrative words. "
                "Prefer normalised multi-word phrases over isolated single words."
            ),
        )
        extracted_keywords_en: List[str] = Field(
            ...,
            description=(
                "Accurate English translations of extracted_keywords_cs. "
                "Do not copy Czech words unchanged."
            ),
        )
        teater_category: TermEnum = Field(
            ...,
            description="The single most relevant category from the thematic vocabulary.",
        )
        confidence_score: float = Field(
            ...,
            ge=0.0,
            le=1.0,
            description=(
                "Confidence that the selected teater_category is correct. "
                "1.0 — unambiguous match, no interpretation required. "
                "0.7–0.9 — reasonable but non-obvious match. "
                "0.5–0.7 — multiple categories could apply. "
                "< 0.5 — forced guess. "
                "Do NOT output 1.0 uniformly — this field is used for filtering."
            ),
        )

        def category_name(self) -> str:
            return self.teater_category.value

    return ConstrainedEnrichment


def _collect_vocab_terms(vocab_data: dict) -> List[dict]:
    """Flatten ``vocab_data`` into a list of ``{theme, cs, en}`` term dicts,
    with the fixed 'Nerelevantní (meta-text)' administrative term prepended.

    Shared by build_system_prompt() and build_document_system_prompt() —
    the two callers differ only in header/footer text, not in how
    vocabulary terms are gathered from the nested theme/keyword structure."""
    raw_terms: List[dict] = [
        {
            "theme": "Administrative / Meta",
            "cs": "Nerelevantní (meta-text)",
            "en": "Irrelevant / Meta-text",
        }
    ]
    for theme, data in vocab_data.items():
        if theme.lower() == "other":
            continue
        if isinstance(data, dict):
            if "keywords" in data and isinstance(data["keywords"], dict):
                cs_list = data["keywords"].get("cs", [])
                en_list = data["keywords"].get("en", [])
                for i, cs_key in enumerate(cs_list):
                    en = en_list[i] if i < len(en_list) else cs_key
                    raw_terms.append({"theme": theme, "cs": cs_key, "en": en})
            else:
                for cs_key, pair in data.items():
                    en = pair.get("en", cs_key) if isinstance(pair, dict) else cs_key
                    raw_terms.append({"theme": theme, "cs": cs_key, "en": en})
    return raw_terms


def _render_vocab_prompt(
    header: str, term_list: List[dict], other_cap: int = 15, footer: str = ""
) -> str:
    """Render ``term_list`` under ``header``, grouped by theme, with an
    'Other (Misc)' tail capped at ``other_cap`` terms, then ``footer``.

    The prompt-rendering half shared by build_system_prompt() and
    build_document_system_prompt() (single-line uses ``_EXAMPLES_FOOTER``,
    whole-document uses no footer — see their thin wrappers below)."""
    themes: Dict[str, List[str]] = {}
    other_terms: List[dict] = []
    for t in term_list:
        if t["theme"] == "Other":
            other_terms.append(t)
        else:
            themes.setdefault(t["theme"], []).append(f"{t['cs']} ({t['en']})")

    prompt = header
    for theme_name, lines in themes.items():
        prompt += f"\n--- {theme_name} ---\n"
        prompt += "\n".join(f"- {line}" for line in lines) + "\n"
    if other_terms:
        prompt += "\n--- Other (Misc) ---\n"
        prompt += "\n".join(f"- {t['cs']} ({t['en']})" for t in other_terms[:other_cap]) + "\n"
    prompt += footer
    return prompt


def _fit_vocab_prompt(
    header: str,
    raw_terms: List[dict],
    max_tokens: int,
    skip_truncation: bool = False,
    footer: str = "",
    verbose: bool = False,
) -> Tuple[str, List[str]]:
    """Render ``raw_terms`` under ``header``/``footer``, binary-searching for
    the largest prefix that fits ``max_tokens`` if the full vocabulary
    doesn't. ``verbose=True`` prints the ``[vocab]``/``[WARN]`` progress
    lines (matches build_system_prompt()'s prior behaviour); the
    whole-document prompt renders silently (matches
    build_document_system_prompt()'s prior behaviour) — callers below
    preserve each function's original verbosity via this flag."""

    def _render(term_list: List[dict]) -> str:
        return _render_vocab_prompt(header, term_list, footer=footer)

    full_prompt = _render(raw_terms)
    token_count = approx_token_count(full_prompt)

    if verbose:
        print(f"[vocab] {len(raw_terms)} terms, ~{token_count} tokens total (char-based estimate)")

    if skip_truncation:
        if verbose:
            print(f"[vocab] Injecting full vocabulary (~{token_count} tokens, no truncation).")
        return full_prompt, [t["cs"] for t in raw_terms]

    if token_count <= max_tokens:
        if verbose:
            print("[vocab] Full vocabulary fits within (approximate) token budget.")
        return full_prompt, [t["cs"] for t in raw_terms]

    if verbose:
        print(
            f"[WARN] Vocabulary (~{token_count} tokens) exceeds budget "
            f"({max_tokens}). Binary-searching for largest fitting prefix…"
        )

    lo, hi = 0, len(raw_terms)
    while lo < hi - 1:
        mid = (lo + hi) // 2
        if approx_token_count(_render(raw_terms[:mid])) <= max_tokens:
            lo = mid
        else:
            hi = mid

    surviving_terms = raw_terms[:lo]
    surviving_prompt = _render(surviving_terms)
    surviving_cs = [t["cs"] for t in surviving_terms]

    if verbose:
        print(
            f"[vocab] Truncated to {len(surviving_cs)} terms "
            f"(~{approx_token_count(surviving_prompt)} tokens)."
        )
    return surviving_prompt, surviving_cs


def build_system_prompt(
    vocab_data: dict,
    max_tokens: int,
    skip_truncation: bool = False,
) -> Tuple[str, List[str]]:
    """Same vocabulary-truncation strategy as llm_run.build_system_prompt, but
    driven by approx_token_count() instead of a tokenizer — no HF/torch
    dependency, at the cost of an approximate (not exact) token budget."""
    raw_terms = _collect_vocab_terms(vocab_data)
    return _fit_vocab_prompt(
        _SYSTEM_HEADER,
        raw_terms,
        max_tokens,
        skip_truncation,
        footer=_EXAMPLES_FOOTER,
        verbose=True,
    )


_DOC_SYSTEM_HEADER = (
    "You are an expert archaeological data extractor. "
    "You will be given a WHOLE DOCUMENT (rendered from a digitized archival "
    "record). Scan it and extract EVERY passage with direct archaeological "
    "significance — sites, finds, methods, periods, materials.\n"
    "For EACH such passage, return one item with:\n"
    "  - locator: a short verbatim snippet (max 8 words) copied EXACTLY from "
    "the document, unique enough to locate the passage (prefer including the "
    "'## Page N' heading text nearest above it if the document has page "
    "headings).\n"
    "  - page: the page number of that passage, read from the nearest "
    "'<!-- PAGE_BREAK: pg_N -->' or '## Page N' marker ABOVE it (just the number/"
    "label, e.g. 3); null if the document has no page markers.\n"
    "  - extracted_keywords_cs / extracted_keywords_en, teater_category, "
    "confidence_score — same meaning as the single-line task.\n"
    "The document may contain HTML-comment layout cues (e.g. "
    "'<!-- BBOX: … -->', '<!-- FONT: … -->'); use them as positional hints but "
    "never extract or quote them as content.\n"
    "Administrative text, tables of contents, headings, author names, and "
    "literature references are NOT extraction targets — skip them entirely "
    "rather than emitting a 'Nerelevantní (meta-text)' item for each. "
    "If the document has no archaeologically relevant passages, return an "
    "empty items list.\n"
    "You MUST respond ONLY with a valid JSON object matching the requested "
    "schema.\n\n"
    "THEMATIC VOCABULARY:\n"
)


def build_document_schema(term_names: List[str]) -> type:
    """Whole-document variant of build_schema(): a wrapper model holding a
    list of located enrichment items, instead of one object per target line.
    Used by run_document_level() for BACKEND=openrouter/ollama .md input."""
    if not term_names:
        raise ValueError("term_names is empty — vocabulary failed to load or was fully truncated.")

    TermEnum = enum.Enum("TermEnum", {f"term_{i}": name for i, name in enumerate(term_names)})

    class LocatedEnrichment(BaseModel):
        locator: str = Field(
            ...,
            description="Short verbatim snippet (max 8 words) copied exactly from the document.",
        )
        page: Optional[str] = Field(
            None,
            description=(
                "Page number/label of the located passage, taken from the nearest "
                "'<!-- PAGE_BREAK: pg_N -->' or '## Page N' marker above it "
                "(a string so labels like 'iv' or 'A-1' are allowed). Null if unknown."
            ),
        )
        extracted_keywords_cs: List[str] = Field(default_factory=list)
        extracted_keywords_en: List[str] = Field(default_factory=list)
        teater_category: TermEnum = Field(
            ...,
            description="The single most relevant category from the thematic vocabulary.",
        )
        confidence_score: float = Field(..., ge=0.0, le=1.0)

        def category_name(self) -> str:
            return self.teater_category.value

    class DocumentEnrichment(BaseModel):
        items: List[LocatedEnrichment] = Field(default_factory=list)

    return DocumentEnrichment


def build_document_system_prompt(
    vocab_data: dict,
    max_tokens: int,
    skip_truncation: bool = False,
) -> Tuple[str, List[str]]:
    """Same vocabulary-injection/truncation as build_system_prompt(), with the
    whole-document instruction header instead of the single-line one."""
    raw_terms = _collect_vocab_terms(vocab_data)
    return _fit_vocab_prompt(
        _DOC_SYSTEM_HEADER, raw_terms, max_tokens, skip_truncation, verbose=False
    )


# Inputs that aren't a native pipeline format but can be pre-converted to
# visually-rich Markdown (document-level) on the fly — see prepare_document_input.
DOC_CONVERT_EXTENSIONS = frozenset({".pdf", ".docx"})


def prepare_document_input(path: Path, cache_dir: Optional[Path] = None, ocr: bool = False) -> Path:
    """Resolve an input file to something the pipeline can read.

    ``.pdf`` / ``.docx`` are converted to visually-rich Markdown (via
    ``api_util.doc_to_visual_md``) and cached as ``<stem>.md`` under a
    ``_visual_md_cache`` sibling dir (not re-scanned by the top-level input
    enumeration); the cached path is returned. The conversion is idempotent —
    skipped when the cached ``.md`` is newer than the source. Any other file
    type is returned unchanged. The heavy converter deps are imported lazily so
    remote/lightweight clients don't pull them unless a PDF/DOCX is actually fed.
    """
    path = Path(path)
    if path.suffix.lower() not in DOC_CONVERT_EXTENSIONS:
        return path

    from api_util.doc_to_visual_md import convert_to_visual_md

    cache = Path(cache_dir) if cache_dir else path.parent / "_visual_md_cache"
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f"{path.stem}.md"
    if out.exists() and out.stat().st_mtime >= path.stat().st_mtime:
        return out
    out.write_text(convert_to_visual_md(path, ocr=ocr), encoding="utf-8")
    return out


def enrichment_block(doc_id: str, results: List[dict]) -> dict:
    """Project this repo's ``*_enriched.json`` records onto the ``enrichment`` block
    of the paired per-document record (see ``atrium_document.py``).

    Handles both record shapes: the document-level one (``locator``/``page``, from
    ``run_document_level``) and the line-level one (``page``/``line``, from
    ``run_line_level``). Only the fields actually present are emitted, and a
    ``[Source: <doc_id>, Page N]`` citation is added whenever a page is known.
    """
    items: List[dict] = []
    for record in results:
        item: dict = {}
        for key in ("locator", "page", "line"):
            if record.get(key) is not None:
                item[key] = record[key]
        item.update(record.get("enrichment") or {})
        if item.get("page") is not None:
            item["citation"] = f"[Source: {doc_id}, Page {item['page']}]"
        items.append(item)
    return {"items": items}


def write_document_record(
    doc_id: str,
    results: List[dict],
    record_dir: Path,
    run_id: Optional[str] = None,
    paradata_ref: str = "",
    enriched_path: Optional[Path] = None,
    markdown_from: Optional[Path] = None,
    detail: str = "full",
    license_detail: Optional[dict] = None,
    used_markdown_input: bool = False,
) -> Optional[Path]:
    """Write/update this document's paired record, contributing llm-enrich's block only.

    Reads ``<record_dir>/<doc_id>.document.json`` as the baseline when it exists and
    writes it back with the ``enrichment`` block replaced — every other tool's block
    passes through untouched. With no baseline present the record is just this tool's
    own part, which is the intended standalone behaviour.

    The ``regenerable.markdown`` recipe records how to rebuild the Markdown this run
    actually fed the LLM (rule: never reference a transient artifact by a stored path).
    Two cases, in priority order:

    * ``used_markdown_input=True`` (a real ``run_document_level`` call, i.e. the input
      was ``.md``/``.txt`` — whether from a pre-converted PDF/DOCX or an upstream
      ``xml_to_md.py --format layout`` pass over TEITOK) — the recipe points at THIS
      SAME document JSON via ``json_to_md``, since it is self-sufficient: a consumer
      holding only the JSON can regenerate equivalent Markdown without also having to
      retain the original PDF/DOCX/TEITOK file (issue #13 §5).
    * Otherwise, if ``markdown_from`` is given (the legacy PDF/DOCX-source path,
      pre-``json_to_md``), fall back to the original ``doc_to_visual_md`` recipe.

    Neither is written for a line-level run (CSV/TEITOK row-by-row) — no Markdown was
    ever fed to the LLM in that case, so no recipe should claim one can be regenerated.
    Returns the record path, or None when the optional ``atrium_document`` module is
    unavailable.
    """
    try:
        from atrium_document import FILE_SUFFIX, DocumentRecord
    except ImportError:
        print(
            "[document] atrium_document.py not available — skipping paired record",
            file=sys.stderr,
        )
        return None

    record_dir = Path(record_dir)
    record_dir.mkdir(parents=True, exist_ok=True)
    baseline = record_dir / f"{doc_id}{FILE_SUFFIX}"

    with DocumentRecord.open(
        doc_id,
        "llm-enrich",
        baseline=str(baseline) if baseline.exists() else None,
        run_id=run_id,
        paradata_ref=paradata_ref,
        out_dir=str(record_dir),
    ) as doc:
        doc.set_block("enrichment", enrichment_block(doc_id, results))
        if enriched_path is not None:
            doc.add_derived_from("enriched", str(enriched_path))
        if used_markdown_input:
            doc.add_regenerable(
                "markdown",
                {"from": f"{doc_id}{FILE_SUFFIX}", "converter": "json_to_md@1.0", "detail": detail},
            )
        elif markdown_from is not None:
            doc.add_regenerable(
                "markdown",
                {"from": str(markdown_from), "converter": "doc_to_visual_md", "detail": detail},
            )
        if license_detail:
            doc.add_license_detail(license_detail)

    return baseline


def run_document_level(
    input_path: Path,
    chat_fn: ChatFn,
    system_prompt: str,
    DocumentEnrichmentModel: type,
    user_content_builder: Optional[Callable[[str], Any]] = None,
) -> Tuple[List[dict], Dict[str, int]]:
    """
    Run whole-document enrichment over a single Markdown/plain-text file
    (typically api_util/xml_to_md.py output). One chat call per document,
    returning every located passage instead of one record per input row.

    ``user_content_builder``, when supplied, is called with the raw document
    text and its return value becomes the user message's ``content`` as-is
    (e.g. OpenRouter's file-attachment content-part list) — this is how
    --attach-as-file actually reaches the wire. When omitted, the document
    text is inlined as plain message text (``DOCUMENT:\n<text>``), matching
    every caller's original behaviour.
    """
    file_id = Path(input_path).stem
    stats: Dict[str, int] = {"processed": 0, "skipped_filter": 0, "skipped_error": 0, "aborted": 0}

    doc_text = Path(input_path).read_text(encoding="utf-8")
    user_content: Any = (
        user_content_builder(doc_text) if user_content_builder else f"DOCUMENT:\n{doc_text}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    try:
        result_json = chat_fn(messages)
        try:
            semantic_data = DocumentEnrichmentModel.model_validate_json(result_json)
        except ValidationError:
            raw_dict = json.loads(result_json, strict=False)
            semantic_data = DocumentEnrichmentModel.model_validate(raw_dict)
    except Exception as exc:
        print(f"  [{file_id}] Document-level inference/validation error: {exc}")
        stats["skipped_error"] += 1
        stats["aborted"] = 1
        return [], stats

    enriched: List[dict] = []
    for item in semantic_data.items:
        dump_data = item.model_dump()
        dump_data["teater_category"] = item.category_name()
        if dump_data["teater_category"] == "Nerelevantní (meta-text)":
            dump_data["extracted_keywords_cs"] = []
            dump_data["extracted_keywords_en"] = []
        enriched.append(
            {
                "file_id": file_id,
                "locator": dump_data.pop("locator"),
                "page": dump_data.pop("page", None),
                "enrichment": dump_data,
            }
        )
    stats["processed"] = len(enriched)
    return enriched, stats


# ---------------------------------------------------------------------------
# 8. Line-level driver — shared by openrouter_client.py and ollama_client.py
# ---------------------------------------------------------------------------


def _coerce_int(value: Any, default: int = 0) -> int:
    """Best-effort int coercion for a row's page_num/line_num field.

    A blank or non-numeric value coerces to `default` instead of raising —
    the line is still processed. Previously run_line_level treated a
    ValueError/TypeError here as a filter-skip and silently dropped the row,
    which mislabelled a data problem as a quality-filter decision."""
    try:
        return int(value)
    except (ValueError, TypeError):
        return default


def run_line_level(
    input_path: Path,
    chat_fn: ChatFn,
    system_prompt: str,
    EnrichmentModel: type,
    include_non_text: bool = True,
    min_char_count: int = 3,
    min_char_non_text: int = 8,
    min_alpha_ratio_non_text: float = 0.40,
    max_consecutive_errors: int = 10,
) -> Tuple[List[dict], Dict[str, int]]:
    """
    Run backend-agnostic line-level enrichment over every qualifying line in
    a single CSV/TEITOK document, mirroring llm_utils.process_document's
    contract (same stats keys, same output record shape, same 10-consecutive-
    error abort behaviour) so results are comparable across BACKEND values.

    ``chat_fn`` does the actual HTTP call; everything else — filtering,
    context-window building, schema validation — is shared here.
    """
    file_id = doc_id_from_path(input_path)
    enriched_lines: List[dict] = []
    stats: Dict[str, int] = {
        "processed": 0,
        "skipped_filter": 0,
        "skipped_error": 0,
        "aborted": 0,
    }
    consecutive_errors = 0
    page_num = line_num = 0

    rows = read_input_rows(input_path)

    for i, row in enumerate(rows):
        try:
            page_num = _coerce_int(row.get("page_num", row.get("page", 0)))
            line_num = _coerce_int(row.get("line_num", row.get("line", 0)))

            text_chunk = row.get("text", "").strip()
            categ = row.get("categ", "").strip()
            quality_score = float(row.get("quality_score") or 0.0)

            should_process, _ = should_process_line(
                text_chunk,
                categ,
                quality_score,
                include_non_text,
                min_char_count,
                min_char_non_text,
                min_alpha_ratio_non_text,
            )
            if not should_process:
                stats["skipped_filter"] += 1
                continue

            context_chunk = get_context_window(rows, i, window=2)
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"DOCUMENT CONTEXT:\n{context_chunk}\n\n"
                        "Task: Extract keywords and determine the TEATER category "
                        "ONLY for the line marked inside <target_line>."
                    ),
                },
            ]

            result_json = chat_fn(messages)
            dump_data = validate_llm_output(
                result_json, EnrichmentModel, file_id, page_num, line_num
            )

            enriched_lines.append(
                {
                    "file_id": file_id,
                    "page": page_num,
                    "line": line_num,
                    "categ": categ,
                    "quality_score": quality_score,
                    "original_text": text_chunk,
                    "enrichment": dump_data,
                }
            )
            stats["processed"] += 1
            consecutive_errors = 0

        except Exception as exc:
            print(f"  [{file_id}] Inference error P{page_num} L{line_num}: {exc}")
            stats["skipped_error"] += 1
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                stats["aborted"] = 1
                print(f"  [{file_id}] Aborting after {consecutive_errors} consecutive errors.")
                break

    return enriched_lines, stats
