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
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field, ValidationError

from api_util import teitok_read
from api_util.teitok_read import doc_id_from_path  # noqa: F401  (re-exported for clients)

# ---------------------------------------------------------------------------
# 1. Config loader — duplicated from llm_utils.load_config
# ---------------------------------------------------------------------------


def _unquote(value: str) -> str:
    """Strip one matched pair of surrounding quotes from a config value.

    Config files here are shell-flavoured KEY=VALUE, and both quoted and bare
    values occur in the wild — this repo's own llm_config.txt writes them bare,
    while generated configs (the cross-repo e2e workflow, deployment templates)
    quote paths out of shell habit. Without this, a quoted value keeps its quote
    characters and every path built from it is wrong by two bytes.

    That is not hypothetical: `VOCAB_PATH="/workspace/work/…json"` parsed to a
    path that could not exist, VocabularyManager fell through to auto-sync, and
    the run produced a single-term enum that rejected every correct answer the
    model gave. atrium-nlp-enrich's sibling parser (api_util/summarize_nt_udp.py)
    has always stripped quotes; the two repos consume configs written by the same
    hand, so differing on this is a trap rather than a design choice.

    Only a MATCHED outer pair is removed, so a Windows path or a value with an
    apostrophe inside survives untouched.
    """
    for quote in ('"', "'"):
        if len(value) >= 2 and value.startswith(quote) and value.endswith(quote):
            return value[1:-1]
    return value


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
                config[key.strip()] = _unquote(value.strip())
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

    if dump_data.get("teater_category") == META_TERM:
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


#: The one administrative label _collect_vocab_terms() prepends unconditionally.
#: It is not part of any vocabulary file, so an enum consisting of it alone proves
#: that zero real terms survived loading + theme filtering.
META_TERM = "Nerelevantní (meta-text)"


def _assert_vocabulary_reached_the_model(term_names: List[str]) -> None:
    """Reject a term list that carries no actual vocabulary.

    The empty case was always caught. The meta-only case was not, and it is the
    one that actually happened: a vocabulary that loads but whose every term is
    filtered out yields a one-value enum, and pydantic then rejects each correct
    answer the model returns with a validation error. The run completes, exits 0,
    and reports "records enriched: 0" — a silent quality collapse that reads like
    a model failure. Both cases mean the same thing and neither is a judgement
    call about size: the check is exact, not a threshold.
    """
    if not term_names:
        raise ValueError("term_names is empty — vocabulary failed to load or was fully truncated.")
    if set(term_names) <= {META_TERM}:
        raise ValueError(
            f"Vocabulary contains no terms beyond the fixed {META_TERM!r} category, so "
            "every enrichment would be forced to that one label. Check VOCAB_PATH points "
            "at a built vocabulary (see vocab_build.py) and that taxonomy_config.json "
            "does not set in_prompt=false on every theme."
        )


def build_schema(term_names: List[str]) -> type:
    _assert_vocabulary_reached_the_model(term_names)

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


def excluded_prompt_themes(vocab_mgr: Any) -> Set[str]:
    """Themes to withhold from the model, derived from taxonomy_config.json.

    A theme is withheld when its ``in_prompt`` flag is false; absent the flag the
    default is the historical one — everything except "Other" reaches the model.

    The trailing guard is not redundant. VocabularyManager falls back to a
    BUILT-IN taxonomy when data_samples/taxonomy_config.json is missing, and that
    fallback declares no "Other" theme at all — so a bare comprehension would
    produce an empty exclusion set and silently start injecting the ~779-term
    Other bucket into every prompt. A config that is simply silent about Other
    must mean "unchanged", never "enable it"; enabling it takes an explicit
    ``"Other": {"in_prompt": true}``.

    Accepts any object exposing ``themes()`` (i.e. a VocabularyManager); typed
    loosely so this module keeps its no-heavy-imports property.
    """
    try:
        themes = vocab_mgr.themes()
    except AttributeError:  # pragma: no cover — a manager predating themes()
        return {"other"}
    excluded = {
        name.lower()
        for name, cfg in themes.items()
        if isinstance(cfg, dict) and not cfg.get("in_prompt", name.lower() != "other")
    }
    if "other" not in {name.lower() for name in themes}:
        excluded.add("other")
    return excluded


def _collect_vocab_terms(
    vocab_data: dict, excluded_themes: Optional[Set[str]] = None
) -> List[dict]:
    """Flatten ``vocab_data`` into a list of ``{theme, cs, en}`` term dicts,
    with the fixed 'Nerelevantní (meta-text)' administrative term prepended.

    Shared by build_system_prompt() and build_document_system_prompt() —
    the two callers differ only in header/footer text, not in how
    vocabulary terms are gathered from the nested theme/keyword structure.

    ``excluded_themes`` names the themes to withhold from the model,
    lower-cased. It defaults to ``{"other"}`` — the behaviour this function
    hard-coded before — but the callers derive it from each theme's
    ``in_prompt`` flag in taxonomy_config.json via
    :func:`excluded_prompt_themes`, so which terms the model can reach is a
    reviewable configuration decision rather than a literal in the prompt
    builder. This matters: a term absent from the prompt is unreachable by
    construction, so withholding one makes "the model was wrong" and "the
    label was withheld" score identically.

    Keys starting with "_" are skipped unconditionally. Nothing writes such a
    key into a nested vocabulary today (provenance lives in a sidecar
    ``*.meta.json`` precisely so it cannot), but a stray one would otherwise
    be rendered to the model as a phantom theme."""
    skip = {"other"} if excluded_themes is None else {t.lower() for t in excluded_themes}
    raw_terms: List[dict] = [
        {
            "theme": "Administrative / Meta",
            "sub": "",
            "cs": META_TERM,
            "en": "Irrelevant / Meta-text",
        }
    ]
    for theme, data in vocab_data.items():
        if theme.startswith("_") or theme.lower() in skip:
            continue
        if isinstance(data, dict):
            if "keywords" in data and isinstance(data["keywords"], dict):
                cs_list = data["keywords"].get("cs", [])
                en_list = data["keywords"].get("en", [])
                for i, cs_key in enumerate(cs_list):
                    en = en_list[i] if i < len(en_list) else cs_key
                    raw_terms.append({"theme": theme, "sub": "", "cs": cs_key, "en": en})
            else:
                for cs_key, pair in data.items():
                    en = pair.get("en", cs_key) if isinstance(pair, dict) else cs_key
                    sub = pair.get("sub", "") if isinstance(pair, dict) else ""
                    raw_terms.append({"theme": theme, "sub": sub, "cs": cs_key, "en": en})
    return raw_terms


def _render_vocab_prompt(header: str, term_list: List[dict], footer: str = "") -> str:
    """Render ``term_list`` grouped by facet, then by the source's own subgroup.

    Both AMCR and TEATER curate a second level — 50 heslars, and TEATER's depth-2 groups
    — and flattening a 700-term facet into one undifferentiated list throws that away.
    Two levels cost ~100 header lines and give the model structure a domain expert
    already built."""
    groups: Dict[Tuple[str, str], List[str]] = {}
    for t in term_list:
        key = (t["theme"], t.get("sub") or "")
        groups.setdefault(key, []).append(f"{t['cs']} ({t['en']})")

    prompt = header
    for (theme_name, sub_name), lines in groups.items():
        title = f"{theme_name} / {sub_name}" if sub_name else theme_name
        prompt += f"\n--- {title} ---\n"
        prompt += "\n".join(f"- {line}" for line in lines) + "\n"
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
    excluded_themes: Optional[Set[str]] = None,
) -> Tuple[str, List[str]]:
    """Same vocabulary-truncation strategy as llm_run.build_system_prompt, but
    driven by approx_token_count() instead of a tokenizer — no HF/torch
    dependency, at the cost of an approximate (not exact) token budget.

    ``excluded_themes`` is forwarded to _collect_vocab_terms(); omitting it
    keeps the historical behaviour of withholding only "Other"."""
    raw_terms = _collect_vocab_terms(vocab_data, excluded_themes)
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
    # Spelled out rather than delegated. This used to read "same meaning as the
    # single-line task" — a task this prompt never shows the model, because the
    # document-level prompt is built from _DOC_SYSTEM_HEADER alone. So the four
    # remaining fields were specified nowhere the model could see, and it invented
    # a shape: `page` as an integer, the keyword lists as one comma-joined string,
    # and a `teater_category` of its own wording. atrium-project run 34123820218 is
    # that, five times over, on a document it had otherwise read correctly.
    "  - extracted_keywords_cs: a JSON ARRAY of Czech terms found in the passage, "
    'e.g. ["sonda", "kulturní vrstva"]. NEVER a single comma-separated string. '
    "Empty array if none.\n"
    "  - extracted_keywords_en: a JSON ARRAY of English translations of "
    "extracted_keywords_cs, same length and order. NEVER a single "
    "comma-separated string.\n"
    # The heading warning is not padding. _render_vocab_prompt() emits the vocabulary as
    # `--- {theme} / {sub} ---` section headings over `- cs (en)` bullets, and in
    # atrium-project run 34125325468 the model answered 'Artefact / druh předmětu' for
    # every item — a heading this very renderer had printed. Nothing in the prompt had
    # ever said which of the two kinds of line is selectable.
    "  - teater_category: ONE value copied EXACTLY, character for character, from "
    "the THEMATIC VOCABULARY list below — including its diacritics and any "
    "parenthesised qualifier. Do not translate it, do not shorten it, do not "
    "invent a category.\n"
    "    The vocabulary is printed as sections. A line of the form "
    "'--- Something / Something ---' is a SECTION HEADING and is NOT a category: "
    "never return one. Only the entries listed under a heading are categories, and "
    "each is printed as '- <czech term> (<english gloss>)'. Return ONLY the Czech "
    "term, without the English gloss and without its surrounding parentheses: from "
    "'- sonda (trench)' the correct value is exactly 'sonda'.\n"
    "    If no listed term fits the passage, the passage is not an extraction "
    "target: omit the item entirely.\n"
    "  - confidence_score: a number between 0.0 and 1.0.\n"
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


#: The document-level counterpart of _EXAMPLES_FOOTER.
#:
#: build_system_prompt() has passed _EXAMPLES_FOOTER to _fit_vocab_prompt() since it was
#: written; build_document_system_prompt() passed no footer at all, so the whole-document
#: prompt shipped without a single worked example. A prompt that only DESCRIBES a JSON
#: shape and never shows one is how atrium-project run 34123820218 got five items whose
#: content was right and whose every field was the wrong type. The example is the cheap
#: half of the fix; the tolerant parse in _repair_document_response() is the other half.
#:
#: `page` is quoted here deliberately — it is a STRING in the schema so labels like "iv"
#: or "A-1" survive, and an unquoted 1 is exactly what the model returned instead.
_DOC_EXAMPLES_FOOTER = (
    "\nEXAMPLE OF THE REQUIRED OUTPUT SHAPE:\n\n"
    "For a document containing:\n"
    "  ## Page 2\n"
    "  Sonda II odkryla cast valoveho telesa.\n"
    "  Nalezeny zlomky keramiky z raneho stredoveku.\n\n"
    "Correct output:\n"
    "{\n"
    '  "items": [\n'
    "    {\n"
    '      "locator": "Sonda II odkryla cast",\n'
    '      "page": "2",\n'
    '      "extracted_keywords_cs": ["sonda", "valové těleso"],\n'
    '      "extracted_keywords_en": ["trench", "rampart body"],\n'
    '      "teater_category": "<an exact term from the vocabulary above>",\n'
    '      "confidence_score": 0.9\n'
    "    }\n"
    "  ]\n"
    "}\n\n"
    'Note: "page" is a STRING, both keyword fields are ARRAYS, and '
    '"teater_category" is copied verbatim from the vocabulary. A document with '
    'nothing archaeological in it returns {"items": []}.\n'
)


def build_document_schema(term_names: List[str]) -> type:
    """Whole-document variant of build_schema(): a wrapper model holding a
    list of located enrichment items, instead of one object per target line.
    Used by run_document_level() for BACKEND=openrouter/ollama .md input."""
    _assert_vocabulary_reached_the_model(term_names)

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

    #: The enum's values, hung on the class so _repair_document_response() can resolve a
    #: near-miss category without reaching back into pydantic's internals. Assigned after
    #: class creation so pydantic does not mistake it for a field.
    DocumentEnrichment.allowed_terms = tuple(term_names)
    return DocumentEnrichment


def build_document_system_prompt(
    vocab_data: dict,
    max_tokens: int,
    skip_truncation: bool = False,
    excluded_themes: Optional[Set[str]] = None,
) -> Tuple[str, List[str]]:
    """Same vocabulary-injection/truncation as build_system_prompt(), with the
    whole-document instruction header instead of the single-line one."""
    raw_terms = _collect_vocab_terms(vocab_data, excluded_themes)
    return _fit_vocab_prompt(
        _DOC_SYSTEM_HEADER,
        raw_terms,
        max_tokens,
        skip_truncation,
        footer=_DOC_EXAMPLES_FOOTER,
        verbose=False,
    )


# Inputs that aren't a native pipeline format but can be pre-converted to
# visually-rich Markdown (document-level) on the fly — see prepare_document_input.
DOC_CONVERT_EXTENSIONS = frozenset({".pdf", ".docx"})

#: An atrium_document record. Matched on the FULL name, not Path.suffix: the suffix
#: of "minimal.document.json" is ".json", which is why this could not simply join
#: DOC_CONVERT_EXTENSIONS. Kept as a literal here rather than imported from
#: api_util.doc_to_visual_md so this module stays free of the converter's deps —
#: the import there is lazy and must remain so.
DOCUMENT_JSON_SUFFIX = ".document.json"


def is_convertible_input(path: Path) -> bool:
    """Whether prepare_document_input() will render this to Markdown.

    Mirrors ``api_util.doc_to_visual_md.is_supported``. The two must agree: the
    converter accepted ``*.document.json`` from the day it was written, but the
    gate in front of it tested ``Path.suffix in DOC_CONVERT_EXTENSIONS``, which a
    record can never satisfy. Feeding one to the pipeline therefore skipped
    conversion, fell through to the line-level branch, and had csv.DictReader parse
    a JSON file — yielding zero records, zero errors, no API call and exit 0
    (atrium-project run 34039707673). Callers use this for input enumeration too,
    so a directory of records is discovered rather than silently ignored.
    """
    return str(path).lower().endswith(DOCUMENT_JSON_SUFFIX) or (
        Path(path).suffix.lower() in DOC_CONVERT_EXTENSIONS
    )


#: Formats each branch of the dispatch can actually read. Document-level is also
#: spelled `_DOC_INPUT_EXTENSIONS` inside each client, where it selects the branch;
#: here the two sets together answer a different question — whether ANY branch can
#: read the file at all.
DOC_LEVEL_EXTENSIONS = frozenset({".md", ".txt"})
LINE_LEVEL_EXTENSIONS = frozenset({".csv"})


def has_reader(path: Path) -> bool:
    """Whether some branch of the client dispatch can read this file.

    Nothing checked this before, and the dispatch has no else: a file that is
    neither document-level nor convertible simply fell through to the line-level
    branch, whatever it was. csv.DictReader does not object to being handed JSON —
    it just yields nothing — so an unreadable input cost a full CI round trip to
    diagnose instead of one line of output.
    """
    name = str(path).lower()
    return (
        Path(path).suffix.lower() in DOC_LEVEL_EXTENSIONS
        or Path(path).suffix.lower() in LINE_LEVEL_EXTENSIONS
        or name.endswith(".teitok.xml")
        or is_convertible_input(path)
    )


def _markdown_cache_stem(path: Path) -> str:
    """Base name for the cached Markdown. ``Path.stem`` strips only the last
    extension, which would render "minimal.document.json" to "minimal.document.md";
    stripping the whole compound suffix keeps the cache name aligned with the
    ``doc_id`` canonical_doc_id() derives from the same file."""
    name = path.name
    if name.lower().endswith(DOCUMENT_JSON_SUFFIX):
        return name[: -len(DOCUMENT_JSON_SUFFIX)]
    return path.stem


def prepare_document_input(path: Path, cache_dir: Optional[Path] = None, ocr: bool = False) -> Path:
    """Resolve an input file to something the pipeline can read.

    ``.pdf`` / ``.docx`` / ``*.document.json`` are converted to visually-rich
    Markdown (via ``api_util.doc_to_visual_md``) and cached as ``<stem>.md`` under a
    ``_visual_md_cache`` sibling dir (not re-scanned by the top-level input
    enumeration); the cached path is returned. The conversion is idempotent —
    skipped when the cached ``.md`` is newer than the source. Any other file
    type is returned unchanged. The heavy converter deps are imported lazily so
    remote/lightweight clients don't pull them unless one is actually fed.
    """
    path = Path(path)
    if not is_convertible_input(path):
        return path

    from api_util.doc_to_visual_md import convert_to_visual_md

    cache = Path(cache_dir) if cache_dir else path.parent / "_visual_md_cache"
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f"{_markdown_cache_stem(path)}.md"
    if out.exists() and out.stat().st_mtime >= path.stat().st_mtime:
        return out
    out.write_text(convert_to_visual_md(path, ocr=ocr), encoding="utf-8")
    return out


#: The four outcomes one llm-enrich pass over one document can have.
#:
#: Until atrium-project#49 three of them were indistinguishable in the emitted artifact.
#: `--document-json`/`--document-json-out` copies the caller's BASELINE into a scratch dir
#: and copies whatever is in that dir back out at the end, so a run that contributed
#: nothing shipped the untouched baseline, announced "[document] Record written", and
#: exited 0. A crashed inference did the same. So did a healthy one that located nothing.
#: The consumer — atrium-project's tools/e2e/e2e_assert.py, and every downstream tool —
#: saw one record with no `enrichment` block and could only guess which of the three it
#: was looking at. Run 34090340995 is the guess going wrong: the digital smoke reported
#: "'enrichment' block missing from llm-enrich stage" for what was in fact a correct,
#: successful, empty enrichment of a fixture with no archaeological content in it.
OUTCOME_CONTRIBUTED = "contributed"
OUTCOME_EMPTY = "empty"
OUTCOME_FAILED = "failed"
OUTCOME_NOT_ASKED = "not-asked"

#: The outcomes that mean llm-enrich has something to say about this document, i.e. the
#: ones that MUST write the `enrichment` block. `empty` is in here on purpose: a stage
#: that ran and found nothing is a different fact from a stage that never ran, and
#: `assembled.blocks` is the record's account of which tool contributed what — so the
#: only honest way to record "llm-enrich looked and there was nothing" is an enrichment
#: block with an empty `items` list, stamped by llm-enrich.
_CONTRIBUTING_OUTCOMES = frozenset({OUTCOME_CONTRIBUTED, OUTCOME_EMPTY})


def classify_outcome(results: List[dict], stats: Dict[str, int]) -> str:
    """Which of the four outcomes this pass had, from the driver's own stats.

    Order matters. Partial success is still a contribution: run_line_level() can enrich
    nine rows and error on the tenth, and that run has results to write — the error is
    already in `skipped_error` and in the paradata, and refusing the record over it would
    throw away nine good enrichments.

    `attempted` is what makes the empty/not-asked split possible; both have
    ``processed == 0`` and neither raises. See the stats dicts in run_document_level()
    and run_line_level().
    """
    if results:
        return OUTCOME_CONTRIBUTED
    if stats.get("aborted") or stats.get("skipped_error"):
        return OUTCOME_FAILED
    if stats.get("attempted"):
        return OUTCOME_EMPTY
    return OUTCOME_NOT_ASKED


def contributes_document_record(results: List[dict], stats: Dict[str, int]) -> bool:
    """Whether this pass may write its `enrichment` block onto the paired record.

    True for a real enrichment and for a model that was asked and located nothing.
    False when the model was never successfully consulted — there is no verdict to
    record, and writing an empty block would claim one.
    """
    return classify_outcome(results, stats) in _CONTRIBUTING_OUTCOMES


def enrichment_block(doc_id: str, results: List[dict]) -> dict:
    """Project this repo's ``*_enriched.json`` records onto the ``enrichment`` block
    of the paired per-document record (see ``atrium_document.py``).

    Handles both record shapes: the document-level one (``locator``/``page``, from
    ``run_document_level``) and the line-level one (``page``/``line``, from
    ``run_line_level``). Only the fields actually present are emitted, and a
    ``[Source: <doc_id>, Page N]`` citation is added whenever a page is known.

    ``page`` is emitted as a STRING because that is what the schema says it is — the same
    reason ``lines[].page`` is a string, so a label like ``"iv"`` or ``"A-1"`` survives.
    ``run_line_level`` coerces ``page_num`` to an int for its own arithmetic, so every
    line-level run used to write an integer here and the resulting record was
    schema-INVALID — caught the moment the Layer D gate below was actually wired
    (atrium-project#10, D4), having gone unnoticed for as long as nothing validated.
    """
    items: List[dict] = []
    for record in results:
        item: dict = {}
        for key in ("locator", "page", "line"):
            value = record.get(key)
            if value is not None:
                # `line` stays an int: the schema does not constrain it, and
                # BLOCK_KEY_FIELDS keys `lines[]` on it as an integer.
                item[key] = str(value) if key == "page" else value
        item.update(record.get("enrichment") or {})
        if item.get("page") is not None:
            item["citation"] = f"[Source: {doc_id}, Page {item['page']}]"
        items.append(item)
    return {"items": items}


#: One-shot latch so a DISABLED gate is announced once per process, not once per document.
#: See schema_gate().
_schema_gate_disabled_warned = False


def schema_gate(record: Dict[str, Any], what: str) -> Optional[str]:
    """Validate one record against ``atrium_document.schema.json``.

    Returns None when it validates, or a one-line description of the schema error when it
    does not. This is plan §2's **Layer D** — "no doc.json is emitted if validation fails" —
    adopted here for atrium-project#10 (D4), which found ``validate_document()`` called from
    no production path in any of the five repos: the gate documented as normative in
    ``docs/document_schema.md`` was protecting nothing at all.

    Deliberately only answers *"is it valid"*. The POLICY — who raises and who merely warns —
    lives at the two call sites, because it differs for an inherited baseline and for this
    tool's own output; see ``write_document_record()``.

    A missing ``jsonschema`` (RuntimeError from ``validate_document()``), a module vendored
    without its schema (FileNotFoundError from ``load_schema()``) or an unparseable schema
    (JSONDecodeError) all mean the GATE is absent, not that the record is bad — a
    ``jsonschema.ValidationError`` is none of those three, so nothing real is swallowed here.
    They degrade to ONE loud warning and a pass: a gate that
    silently no-ops is indistinguishable in the output from a gate that passed, which is the
    precise failure mode D4 is about. ``jsonschema`` is declared in ``requirements.txt`` — the
    base install every image and the test job actually build from — so the degraded path
    should never be taken in a supported deployment.
    """
    global _schema_gate_disabled_warned
    try:
        from atrium_document import validate_document
    except ImportError:
        return None

    try:
        validate_document(record)
    except (RuntimeError, FileNotFoundError, json.JSONDecodeError) as exc:
        if not _schema_gate_disabled_warned:
            print(
                f"[document] WARNING - schema validation is DISABLED for {what} and every "
                f"record after it: {exc}",
                file=sys.stderr,
            )
            _schema_gate_disabled_warned = True
        return None
    except Exception as exc:
        # jsonschema.ValidationError: `.message` is the human-readable half and `.json_path`
        # points at the offending node. Both are absent on any other validator, hence getattr.
        detail = getattr(exc, "message", None) or str(exc)
        path = getattr(exc, "json_path", "") or ""
        return f"{detail}{f' at {path}' if path else ''}"
    return None


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

    This is also the repo's **single Layer D chokepoint** (atrium-project#10, D4). Every
    write path — both batch clients and ``service/api.py`` — comes through here, so the
    schema gate is applied once, not once per caller. The ecosystem-wide policy is:

    * an **inherited baseline** that does not validate warns and continues (refusing to run
      because an upstream tool wrote something invalid turns one bad record into a stalled
      pipeline, and rule 6 already commits to passing unknown content through);
    * **this tool's own output** that does not validate raises, so the record is never
      emitted — unless the baseline was already invalid, in which case the defect is
      inherited rather than ours and it warns instead.
    """
    try:
        from atrium_document import FILE_SUFFIX, SCHEMA_FILENAME, DocumentRecord, load_document
    except ImportError:
        print(
            "[document] atrium_document.py not available — skipping paired record",
            file=sys.stderr,
        )
        return None

    record_dir = Path(record_dir)
    record_dir.mkdir(parents=True, exist_ok=True)
    baseline = record_dir / f"{doc_id}{FILE_SUFFIX}"

    # Layer D, first half: judge the baseline as it ARRIVED. Read separately from
    # DocumentRecord.open() below (which re-reads it) so the verdict is about the upstream
    # tool's output and not about anything this run has since applied to it. It also sets the
    # severity of the second half — a schema error we inherited is not ours to fail on.
    baseline_was_invalid = False
    if baseline.exists():
        baseline_error = schema_gate(load_document(str(baseline)), f"baseline {baseline.name}")
        if baseline_error:
            baseline_was_invalid = True
            print(
                f"[document] WARNING - inherited baseline {baseline.name} does not validate "
                f"against {SCHEMA_FILENAME} ({baseline_error}) - continuing anyway (rule 6), "
                f"and demoting this run's own output check to a warning",
                file=sys.stderr,
            )

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

        # Layer D, second half: never EMIT an invalid record. Raising here — INSIDE the
        # `with` — is what enforces that: DocumentRecord.__exit__ finalises only when no
        # exception is in flight, so nothing reaches disk and the next tool never loads a
        # record this one knew was broken. Both clients call this from inside their per-file
        # try/except, so one bad document is logged and skipped rather than killing the run.
        own_error = schema_gate(doc.to_dict(), f"{doc_id}{FILE_SUFFIX}")
        if own_error:
            if baseline_was_invalid:
                print(
                    f"[document] WARNING - {doc_id}{FILE_SUFFIX} does not validate against "
                    f"{SCHEMA_FILENAME} ({own_error}) - emitting it anyway because the "
                    f"baseline was already invalid; fix the upstream record first",
                    file=sys.stderr,
                )
            else:
                raise RuntimeError(
                    f"llm-enrich's own document record for {doc_id} does not validate against "
                    f"{SCHEMA_FILENAME}: {own_error} - refusing to emit it (Layer D)"
                )

    return baseline


# ---------------------------------------------------------------------------
# 7b. Repairing a document-level reply — atrium-project#49 / run 34123820218
# ---------------------------------------------------------------------------
#
# The whole-document request goes out as `response_format: {"type": "json_object"}`
# unless --structured-outputs is passed, and the E2E does not pass it: with a 4719-value
# enum the json_schema variant is far larger than most providers accept, and
# --provider-data-collection deny narrows routing to providers whose structured-output
# support varies. So the shape is enforced by the PROMPT, and the prompt is advice.
#
# gpt-4o-mini's deviations, observed in full in run 34123820218 (five items, four
# validation errors each, on content it had otherwise read correctly):
#
#   page                    -> 1 (int)                      instead of "1"
#   extracted_keywords_cs   -> "hradiste, lokalita, Beroun" instead of [...]
#   extracted_keywords_en   -> "fortress, site, Beroun"     instead of [...]
#   teater_category         -> a term of its own wording, not one from the vocabulary
#
# The first three are unambiguous formatting slips over a correct answer, and throwing
# the answer away for them is the wrong trade. The fourth is not a formatting slip: an
# unlisted category is a claim the vocabulary does not support, and inventing a mapping
# for it would fabricate data. So the first three are coerced and the fourth is resolved
# only against the vocabulary itself — exact, then case- and whitespace-insensitive —
# and the item is DROPPED, loudly and counted, when that fails.

_KEYWORD_SEPARATORS = re.compile(r"[;,]")

#: One trailing "(...)" group, used only as a last resort — see _term_resolver().
_PARENTHETICAL_TAIL = re.compile(r"\s*\([^()]*\)\s*$")

#: A value shaped like one of _render_vocab_prompt()'s own `--- theme / sub ---` headings.
#: Recognised purely so the drop reason can NAME the mistake: a heading coming back as a
#: category means the prompt failed to distinguish its two kinds of line, which is a
#: prompt bug to fix, not a model quirk to absorb. Deliberately never resolved to a term —
#: a heading names a whole section, so picking any member of it would be a guess.
_LOOKS_LIKE_A_HEADING = re.compile(r"^[^/]+ / [^/]+$")


def _as_keyword_list(value: Any) -> List[str]:
    """A keyword field as the list the schema asks for.

    A bare string is split on commas/semicolons — that is the exact form the model
    returns, and it is unambiguous here because a vocabulary keyword never contains one.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        return [part.strip() for part in _KEYWORD_SEPARATORS.split(value) if part.strip()]
    return [str(value).strip()]


def _as_page_label(value: Any) -> Optional[str]:
    """A page as the STRING the schema asks for, preserving non-numeric labels.

    `page` is a string so "iv" or "A-1" survive (the same reason lines[].page is one).
    A float that is a whole number renders as "2", not "2.0" — json.loads gives a float
    for `2.0`, and "2.0" would not match any page label a renderer emits.
    """
    if value is None:
        return None
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    return text or None


def _term_resolver(model: type) -> Callable[[Any], Optional[str]]:
    """Resolve a model-supplied category against the vocabulary the enum was built from.

    Exact match first, then a casefolded/whitespace-collapsed match, which recovers the
    near-misses ("Kostel", "kostel " ) without inventing anything. Anything else is
    unresolvable BY DESIGN: see the module note above.
    """
    allowed = tuple(getattr(model, "allowed_terms", ()) or ())
    loose = {" ".join(term.casefold().split()): term for term in allowed}

    def _loose(value: str) -> Optional[str]:
        return loose.get(" ".join(value.casefold().split()))

    def resolve(value: Any) -> Optional[str]:
        if not isinstance(value, str):
            return None
        if value in allowed:
            return value
        hit = _loose(value)
        if hit is not None:
            return hit
        # The vocabulary is printed as `- <czech> (<english>)`, so a model that copies a
        # whole bullet returns "sonda (trench)". Strip ONE trailing parenthetical and
        # retry — but only here, after the full string has already failed to match, which
        # is what keeps terms that legitimately end in one ("atlantik (paleoklimatologie)",
        # META_TERM itself) from being mangled: those match on the first two attempts.
        trimmed = _PARENTHETICAL_TAIL.sub("", value).strip()
        if trimmed and trimmed != value:
            if trimmed in allowed:
                return trimmed
            return _loose(trimmed)
        return None

    return resolve


def _repair_document_response(result_json: str, model: type, file_id: str) -> Tuple[Any, List[str]]:
    """Coerce a near-miss document-level reply into the schema, or raise.

    Returns the validated model plus one human-readable line per dropped item. Raises
    (to run_document_level's handler, which records an inference error) when the payload
    is not JSON, is not an object with an `items` array, or still fails validation after
    coercion — those are not formatting slips and must stay loud.
    """
    raw = json.loads(result_json, strict=False)
    if isinstance(raw, list):
        # Some replies drop the wrapper and return the array on its own.
        raw = {"items": raw}
    if not isinstance(raw, dict):
        raise ValueError(f"expected a JSON object, got {type(raw).__name__}")

    items = raw.get("items")
    if items is None:
        raise ValueError("reply has no 'items' key")
    if not isinstance(items, list):
        raise ValueError(f"'items' is {type(items).__name__}, expected a list")

    resolve = _term_resolver(model)
    repaired: List[dict] = []
    dropped: List[str] = []

    for index, item in enumerate(items):
        if not isinstance(item, dict):
            dropped.append(f"item {index}: not an object ({type(item).__name__})")
            continue

        raw_category = item.get("teater_category")
        category = resolve(raw_category)
        if category is None:
            hint = ""
            if isinstance(raw_category, str) and _LOOKS_LIKE_A_HEADING.match(raw_category.strip()):
                hint = (
                    " — that is one of the vocabulary's own '--- theme / sub ---' SECTION "
                    "HEADINGS, not a term under it"
                )
            dropped.append(
                f"item {index}: teater_category {raw_category!r} is not in the vocabulary{hint}"
            )
            continue

        fixed = dict(item)
        fixed["teater_category"] = category
        fixed["page"] = _as_page_label(item.get("page"))
        fixed["extracted_keywords_cs"] = _as_keyword_list(item.get("extracted_keywords_cs"))
        fixed["extracted_keywords_en"] = _as_keyword_list(item.get("extracted_keywords_en"))
        try:
            fixed["confidence_score"] = min(1.0, max(0.0, float(item.get("confidence_score"))))
        except (TypeError, ValueError):
            dropped.append(
                f"item {index}: confidence_score {item.get('confidence_score')!r} is not a number"
            )
            continue
        repaired.append(fixed)

    for reason in dropped:
        print(f"    [dropped] {reason}")

    if items and not repaired:
        # Every single item unusable is a systematic mismatch, and it must NOT become an
        # empty verdict. An `enrichment: {items: []}` block is the record's way of saying
        # "the model looked and there was nothing here" — reporting it when the model in
        # fact found five things we could not read would be a lie in the record and a
        # green digital smoke that enriched nothing. A gate that goes quiet is worse than
        # one that goes red. So this raises, and run_document_level's handler turns it
        # into an inference error, which under atrium-project#49's contract means no
        # document record and a non-zero exit.
        raise ValueError(
            f"the model returned {len(items)} item(s) and none survived repair: "
            + "; ".join(dropped)
        )

    print(
        f"  [{file_id}] repaired a non-conforming document reply: "
        f"{len(repaired)} item(s) recovered, {len(dropped)} dropped"
    )
    return model.model_validate({"items": repaired}), dropped


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
    stats: Dict[str, int] = {
        "processed": 0,
        "skipped_filter": 0,
        "skipped_error": 0,
        "aborted": 0,
        # `attempted` counts model calls MADE, not records produced, and it is the only
        # thing that separates "we asked and it located nothing" from "we never asked"
        # (atrium-project#49). `processed` cannot: it is 0 for both. See
        # classify_outcome() for why the difference decides whether a document record
        # is written at all.
        "attempted": 0,
    }

    doc_text = Path(input_path).read_text(encoding="utf-8")
    user_content: Any = (
        user_content_builder(doc_text) if user_content_builder else f"DOCUMENT:\n{doc_text}"
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # Counted BEFORE the call, not after: a call that raises was still an attempt, and
    # the failure branch below needs to be distinguishable from "no input reached the
    # model" rather than from "the model answered".
    stats["attempted"] = 1
    try:
        result_json = chat_fn(messages)
        try:
            semantic_data = DocumentEnrichmentModel.model_validate_json(result_json)
        except ValidationError:
            # The repair pass. This used to re-validate the SAME payload against the
            # SAME strict model, so for a shape error it could only raise again —
            # a retry that cannot succeed. _repair_document_response() coerces the
            # deviations the model actually produces before re-validating.
            semantic_data, dropped = _repair_document_response(
                result_json, DocumentEnrichmentModel, file_id
            )
            stats["repaired"] = 1
            stats["dropped_items"] = len(dropped)
    except Exception as exc:
        print(f"  [{file_id}] Document-level inference/validation error: {exc}")
        stats["skipped_error"] += 1
        stats["aborted"] = 1
        return [], stats

    enriched: List[dict] = []
    for item in semantic_data.items:
        dump_data = item.model_dump()
        dump_data["teater_category"] = item.category_name()
        if dump_data["teater_category"] == META_TERM:
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
        # See run_document_level() for what `attempted` is for. Here it counts the ROWS
        # actually sent to the model — a CSV whose every row was dropped by
        # should_process_line() never consulted it, and must not be reported as an
        # enrichment that found nothing.
        "attempted": 0,
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

            stats["attempted"] += 1
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
