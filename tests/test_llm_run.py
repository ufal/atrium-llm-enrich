"""
tests/test_llm_run.py
=====================
Tests for the torch-free logic units of the llm_run.py entry point:
schema construction, system-prompt assembly + token-budget truncation, and
the abort-marker sidecar. The runtime loop (process_document*) needs a real
model and is exercised only in the slow GPU lane.

Importing ``llm_run`` pulls in torch at module top, so this module skips on
the fast lane and runs under requirements_llm.txt.
"""

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from llm_run import (  # noqa: E402
    _write_abort_marker,
    build_schema,
    build_system_prompt,
)


class _FakeTokenizer:
    """Whitespace tokenizer: no ``tokenize`` attr, so count_tokens uses encode()."""

    def encode(self, text):
        return text.split()


# ── build_schema ─────────────────────────────────────────────────────────────


def test_build_schema_empty_raises():
    with pytest.raises(ValueError, match="term_names is empty"):
        build_schema([])


def test_build_schema_accepts_valid_category():
    Model = build_schema(["kostel", "hrad"])
    inst = Model(
        extracted_keywords_cs=["věž"],
        extracted_keywords_en=["tower"],
        teater_category="hrad",
        confidence_score=0.8,
    )
    assert inst.category_name() == "hrad"


def test_build_schema_rejects_unknown_category():
    Model = build_schema(["kostel", "hrad"])
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        Model(
            extracted_keywords_cs=[],
            extracted_keywords_en=[],
            teater_category="not-a-term",
            confidence_score=0.5,
        )


# ── build_system_prompt ──────────────────────────────────────────────────────

_VOCAB = {
    "Structures": {"keywords": {"cs": ["kostel", "hrad"], "en": ["church", "castle"]}},
    "Periods": {"keywords": {"cs": ["středověk"], "en": ["Middle Ages"]}},
}


def test_build_system_prompt_full_fit():
    prompt, terms, _, _ = build_system_prompt(_VOCAB, _FakeTokenizer(), max_tokens=100_000)
    # The mandatory meta-text fallback term is always injected first.
    assert terms[0] == "Nerelevantní (meta-text)"
    for cs in ("kostel", "hrad", "středověk"):
        assert cs in terms
        assert cs in prompt
    assert "THEMATIC VOCABULARY" in prompt


def test_build_system_prompt_truncates_under_tiny_budget():
    full, full_terms, _, _ = build_system_prompt(_VOCAB, _FakeTokenizer(), max_tokens=100_000)
    trunc, trunc_terms, _, _ = build_system_prompt(_VOCAB, _FakeTokenizer(), max_tokens=40)
    assert len(trunc_terms) < len(full_terms)
    assert len(trunc) < len(full)


def test_build_system_prompt_skip_truncation_keeps_full():
    # Even with a tiny budget, skip_truncation must return the full vocabulary
    # (used when vLLM prefix caching makes truncation pointless).
    prompt, terms, _, _ = build_system_prompt(
        _VOCAB, _FakeTokenizer(), max_tokens=1, skip_truncation=True
    )
    for cs in ("kostel", "hrad", "středověk"):
        assert cs in terms


# ── _write_abort_marker ──────────────────────────────────────────────────────


def test_write_abort_marker(tmp_path):
    out_file = tmp_path / "CTX1_enriched.json"
    _write_abort_marker(out_file, {"processed": 7, "skipped_error": 10}, reason="boom")
    marker = tmp_path / "CTX1_enriched.abort.json"
    assert marker.exists()
    data = json.loads(marker.read_text(encoding="utf-8"))
    assert data["aborted"] is True
    assert data["abort_reason"] == "boom"
    assert data["processed_before_abort"] == 7
    assert data["errors_before_abort"] == 10
