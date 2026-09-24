"""
tests/test_llm_utils.py
=======================
Tests for the pure helpers in llm_utils.py.

llm-enrich owns the canonical LLM engine yet previously shipped no direct
coverage of llm_utils.py (only test_offload_budget.py exercised the offload
subsystem). This mirrors the nlp-enrich twin's suite so both copies of the
engine stay covered in lockstep — important for the shared-engine drift check.

Importing ``llm_utils`` needs torch + transformers (GPU lane only), so the
whole module skips on the fast lane and runs under requirements_llm.txt.
"""

import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from pydantic import BaseModel, Field  # noqa: E402

from llm_utils import (  # noqa: E402
    _row_quality,
    _should_process_line,
    get_context_window,
    read_input_rows,
    validate_llm_output,
)


class DummyEnrichment(BaseModel):
    teater_category: str
    # Enforce a max of 1.0 so that 1.5 triggers the ValidationError fallback block
    confidence_score: float = Field(..., le=1.0)
    extracted_keywords_cs: list[str]

    def category_name(self):
        return self.teater_category


def test_validate_llm_output_success():
    """Standard valid JSON parsing."""
    valid_json = (
        '{"teater_category": "Osoby", "confidence_score": 0.95, "extracted_keywords_cs": ["Jan"]}'
    )
    result = validate_llm_output(valid_json, DummyEnrichment, "doc1", 1, 1)
    assert result["teater_category"] == "Osoby"
    assert result["confidence_score"] == 0.95


def test_validate_llm_output_fallback_recovery():
    """Recovery when strict JSON validation fails but fallback parsing works."""
    # Score is 1.5, which fails the strict Field(le=1.0) check.
    # The helper bounds it to 1.0 during the fallback sequence.
    recoverable_json = (
        '{"teater_category": "Místa", "confidence_score": 1.5, "extracted_keywords_cs": ["Praha"]}'
    )
    result = validate_llm_output(recoverable_json, DummyEnrichment, "doc1", 1, 1)
    assert result["confidence_score"] == 1.0


def test_validate_llm_output_meta_text_clearing():
    """Meta-text correctly strips out keywords."""
    meta_json = '{"teater_category": "Nerelevantní (meta-text)", "confidence_score": 0.9, "extracted_keywords_cs": ["fake"]}'
    result = validate_llm_output(meta_json, DummyEnrichment, "doc1", 1, 1)
    assert result["extracted_keywords_cs"] == []


def test_should_process_line_noise_rejection():
    """The quality filter drops low-quality and 'Trash' lines."""
    should_proc, _ = _should_process_line("Some text", "Empty", 0.30, True, 3, 8, 0.40)
    assert not should_proc

    should_proc, _ = _should_process_line("Good length text", "Trash", 0.80, True, 3, 8, 0.40)
    assert not should_proc


def test_a_row_without_a_quality_score_is_not_trash():
    """A TEITOK document (or a text table) has no line quality. Treating the missing score as
    0.0 turned every such row into "Trash", so a .teitok.xml input enriched nothing (#13 P5.1)."""
    should_proc, _ = _should_process_line("A readable line", "", None, True, 3, 8, 0.40)
    assert should_proc
    should_proc, _ = _should_process_line("ab", "", None, True, 3, 8, 0.40)
    assert not should_proc  # the length rules still apply


def test_row_quality_reads_the_score_or_none():
    assert _row_quality({"quality_score": "0.91"}) == 0.91
    assert _row_quality({"quality_score": 0.2}) == 0.2
    assert _row_quality({"quality_score": ""}) is None
    assert _row_quality({"quality_score": None}) is None
    assert _row_quality({}) is None
    assert _row_quality({"quality_score": "n/a"}) is None


def test_teitok_rows_reach_the_model(tmp_path):
    teitok = tmp_path / "doc.teitok.xml"
    teitok.write_text(
        '<TEI><text><body><pb n="1"/><div><s id="s-1" text="Výzkum proběhl v Praze.">'
        "<tok>Výzkum</tok></s></div></body></text></TEI>",
        encoding="utf-8",
    )
    rows = read_input_rows(teitok)
    assert rows and rows[0]["quality_score"] is None
    should_proc, reason = _should_process_line(
        rows[0]["text"], rows[0]["categ"], _row_quality(rows[0]), True, 3, 8, 0.40
    )
    assert should_proc, reason


def test_get_context_window_formatting():
    """Context windows correctly wrap the target line with <target_line>."""
    rows = [
        {"text": "Line 1", "page_num": 1, "line_num": 1, "categ": ""},
        {"text": "Line 2", "page_num": 1, "line_num": 2, "categ": ""},
        {"text": "Line 3", "page_num": 1, "line_num": 3, "categ": ""},
    ]

    context = get_context_window(rows, center_idx=1, window=1)
    assert "<target_line> >>> [P1 L2] Line 2 </target_line>" in context
    assert "[P1 L1] Line 1" in context
    assert "[P1 L3] Line 3" in context
