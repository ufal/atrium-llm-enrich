"""
tests/test_llm_client_shared.py
================================
Tests for llm_client_shared.py — the shared front-end duplicated (by design,
see that module's docstring) from llm_utils.py/llm_run.py for the
remote/lightweight-local backends (openrouter_client.py, ollama_client.py).

These tests exist to catch exactly the failure mode the module's own
docstring warns about: "Kept in sync BY HAND with llm_utils.py / llm_run.py.
If you change the quality filter, the context-window builder, or the
archaeological system prompt over there, mirror the change here." A parity
drift between the two copies would otherwise only surface as a silent
behavioural difference between backends in production.
"""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

import llm_client_shared as lcs
from llm_client_shared import (
    approx_token_count,
    build_document_schema,
    build_schema,
    get_context_window,
    load_config,
    run_document_level,
    run_line_level,
    should_process_line,
    validate_llm_output,
)

# ── load_config ──────────────────────────────────────────────────────────────


def test_load_config_parses_key_value_pairs(tmp_path):
    cfg_file = tmp_path / "test_config.txt"
    cfg_file.write_text(
        "# a comment\n\nMODEL_KEY=qwen-3.6-27b-it\nOPENROUTER_API_KEY = sk-or-abc \n",
        encoding="utf-8",
    )
    config = load_config(str(cfg_file))
    assert config["MODEL_KEY"] == "qwen-3.6-27b-it"
    # both key and value are stripped of surrounding whitespace
    assert config["OPENROUTER_API_KEY"] == "sk-or-abc"


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(str(tmp_path / "does_not_exist.txt"))


# ── approx_token_count ───────────────────────────────────────────────────────


def test_approx_token_count_is_character_based():
    # _CHARS_PER_TOKEN_ESTIMATE = 4, floor division, minimum of 1
    assert approx_token_count("") == 1
    assert approx_token_count("ab") == 1
    assert approx_token_count("a" * 8) == 2
    assert approx_token_count("a" * 401) == 100


# ── should_process_line — parity target: llm_utils._should_process_line ────


def test_should_process_line_noise_rejection():
    should_proc, _ = should_process_line("Some text", "Empty", 0.30, True, 3, 8, 0.40)
    assert not should_proc

    should_proc, _ = should_process_line("Good length text", "Trash", 0.80, True, 3, 8, 0.40)
    assert not should_proc


def test_should_process_line_low_quality_score_forces_trash():
    # quality_score < 0.40 always downgrades categ to "Trash" regardless of the
    # original categ value, then Trash is always skipped.
    should_proc, reason = should_process_line(
        "Reasonably long text", "Text", 0.10, True, 3, 8, 0.40
    )
    assert not should_proc
    assert "Trash" in reason


def test_should_process_line_mid_quality_score_forces_noisy_but_keeps():
    # 0.40 <= quality_score < 0.70 downgrades to "Noisy", which is NOT in
    # _ALWAYS_SKIP_CATEG, so a long-enough line still passes.
    should_proc, _ = should_process_line(
        "Reasonably long text here", "Text", 0.55, True, 3, 8, 0.40
    )
    assert should_proc


def test_should_process_line_empty_text_always_rejected():
    should_proc, reason = should_process_line("", "Text", 0.95, True, 3, 8, 0.40)
    assert not should_proc
    assert reason == "empty text"


def test_should_process_line_non_text_respects_include_flag():
    should_proc, _ = should_process_line("012/345", "Non-text", 0.95, False, 3, 8, 0.40)
    assert not should_proc

    # Long enough + high enough alpha ratio + include_non_text=True -> passes
    should_proc, _ = should_process_line("archaeological find", "Non-text", 0.95, True, 3, 8, 0.40)
    assert should_proc


def test_should_process_line_non_text_alpha_ratio_gate():
    # "12345678" is 8 chars (meets min_char_non_text) but 0% alphabetic.
    should_proc, reason = should_process_line("12345678", "Non-text", 0.95, True, 3, 8, 0.40)
    assert not should_proc
    assert "alpha ratio" in reason


def test_should_process_line_unknown_categ_uses_min_char_count():
    should_proc, _ = should_process_line("ab", "", 0.95, True, 3, 8, 0.40)
    assert not should_proc

    should_proc, _ = should_process_line("abcd", "", 0.95, True, 3, 8, 0.40)
    assert should_proc


# ── get_context_window — parity target: llm_utils.get_context_window ───────


def test_get_context_window_wraps_target_line():
    rows = [
        {"text": "Line 1", "page_num": 1, "line_num": 1, "categ": ""},
        {"text": "Line 2", "page_num": 1, "line_num": 2, "categ": ""},
        {"text": "Line 3", "page_num": 1, "line_num": 3, "categ": ""},
    ]
    context = get_context_window(rows, center_idx=1, window=1)
    assert "<target_line> >>> [P1 L2] Line 2 </target_line>" in context
    assert "[P1 L1] Line 1" in context
    assert "[P1 L3] Line 3" in context


def test_get_context_window_excludes_other_pages():
    rows = [
        {"text": "Page 1 line", "page_num": 1, "line_num": 1, "categ": ""},
        {"text": "Page 2 target", "page_num": 2, "line_num": 1, "categ": ""},
        {"text": "Page 2 line 2", "page_num": 2, "line_num": 2, "categ": ""},
    ]
    context = get_context_window(rows, center_idx=1, window=2)
    assert "<target_line> >>> [P2 L1] Page 2 target </target_line>" in context
    # Row 0 is on a different page than the target and is not the target itself
    assert "Page 1 line" not in context


def test_get_context_window_skips_noise_neighbours():
    rows = [
        {"text": "Trash neighbour", "page_num": 1, "line_num": 1, "categ": "Trash"},
        {"text": "Target line", "page_num": 1, "line_num": 2, "categ": ""},
        {"text": "Clean neighbour", "page_num": 1, "line_num": 3, "categ": ""},
    ]
    context = get_context_window(rows, center_idx=1, window=1)
    assert "Trash neighbour" not in context
    assert "Clean neighbour" in context


# ── validate_llm_output — parity target: llm_utils.validate_llm_output ─────


class _DummyEnrichment:
    """Minimal stand-in mimicking the Pydantic models build_schema() produces."""

    def __init__(self, teater_category, confidence_score, extracted_keywords_cs):
        self.teater_category = teater_category
        self.confidence_score = confidence_score
        self.extracted_keywords_cs = extracted_keywords_cs

    def category_name(self):
        return self.teater_category

    def model_dump(self):
        return {
            "teater_category": self.teater_category,
            "confidence_score": self.confidence_score,
            "extracted_keywords_cs": self.extracted_keywords_cs,
        }

    @classmethod
    def model_validate_json(cls, data):
        d = json.loads(data)
        if d.get("confidence_score", 0) > 1.0:
            raise ValidationError.from_exception_data("DummyEnrichment", [])
        return cls(d["teater_category"], d["confidence_score"], d.get("extracted_keywords_cs", []))

    @classmethod
    def model_validate(cls, d):
        return cls(d["teater_category"], d["confidence_score"], d.get("extracted_keywords_cs", []))


def test_validate_llm_output_success():
    valid_json = (
        '{"teater_category": "kostel", "confidence_score": 0.95, "extracted_keywords_cs": ["Jan"]}'
    )
    result = validate_llm_output(valid_json, _DummyEnrichment, "doc1", 1, 1)
    assert result["teater_category"] == "kostel"
    assert result["confidence_score"] == 0.95


def test_validate_llm_output_fallback_clamps_confidence():
    # confidence_score=1.5 fails strict model_validate_json (per _DummyEnrichment's
    # simulated Field(le=1.0)); the fallback path clamps it into [0, 1].
    recoverable_json = (
        '{"teater_category": "kostel", "confidence_score": 1.5, "extracted_keywords_cs": ["x"]}'
    )
    result = validate_llm_output(recoverable_json, _DummyEnrichment, "doc1", 1, 1)
    assert result["confidence_score"] == 1.0


def test_validate_llm_output_meta_text_clears_keywords():
    meta_json = (
        '{"teater_category": "Nerelevantn\\u00ed (meta-text)", '
        '"confidence_score": 0.9, "extracted_keywords_cs": ["fake"]}'
    )
    result = validate_llm_output(meta_json, _DummyEnrichment, "doc1", 1, 1)
    assert result["extracted_keywords_cs"] == []


# ── build_schema / build_document_schema ────────────────────────────────────


def test_build_schema_rejects_empty_term_list():
    with pytest.raises(ValueError):
        build_schema([])


def test_build_schema_constrains_category_enum():
    Model = build_schema(["kostel", "Nerelevantn\u00ed (meta-text)"])
    instance = Model.model_validate(
        {
            "extracted_keywords_cs": ["z\u00e1klady"],
            "extracted_keywords_en": ["foundations"],
            "teater_category": "kostel",
            "confidence_score": 0.9,
        }
    )
    assert instance.category_name() == "kostel"
    with pytest.raises(ValidationError):
        Model.model_validate(
            {
                "extracted_keywords_cs": [],
                "extracted_keywords_en": [],
                "teater_category": "not_in_vocabulary",
                "confidence_score": 0.5,
            }
        )


def test_build_document_schema_defaults_to_empty_items():
    Model = build_document_schema(["kostel"])
    instance = Model.model_validate({})
    assert instance.items == []


def test_build_document_schema_exposes_optional_page():
    Model = build_document_schema(["kostel"])
    # page is optional (defaults to None) and accepts string labels
    item_fields = Model.model_fields["items"].annotation
    inst = Model.model_validate(
        {"items": [{"locator": "x", "teater_category": "kostel", "confidence_score": 0.5}]}
    )
    assert inst.items[0].page is None
    inst2 = Model.model_validate(
        {
            "items": [
                {"locator": "x", "page": "iv", "teater_category": "kostel", "confidence_score": 0.5}
            ]
        }
    )
    assert inst2.items[0].page == "iv"
    assert item_fields is not None  # schema built without error


# ── run_line_level / run_document_level — end-to-end with a fake chat_fn ───


def _fake_line_chat_fn(messages):
    return json.dumps(
        {
            "extracted_keywords_cs": ["z\u00e1klady"],
            "extracted_keywords_en": ["foundations"],
            "teater_category": "kostel",
            "confidence_score": 0.9,
        }
    )


def test_run_line_level_processes_csv(tmp_path):
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text(
        "file_id,page_num,line_num,categ,quality_score,text\n"
        "doc1,1,1,Text,0.95,Vyzkum odhalil zaklady kostela.\n",
        encoding="utf-8",
    )
    Model = build_schema(["kostel"])
    results, stats = run_line_level(csv_path, _fake_line_chat_fn, "system prompt", Model)
    assert stats["processed"] == 1
    assert stats["skipped_error"] == 0
    assert results[0]["enrichment"]["teater_category"] == "kostel"


def test_run_line_level_aborts_after_consecutive_errors(tmp_path):
    csv_path = tmp_path / "sample.csv"
    rows = "\n".join(f"doc1,1,{i},Text,0.95,line number {i} text" for i in range(1, 4))
    csv_path.write_text(
        "file_id,page_num,line_num,categ,quality_score,text\n" + rows + "\n", encoding="utf-8"
    )

    def _broken_chat_fn(messages):
        raise RuntimeError("simulated backend failure")

    Model = build_schema(["kostel"])
    results, stats = run_line_level(
        csv_path, _broken_chat_fn, "system prompt", Model, max_consecutive_errors=2
    )
    assert results == []
    assert stats["aborted"] == 1
    assert stats["skipped_error"] == 2


def test_run_document_level_returns_located_items(tmp_path):
    doc_path = tmp_path / "sample.md"
    doc_path.write_text(
        "# doc1\n\n## Page 1\n\nVyzkum odhalil zaklady kostela.\n", encoding="utf-8"
    )

    def _fake_doc_chat_fn(messages):
        return json.dumps(
            {
                "items": [
                    {
                        "locator": "zaklady kostela",
                        "page": "1",
                        "extracted_keywords_cs": ["z\u00e1klady"],
                        "extracted_keywords_en": ["foundations"],
                        "teater_category": "kostel",
                        "confidence_score": 0.9,
                    }
                ]
            }
        )

    DocModel = build_document_schema(["kostel"])
    results, stats = run_document_level(doc_path, _fake_doc_chat_fn, "system prompt", DocModel)
    assert stats["processed"] == 1
    assert results[0]["locator"] == "zaklady kostela"
    assert results[0]["page"] == "1"  # surfaced at top level for [Source: \u2026, Page N]
    assert "page" not in results[0]["enrichment"]  # not duplicated inside enrichment
    assert results[0]["enrichment"]["teater_category"] == "kostel"


# \u2500\u2500 prepare_document_input (auto-convert seam) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500


def test_prepare_document_input_passthrough_for_native(tmp_path):
    from llm_client_shared import prepare_document_input

    for name in ("a.md", "b.txt", "c.csv", "d.teitok.xml"):
        p = tmp_path / name
        p.write_text("x", encoding="utf-8")
        assert prepare_document_input(p) == p  # unchanged


def test_prepare_document_input_converts_and_caches(tmp_path, monkeypatch):
    import llm_client_shared

    calls = []

    def fake_convert(path, ocr=False):
        calls.append((str(path), ocr))
        return f"# {Path(path).stem}\n\n## Page 1\n\nbody\n"

    # convert_to_visual_md is imported lazily inside the helper from this module.
    import api_util.doc_to_visual_md as dv

    monkeypatch.setattr(dv, "convert_to_visual_md", fake_convert)

    src = tmp_path / "report.pdf"
    src.write_bytes(b"%PDF-1.4 dummy")
    out = llm_client_shared.prepare_document_input(src)
    assert out.name == "report.md"
    assert out.parent.name == "_visual_md_cache"
    assert out.read_text(encoding="utf-8").startswith("# report")
    assert len(calls) == 1

    # Idempotent: cached .md newer than source \u2192 no re-conversion.
    out2 = llm_client_shared.prepare_document_input(src)
    assert out2 == out
    assert len(calls) == 1


# ── config quoting + vocabulary guards (2026-08-19) ──────────────────────────
#
# Regression tests for the e2e Stage 5 failure in ufal/atrium-project run
# 32208408456. The hub workflow wrote `VOCAB_PATH="/workspace/work/….json"`;
# load_config kept the quote characters, so the path could not exist,
# VocabularyManager auto-synced, and the resulting enum held one term. Every
# correct answer the model produced was then rejected as a validation error and
# the run exited 0 having enriched nothing.


class TestLoadConfigQuoting:
    """Quoted and bare values must parse identically — nlp-enrich's sibling
    parser has always stripped quotes, and the same hand writes both configs."""

    def _write(self, tmp_path, body):
        p = tmp_path / "cfg.txt"
        p.write_text(body, encoding="utf-8")
        return str(p)

    def test_double_quoted_value_is_unquoted(self, tmp_path):
        cfg = self._write(tmp_path, 'VOCAB_PATH="/workspace/work/union_nested.json"\n')
        assert lcs.load_config(cfg)["VOCAB_PATH"] == "/workspace/work/union_nested.json"

    def test_single_quoted_value_is_unquoted(self, tmp_path):
        cfg = self._write(tmp_path, "VOCAB_PATH='/a/b.json'\n")
        assert lcs.load_config(cfg)["VOCAB_PATH"] == "/a/b.json"

    def test_bare_value_is_untouched(self, tmp_path):
        cfg = self._write(tmp_path, "VOCAB_PATH=/a/b.json\n")
        assert lcs.load_config(cfg)["VOCAB_PATH"] == "/a/b.json"

    def test_inner_apostrophe_survives(self, tmp_path):
        """Only a matched OUTER pair is stripped."""
        cfg = self._write(tmp_path, "NOTE=it's fine\n")
        assert lcs.load_config(cfg)["NOTE"] == "it's fine"

    def test_unbalanced_quote_is_left_alone(self, tmp_path):
        cfg = self._write(tmp_path, 'ODD="unbalanced\n')
        assert lcs.load_config(cfg)["ODD"] == '"unbalanced'

    def test_empty_value_does_not_crash(self, tmp_path):
        cfg = self._write(tmp_path, "EMPTY=\n")
        assert lcs.load_config(cfg)["EMPTY"] == ""


class TestVocabularyReachedTheModel:
    """build_schema must refuse a term list carrying no real vocabulary."""

    def test_empty_term_list_is_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            lcs.build_schema([])

    def test_meta_only_term_list_is_rejected(self):
        """The exact shape run 32208408456 produced: the fixed administrative
        label and nothing else. Accepting it forces every line to that label."""
        with pytest.raises(ValueError, match="no terms beyond"):
            lcs.build_schema([lcs.META_TERM])

    def test_meta_only_is_rejected_for_document_schema_too(self):
        with pytest.raises(ValueError, match="no terms beyond"):
            lcs.build_document_schema([lcs.META_TERM])

    def test_one_real_term_alongside_meta_is_accepted(self):
        """The guard is exact, not a size threshold — one real term is a
        legitimate (if small) vocabulary and must still build."""
        model = lcs.build_schema([lcs.META_TERM, "sonda"])
        assert model is not None
