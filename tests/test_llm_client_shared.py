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


# ── document-record input routing (2026-09-06) ───────────────────────────────
#
# Regression tests for atrium-project run 34039707673. api_util.doc_to_visual_md
# accepted `*.document.json` from the day it was written, but the gate in front of
# it tested `Path.suffix in DOC_CONVERT_EXTENSIONS` — and a record's suffix is
# ".json". So conversion was skipped, the dispatch fell through to the line-level
# branch, csv.DictReader was handed a JSON file, and the run produced zero records,
# zero errors and no API call while exiting 0.


def _record(tmp_path, name="minimal.document.json"):
    p = tmp_path / name
    p.write_text(
        json.dumps(
            {
                "doc_id": "minimal",
                "source": {"origin": "digital-born-pdf"},
                "pages": [{"page": 1}],
                "lines": [{"page": 1, "line": 1, "text": "Náčrt sondy."}],
            }
        ),
        encoding="utf-8",
    )
    return p


class TestDocumentRecordInput:
    def test_record_is_recognised_as_convertible(self, tmp_path):
        assert lcs.is_convertible_input(_record(tmp_path))

    def test_record_is_converted_to_markdown(self, tmp_path):
        """The behaviour the whole bug came down to: the returned path must have a
        `.md` suffix, because that is what selects the document-level branch."""
        out = lcs.prepare_document_input(_record(tmp_path))
        assert out.suffix == ".md"
        assert "Náčrt sondy." in out.read_text(encoding="utf-8")

    def test_cache_name_drops_the_whole_compound_suffix(self, tmp_path):
        """`Path.stem` would give "minimal.document.md" and desynchronise the cache
        name from the doc_id canonical_doc_id() derives from the same file."""
        assert lcs.prepare_document_input(_record(tmp_path)).name == "minimal.md"

    def test_unrelated_json_is_not_treated_as_a_record(self, tmp_path):
        """Matching on ".json" rather than the compound suffix would sweep in every
        JSON file the pipeline is ever pointed at."""
        p = tmp_path / "junk.json"
        p.write_text("{}", encoding="utf-8")
        assert not lcs.is_convertible_input(p)
        assert lcs.prepare_document_input(p) == p

    def test_csv_is_returned_unchanged(self, tmp_path):
        p = tmp_path / "doc.csv"
        p.write_text("page,line\n", encoding="utf-8")
        assert lcs.prepare_document_input(p) == p


@pytest.mark.parametrize(
    "name,readable",
    [
        ("a.csv", True),
        ("b.teitok.xml", True),
        ("c.md", True),
        ("d.txt", True),
        ("e.pdf", True),
        ("f.docx", True),
        ("g.document.json", True),
        ("junk.json", False),
        ("notes.yaml", False),
        ("scan.png", False),
    ],
)
def test_has_reader_matrix(name, readable):
    """The dispatch has no else branch, so anything answering True here must be
    readable by one of them — and anything answering False is refused up front
    rather than silently enriching nothing."""
    assert lcs.has_reader(Path(name)) is readable


# ── outcome classification (atrium-project#49) ───────────────────────────────
#
# `processed == 0` is not one situation, it is three, and until #49 the record could not
# tell them apart: a model that was asked and located nothing, a model whose every call
# failed, and a document that never reached the model at all. `attempted` is the counter
# that splits the first from the third; `skipped_error`/`aborted` split out the second.


def test_document_level_counts_the_attempt_even_when_it_finds_nothing(tmp_path):
    doc_path = tmp_path / "empty_verdict.md"
    doc_path.write_text("# doc1\n\nBlock one, line one.\n", encoding="utf-8")

    DocModel = lcs.build_document_schema(["kostel"])
    results, stats = lcs.run_document_level(
        doc_path, lambda _m: json.dumps({"items": []}), "system prompt", DocModel
    )

    assert results == []
    assert stats["processed"] == 0
    assert stats["attempted"] == 1, "the model WAS consulted; only `attempted` records that"
    assert lcs.classify_outcome(results, stats) == lcs.OUTCOME_EMPTY
    assert lcs.contributes_document_record(results, stats) is True


def test_document_level_counts_the_attempt_even_when_the_call_raises(tmp_path):
    doc_path = tmp_path / "boom.md"
    doc_path.write_text("# doc1\n\nBlock one, line one.\n", encoding="utf-8")

    def _boom(_messages):
        raise RuntimeError("simulated backend failure")

    DocModel = lcs.build_document_schema(["kostel"])
    results, stats = lcs.run_document_level(doc_path, _boom, "system prompt", DocModel)

    assert stats["attempted"] == 1
    assert lcs.classify_outcome(results, stats) == lcs.OUTCOME_FAILED
    assert lcs.contributes_document_record(results, stats) is False


def test_line_level_never_asks_when_every_row_is_filtered_out(tmp_path):
    """A CSV whose rows all fail the quality filter is NOT an empty enrichment."""
    csv_path = tmp_path / "all_trash.csv"
    csv_path.write_text(
        "file_id,page_num,line_num,categ,quality_score,text\n"
        "doc1,1,1,Trash,0.01,x\n"
        "doc1,1,2,Trash,0.01,y\n",
        encoding="utf-8",
    )

    Model = lcs.build_schema(["kostel"])
    results, stats = lcs.run_line_level(csv_path, _fake_line_chat_fn, "system prompt", Model)

    assert results == []
    assert stats["attempted"] == 0
    assert stats["skipped_filter"] == 2
    assert lcs.classify_outcome(results, stats) == lcs.OUTCOME_NOT_ASKED
    assert lcs.contributes_document_record(results, stats) is False


def test_partial_success_is_still_a_contribution():
    """Nine good rows and one error is a record worth writing, not a failed run."""
    results = [{"file_id": "doc1"}]
    stats = {"processed": 1, "skipped_filter": 0, "skipped_error": 1, "aborted": 0, "attempted": 2}

    assert lcs.classify_outcome(results, stats) == lcs.OUTCOME_CONTRIBUTED
    assert lcs.contributes_document_record(results, stats) is True


def test_enrichment_block_of_an_empty_run_is_schema_valid(tmp_path):
    """The empty block has to survive Layer D, or writing it would trade one bug for another.

    write_document_record() is the repo's single write chokepoint and it RAISES on a record
    of its own that does not validate, so reaching a written file at all is the assertion;
    the explicit validate_document() below just says so out loud.
    """
    from atrium_document import load_document, validate_document

    assert lcs.enrichment_block("doc1", []) == {"items": []}

    path = lcs.write_document_record("doc1", [], tmp_path)
    assert path is not None, "an empty enrichment must still produce a record"

    record = load_document(str(path))
    validate_document(record)
    assert record["enrichment"] == {"items": []}
    assert record["assembled"]["blocks"]["enrichment"]["program"] == "llm-enrich"


# ── repairing a non-conforming document reply (atrium-project run 34123820218) ──
#
# The digital smoke's first run against enrichable content got five items whose CONTENT
# was right — "sonda, valove teleso", "zlomky keramiky, rany stredovek" — and whose every
# field was the wrong type, so all five were thrown away and the document aborted. The
# request goes out as plain `json_object` (a 4719-value enum is too big for the
# json_schema variant on most providers), so the shape is enforced by the prompt, and the
# document-level prompt had no worked example at all.

#: gpt-4o-mini's actual reply from that run, reconstructed field for field from the
#: pydantic errors in the job log. Keeping it verbatim is the point: this is not an
#: invented edge case, it is what the model does.
_RUN_34123820218_REPLY = json.dumps(
    {
        "items": [
            {
                "locator": "Lokalita: hradiste u Horni Mezi",
                "page": 1,
                "extracted_keywords_cs": "hradiste, lokalita, Beroun",
                "extracted_keywords_en": "fortress, site, Beroun",
                "teater_category": "hradiště",
                "confidence_score": 0.9,
            },
            {
                "locator": "Sonda II odkryla cast",
                "page": 1,
                "extracted_keywords_cs": "sonda, valove teleso",
                "extracted_keywords_en": "probe, rampart structure",
                "teater_category": "sonda",
                "confidence_score": 0.85,
            },
        ]
    }
)

_TERMS = [lcs.META_TERM, "hradiště", "sonda"]


def test_strict_validation_really_does_reject_that_reply():
    """The premise. If this ever starts passing, the repair pass below is dead weight."""
    with pytest.raises(ValidationError):
        lcs.build_document_schema(_TERMS).model_validate_json(_RUN_34123820218_REPLY)


def test_repair_recovers_the_shape_the_model_actually_returns():
    model = lcs.build_document_schema(_TERMS)
    validated, dropped = lcs._repair_document_response(_RUN_34123820218_REPLY, model, "enrichable")

    assert dropped == []
    first, second = validated.items
    assert first.page == "1", "int page must become the string the schema documents"
    assert first.extracted_keywords_cs == ["hradiste", "lokalita", "Beroun"]
    assert first.extracted_keywords_en == ["fortress", "site", "Beroun"]
    assert second.category_name() == "sonda"


def test_repair_resolves_a_near_miss_category_but_never_invents_one():
    """Case and whitespace are recovered; a term the vocabulary does not contain is not.

    Mapping an unlisted category onto some "closest" listed one would put a claim in the
    record that no vocabulary supports — worse than dropping the item, because the record
    is what goes to FAIR catalogue export.
    """
    model = lcs.build_document_schema(_TERMS)
    reply = json.dumps(
        {
            "items": [
                {
                    "locator": "a",
                    "page": "1",
                    "extracted_keywords_cs": [],
                    "extracted_keywords_en": [],
                    "teater_category": "  SONDA  ",
                    "confidence_score": 0.5,
                },
                {
                    "locator": "b",
                    "page": "1",
                    "extracted_keywords_cs": [],
                    "extracted_keywords_en": [],
                    "teater_category": "a category nobody listed",
                    "confidence_score": 0.5,
                },
            ]
        }
    )

    validated, dropped = lcs._repair_document_response(reply, model, "doc1")

    assert [item.category_name() for item in validated.items] == ["sonda"]
    assert len(dropped) == 1
    assert "a category nobody listed" in dropped[0]


def test_repair_clamps_confidence_and_drops_a_non_numeric_one():
    model = lcs.build_document_schema(_TERMS)
    reply = json.dumps(
        {
            "items": [
                {
                    "locator": "a",
                    "page": None,
                    "extracted_keywords_cs": [],
                    "extracted_keywords_en": [],
                    "teater_category": "sonda",
                    "confidence_score": 1.4,
                },
                {
                    "locator": "b",
                    "page": None,
                    "extracted_keywords_cs": [],
                    "extracted_keywords_en": [],
                    "teater_category": "sonda",
                    "confidence_score": "very sure",
                },
            ]
        }
    )

    validated, dropped = lcs._repair_document_response(reply, model, "doc1")

    assert [item.confidence_score for item in validated.items] == [1.0]
    assert len(dropped) == 1 and "very sure" in dropped[0]


@pytest.mark.parametrize(
    "reply",
    ["not json at all", '{"no_items_key": 1}', '{"items": "a string"}'],
    ids=["unparseable", "no-items", "items-not-a-list"],
)
def test_repair_refuses_a_reply_that_is_not_merely_misshapen(reply):
    """Coercion is for formatting slips. A reply with no items array is a real failure and
    must reach run_document_level's error handler rather than becoming a silent empty."""
    model = lcs.build_document_schema(_TERMS)
    with pytest.raises((ValueError, json.JSONDecodeError)):
        lcs._repair_document_response(reply, model, "doc1")


def test_run_document_level_repairs_and_reports_it_in_stats(tmp_path):
    """End to end: the run that aborted in CI now completes and contributes."""
    doc_path = tmp_path / "enrichable.md"
    doc_path.write_text("# enrichable\n\n## Page 1\n\nSonda II.\n", encoding="utf-8")

    model = lcs.build_document_schema(_TERMS)
    results, stats = lcs.run_document_level(
        doc_path, lambda _m: _RUN_34123820218_REPLY, "system prompt", model
    )

    assert stats["aborted"] == 0 and stats["skipped_error"] == 0
    assert stats["processed"] == 2
    assert stats["repaired"] == 1 and stats["dropped_items"] == 0
    assert lcs.classify_outcome(results, stats) == lcs.OUTCOME_CONTRIBUTED
    assert results[0]["enrichment"]["extracted_keywords_en"] == ["fortress", "site", "Beroun"]


def test_a_conforming_reply_does_not_go_through_the_repair_pass(tmp_path):
    """The fast path must stay untouched — `repaired` is absent when nothing was repaired."""
    doc_path = tmp_path / "clean.md"
    doc_path.write_text("# clean\n\n## Page 1\n\nSonda II.\n", encoding="utf-8")

    reply = json.dumps(
        {
            "items": [
                {
                    "locator": "Sonda II",
                    "page": "1",
                    "extracted_keywords_cs": ["sonda"],
                    "extracted_keywords_en": ["trench"],
                    "teater_category": "sonda",
                    "confidence_score": 0.9,
                }
            ]
        }
    )
    results, stats = lcs.run_document_level(
        doc_path, lambda _m: reply, "system prompt", lcs.build_document_schema(_TERMS)
    )

    assert stats["processed"] == 1
    assert "repaired" not in stats


def test_document_prompt_ships_a_worked_example_and_names_every_field():
    """The other half of the fix. The header used to say the four remaining fields had the
    "same meaning as the single-line task" — a task the document prompt never shows."""
    prompt, _terms = lcs.build_document_system_prompt(
        {"theme": {"a": {"cs": "sonda", "en": "trench"}}}, max_tokens=100_000
    )

    assert "EXAMPLE OF THE REQUIRED OUTPUT SHAPE" in prompt
    assert "same meaning as the single-line task" not in prompt
    assert "NEVER a single comma-separated string" in prompt
    assert '"page": "2"' in prompt, "the example must show page QUOTED"


# ── the category the model actually picks (atrium-project run 34125325468) ────
#
# Second live run against enrichable content, second distinct wrong answer. Every item
# came back as `teater_category: 'Artefact / druh předmětu'` — which is not an invented
# term at all: it is one of _render_vocab_prompt()'s OWN section headings, printed as
# `--- Artefact / druh předmětu ---` above the bullets it groups. The prompt showed the
# model two kinds of line and never said which kind was selectable.


def test_a_section_heading_is_never_resolved_to_a_term_and_says_why():
    """A heading names a whole section; picking any member of it would be a guess.

    The drop reason has to NAME the mistake, because the fix is in the prompt (which now
    says headings are not categories) and a bare "not in the vocabulary" would send the
    next reader looking at the vocabulary file instead.
    """
    model = lcs.build_document_schema([lcs.META_TERM, "sonda"])
    reply = json.dumps(
        {
            "items": [
                {
                    "locator": "Sonda II",
                    "page": 1,
                    "extracted_keywords_cs": ["sonda"],
                    "extracted_keywords_en": ["trench"],
                    "teater_category": "sonda",
                    "confidence_score": 0.9,
                },
                {
                    "locator": "x",
                    "page": 1,
                    "extracted_keywords_cs": [],
                    "extracted_keywords_en": [],
                    "teater_category": "Artefact / druh předmětu",
                    "confidence_score": 0.9,
                },
            ]
        }
    )

    validated, dropped = lcs._repair_document_response(reply, model, "enrichable")

    assert [item.category_name() for item in validated.items] == ["sonda"]
    assert len(dropped) == 1
    assert "SECTION HEADING" in dropped[0]


def test_a_whole_vocabulary_bullet_resolves_to_its_czech_term():
    """The vocabulary prints `- sonda (trench)`; a model that copies the bullet is close
    enough to recover, and the english gloss is not part of the term."""
    model = lcs.build_document_schema([lcs.META_TERM, "sonda"])
    reply = json.dumps(
        {
            "items": [
                {
                    "locator": "Sonda II",
                    "page": "1",
                    "extracted_keywords_cs": [],
                    "extracted_keywords_en": [],
                    "teater_category": "sonda (trench)",
                    "confidence_score": 0.9,
                }
            ]
        }
    )

    validated, dropped = lcs._repair_document_response(reply, model, "doc1")

    assert dropped == []
    assert validated.items[0].category_name() == "sonda"


def test_a_term_that_legitimately_ends_in_a_parenthetical_is_not_mangled():
    """The guard on the gloss-stripping step. Many real terms end in a parenthesised
    qualifier — META_TERM itself, 'atlantik (paleoklimatologie)' — and they must match
    before any stripping is attempted."""
    model = lcs.build_document_schema([lcs.META_TERM, "atlantik (paleoklimatologie)"])
    for term in (lcs.META_TERM, "atlantik (paleoklimatologie)"):
        reply = json.dumps(
            {
                "items": [
                    {
                        "locator": "x",
                        "page": "1",
                        "extracted_keywords_cs": [],
                        "extracted_keywords_en": [],
                        "teater_category": term,
                        "confidence_score": 1.0,
                    }
                ]
            }
        )
        validated, dropped = lcs._repair_document_response(reply, model, "doc1")
        assert dropped == []
        assert validated.items[0].category_name() == term


def test_a_reply_where_nothing_survives_aborts_rather_than_reporting_empty(tmp_path):
    """The vacuity guard, and the reason this is not "assert less".

    An `enrichment: {items: []}` block means "the model looked and there was nothing
    here". Emitting that when the model in fact found five passages we could not read
    would put a false verdict in the record AND turn the digital smoke green while it
    enriched nothing. So it becomes an inference error, which under atrium-project#49's
    contract means no document record and a non-zero exit — red, at the right step.
    """
    doc_path = tmp_path / "enrichable.md"
    doc_path.write_text("# enrichable\n\n## Page 1\n\nSonda II.\n", encoding="utf-8")

    reply = json.dumps(
        {
            "items": [
                {
                    "locator": f"item {i}",
                    "page": 1,
                    "extracted_keywords_cs": [],
                    "extracted_keywords_en": [],
                    "teater_category": "Artefact / druh předmětu",
                    "confidence_score": 0.9,
                }
                for i in range(5)
            ]
        }
    )
    model = lcs.build_document_schema([lcs.META_TERM, "sonda"])

    results, stats = lcs.run_document_level(doc_path, lambda _m: reply, "prompt", model)

    assert results == []
    assert stats["aborted"] == 1 and stats["skipped_error"] == 1
    assert lcs.classify_outcome(results, stats) == lcs.OUTCOME_FAILED
    assert lcs.contributes_document_record(results, stats) is False


def test_an_honestly_empty_reply_is_still_an_empty_verdict(tmp_path):
    """The other side of the guard: {"items": []} must NOT be swept up by it.

    That reply validates strictly, never reaches the repair pass, and stays
    OUTCOME_EMPTY — a stamped empty block, which is what the smoke's minimal.pdf case
    legitimately produces.
    """
    doc_path = tmp_path / "nothing.md"
    doc_path.write_text("# nothing\n\nBlock one, line one.\n", encoding="utf-8")
    model = lcs.build_document_schema([lcs.META_TERM, "sonda"])

    results, stats = lcs.run_document_level(
        doc_path, lambda _m: json.dumps({"items": []}), "prompt", model
    )

    assert results == [] and stats["aborted"] == 0 and stats["skipped_error"] == 0
    assert lcs.classify_outcome(results, stats) == lcs.OUTCOME_EMPTY
    assert lcs.contributes_document_record(results, stats) is True


def test_document_prompt_says_headings_are_not_categories():
    prompt, _terms = lcs.build_document_system_prompt(
        {"Artefact": {"a": {"cs": "sonda", "en": "trench", "sub": "druh předmětu"}}},
        max_tokens=100_000,
    )

    assert "SECTION HEADING and is NOT a category" in prompt
    assert "never return one" in prompt
    assert "'- sonda (trench)' the correct value is exactly 'sonda'" in prompt
