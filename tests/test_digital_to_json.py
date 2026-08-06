"""Tests for api_util/digital_to_json.py — the `digital-convert` originator (hub issue #18 §1a).

WHAT IS TESTED WITHOUT THE HEAVY STACK, AND WHY THAT IS THE RIGHT SPLIT.
`requirements_digital.txt` (docling, pdfplumber, python-docx) is deliberately outside the
base install, so the fast lane cannot exercise Layer A. That is fine, because Layer A is the
only layer that does not carry contract risk: it reads a file format. Everything the
accretion contract can be violated by lives in B, C and D, and all of it is pure Python:

  * **B** — the decode-sanity gate. This is the whole reason the converter exists (a PDF
    whose text layer extracts successfully and is wrong), and it is testable from strings.
  * **C** — block ownership. Writing a field this program is not granted, or using
    `set_block` on a field-split block, is how a co-contributor's data gets erased.
  * **D** — the output gate. A record that fails the round-trip assertion or schema
    validation must NOT be written; "written then flagged" is a different, worse contract.

Layer A gets its own tests, `importorskip`-ed, so they run wherever the stack is installed
and skip cleanly where it is not — rather than being absent and untested everywhere.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from api_util import digital_to_json as d2j

# ──────────────────────────────────────────────────────────────────────────────
# Layer B — the decode-sanity gate
# ──────────────────────────────────────────────────────────────────────────────


def test_confusion_table_is_derived_not_guessed():
    """The table must contain the CP1250/CP1252 pairs, and NOT the ones that coincide.

    `á` (0xE1) and `é` (0xE9) decode identically under both codecs, so a mis-decoded page
    keeps them — which is exactly why "Zpráva" survives in the observed corruption while
    "sondě" does not. If either appeared here the table would be over-eager and would
    condemn clean Czech text.
    """
    assert d2j.CP1250_MISREADS["ì"] == "ě"
    assert d2j.CP1250_MISREADS["è"] == "č"
    assert d2j.CP1250_MISREADS["ø"] == "ř"
    assert "á" not in d2j.CP1250_MISREADS
    assert "é" not in d2j.CP1250_MISREADS


def test_clean_czech_text_scores_perfect():
    report = d2j.decode_sanity("Zpráva o sondě číslo 3. Nalezeny hřeby.")
    assert report.score == 1.0
    assert report.suspicious == 0
    assert report.recovered is None
    assert not report.is_garbage


def test_mojibake_is_detected_and_recovered():
    """The observed real-world corruption, end to end.

    Recovery is asserted because it is the proof the diagnosis is RIGHT: if the text really
    is CP1250 bytes read as CP1252, re-encoding round-trips to the intended string exactly.
    Anything less would mean the heuristic had merely found suspicious characters.
    """
    garbled = "Zpráva o sondì èíslo 3. Nalezeny høeby, vrstva ornice mìla"
    report = d2j.decode_sanity(garbled)
    assert report.is_garbage
    assert report.suspicious == 4
    assert report.recovered == "Zpráva o sondě číslo 3. Nalezeny hřeby, vrstva ornice měla"


def test_score_is_a_ratio_so_long_and_short_lines_compare():
    """A caption and a page must be scorable on the same axis (`quality_score` is 0–1)."""
    short = d2j.decode_sanity("sondì")
    long = d2j.decode_sanity("sondì " + "cistá " * 40)
    assert short.score < long.score
    assert 0.0 <= short.score <= 1.0 and 0.0 <= long.score <= 1.0


def test_one_foreign_character_does_not_condemn_a_clean_line():
    """The failure mode is systematic; a single legitimate `ì` must stay above the cut."""
    report = d2j.decode_sanity(
        "Vrstva ornice, srov. italsky citta ma pekne pole a ratio " * 2 + "ì"
    )
    assert not report.is_garbage


def test_digits_and_punctuation_only_is_not_garbage():
    """No letters means nothing to judge — must not divide by zero or report a verdict."""
    report = d2j.decode_sanity("123 -- 45.6 (7)")
    assert report.score == 1.0
    assert report.letters == 0


def test_garbage_line_gets_the_exact_spelling_json_to_md_filters_on():
    """`categ` is an open string, so a synonym silently disables DROP_CATEGORIES."""
    line = d2j.DigitalLine(page="1", line=0, text="sondì èíslo")
    categ = d2j.classify_line(line, d2j.decode_sanity(line.text))
    assert categ == "Garbage"

    from api_util.json_to_md import DROP_CATEGORIES

    assert categ in DROP_CATEGORIES


def test_inverted_wins_over_garbage():
    """Mirrored text's extracted string is not evidence, so a decode verdict on it is noise."""
    line = d2j.DigitalLine(page="1", line=0, text="sondì èíslo", inverted=True)
    assert d2j.classify_line(line, d2j.decode_sanity(line.text)) == "Inverted"


def test_paragraph_grouping_from_vertical_gaps():
    """2 pt within a paragraph vs 34 pt between — the measured fixture geometry."""
    lines = [
        d2j.DigitalLine(page="1", line=0, text="one", bbox=[72, 62, 275, 74]),
        d2j.DigitalLine(page="1", line=1, text="two", bbox=[72, 76, 274, 88]),
        d2j.DigitalLine(page="1", line=2, text="far", bbox=[72, 122, 256, 134]),
    ]
    d2j.assign_group_ids(lines)
    assert lines[0].group_id == lines[1].group_id
    assert lines[2].group_id != lines[1].group_id


def test_lines_without_geometry_are_left_ungrouped():
    """DOCX has no coordinates; inventing groups from nothing is worse than absence."""
    lines = [d2j.DigitalLine(page="1", line=i, text=f"p{i}") for i in range(3)]
    d2j.assign_group_ids(lines)
    assert all(line.group_id is None for line in lines)


def test_needs_ocr_is_never_set_without_its_reason():
    """The two mean OPPOSITE things on the two originator paths; the reason disambiguates."""
    page = d2j.DigitalPage(page="1", page_index=1)
    page.lines = [d2j.DigitalLine(page="1", line=0, text="sondì èíslo høeby mìla")]
    d2j.normalize(
        d2j.DigitalDocument(doc_id="D", origin=d2j.ORIGIN_PDF, media_type="", pages=[page])
    )
    assert page.needs_ocr is True
    assert page.needs_ocr_reason
    assert "does not decode" in page.needs_ocr_reason
    # ...and it must not claim the page has no text layer, which is false here by definition.
    assert "no extractable text" not in page.needs_ocr_reason.lower()


def test_clean_page_is_not_routed_to_ocr():
    page = d2j.DigitalPage(page="1", page_index=1)
    page.lines = [d2j.DigitalLine(page="1", line=0, text="Zpráva o sondě číslo 3.")]
    d2j.normalize(
        d2j.DigitalDocument(doc_id="D", origin=d2j.ORIGIN_PDF, media_type="", pages=[page])
    )
    assert page.needs_ocr is False
    assert page.quality_band == "Clear"


# ──────────────────────────────────────────────────────────────────────────────
# Layer C — serialization and block ownership
# ──────────────────────────────────────────────────────────────────────────────


def _document(with_table: bool = False, garbled: bool = False) -> d2j.DigitalDocument:
    text = "sondì èíslo høeby" if garbled else "Zpráva o sondě"
    page = d2j.DigitalPage(page="1", page_index=1, width=595.0, height=842.0, unit="pt")
    page.lines = [
        d2j.DigitalLine(page="1", line=0, text=text, bbox=[72, 62, 275, 74], bold=True),
        d2j.DigitalLine(page="1", line=1, text="Vrstva ornice.", bbox=[72, 76, 274, 88]),
    ]
    if with_table:
        page.tables = [
            d2j.DigitalTable(
                table_id="t0",
                page="1",
                n_rows=1,
                n_cols=2,
                group_id="tbl0",
                cells=[{"row": 0, "col": 0, "is_header": True, "group_id": "tbl0-r0c0"}],
            )
        ]
    return d2j.normalize(
        d2j.DigitalDocument(
            doc_id="CTX000000001",
            origin=d2j.ORIGIN_PDF,
            media_type="application/pdf",
            pages=[page],
            sha256="a" * 64,
            filename="CTX000000001.pdf",
        )
    )


def test_record_is_authorised_by_source_origin(tmp_path):
    """§1a: the plane is written by whoever `source.origin` authorises — and it must resolve.

    An origin no `ORIGIN_ORIGINATORS` prefix matches makes the check ABSTAIN, which is how
    §1a silently stops applying to a whole class of documents. So this asserts the routing,
    not merely the string.
    """
    from atrium_document import resolve_originator

    record, _, _ = to_record_in(tmp_path, _document())
    data = record.to_dict()
    assert data["source"]["origin"] == d2j.ORIGIN_PDF
    assert resolve_originator(data["source"]["origin"]) == "digital-convert"


def to_record_in(tmp_path, doc, baseline=None):
    return d2j.to_record(doc, baseline=baseline, run_id="r1", out_dir=str(tmp_path), strict=True)


def test_lines_survive_the_round_trip_with_text_and_bbox(tmp_path):
    """The §1b regression: a grant of only `["group_id"]` silently dropped text and bbox.

    `merge_block()` filters unowned fields SILENTLY, and `lines[]` requires only page+line,
    so the resulting text-free rows validated clean. This is the assertion that catches it.
    """
    record, _, line_rows = to_record_in(tmp_path, _document())
    written = {(r["page"], r["line"]): r for r in record.to_dict()["lines"]}
    for row in line_rows:
        got = written[(row["page"], row["line"])]
        assert got["text"] == row["text"]
        assert got["bbox"] == row["bbox"]
    record.assert_fields_survived("lines", line_rows)


def test_style_is_semantic_only(tmp_path):
    """bold/italic/heading_level are actionable; typeface and point size are not."""
    record, _, _ = to_record_in(tmp_path, _document())
    style = record.to_dict()["lines"][0]["style"]
    assert style == {"bold": True}
    assert "font" not in style and "size" not in style


def test_canvas_unit_is_written_whenever_a_bbox_is(tmp_path):
    record, _, _ = to_record_in(tmp_path, _document())
    canvas = record.to_dict()["pages"][0]["canvas"]
    assert canvas["unit"] == "pt"
    assert canvas["width"] == 595.0 and canvas["height"] == 842.0


def test_no_canvas_without_geometry(tmp_path):
    """A DOCX-shaped document has no coordinates, so `canvas` must be absent, not faked."""
    page = d2j.DigitalPage(page="1", page_index=1)
    page.lines = [d2j.DigitalLine(page="1", line=0, text="Zpráva")]
    doc = d2j.normalize(
        d2j.DigitalDocument(doc_id="D", origin=d2j.ORIGIN_DOCX, media_type="", pages=[page])
    )
    record, _, _ = to_record_in(tmp_path, doc)
    assert "canvas" not in record.to_dict()["pages"][0]


def test_garbage_lines_are_kept_as_rows_but_excluded_from_content(tmp_path):
    """The row stays (with its `categ`, so a reader can see what was rejected and why);
    the mojibake must not be concatenated into `content.text` as if it were readable."""
    record, _, _ = to_record_in(tmp_path, _document(garbled=True))
    data = record.to_dict()
    assert data["lines"][0]["categ"] == "Garbage"
    assert "sondì" not in (data["content"]["text"] or "")
    assert "Vrstva ornice." in data["content"]["text"]


def test_accretion_preserves_another_tools_blocks(tmp_path):
    """Rule 2: everything this program does not own is deep-copied through untouched."""
    baseline = tmp_path / "CTX000000001.document.json"
    baseline.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "record_type": "atrium-document",
                "doc_id": "CTX000000001",
                "page_categories": {"1": "DRAW"},
                "pages": [{"page": "1", "category": "DRAW", "category_confidence": 0.9}],
                "assembled": {"blocks": {"page_categories": {"program": "page-classification"}}},
            }
        ),
        encoding="utf-8",
    )
    record, _, _ = to_record_in(tmp_path, _document(), baseline=str(baseline))
    data = record.to_dict()

    assert data["page_categories"] == {"1": "DRAW"}
    # ...and the co-owned pages[] row keeps page-classification's fields alongside ours.
    page = data["pages"][0]
    assert page["category"] == "DRAW" and page["category_confidence"] == 0.9
    assert page["quality_band"] == "Clear" and page["page_index"] == 1


def test_tables_block_is_written_when_present(tmp_path):
    record, _, _ = to_record_in(tmp_path, _document(with_table=True))
    tables = record.to_dict()["tables"]
    assert tables[0]["table_id"] == "t0"
    assert tables[0]["cells"][0]["group_id"] == "tbl0-r0c0"


# ──────────────────────────────────────────────────────────────────────────────
# Layer D — the output gate
# ──────────────────────────────────────────────────────────────────────────────


def test_valid_record_is_written_and_validates(tmp_path):
    pytest.importorskip("jsonschema")
    record, page_rows, line_rows = to_record_in(tmp_path, _document())
    written = d2j.emit(record, page_rows, line_rows)
    data = json.loads(Path(written).read_text(encoding="utf-8"))
    assert data["doc_id"] == "CTX000000001"
    assert data["assembled"]["blocks"]["lines"]["program"] == "digital-convert"


def test_invalid_record_is_not_written_at_all(tmp_path):
    """ "No doc.json is emitted if validation fails" — not written-then-flagged.

    Also asserts no stray `.tmp` survives: `finalize()` writes to a temp path and renames,
    so a gate that fired after the write would leave one behind.
    """
    jsonschema = pytest.importorskip("jsonschema")
    doc = _document()
    record, page_rows, line_rows = to_record_in(tmp_path, doc)
    record._data["pages"][0]["quality_score"] = "not-a-number"  # schema says number

    # The specific exception, not a blind `Exception`: the point of this test is that the
    # SCHEMA rejected the record, and a bare Exception would pass just as happily on a typo
    # in emit() itself.
    with pytest.raises(jsonschema.exceptions.ValidationError):
        d2j.emit(record, page_rows, line_rows)

    assert list(tmp_path.glob("*.document.json")) == []
    assert list(tmp_path.glob("*.tmp")) == []


def test_field_ownership_drop_is_caught_before_validation(tmp_path):
    """A row stripped of its `text` is a VALID row (lines[] requires only page+line),
    so the schema cannot catch this class — only the round-trip assertion can."""
    record, page_rows, line_rows = to_record_in(tmp_path, _document())
    record._data["lines"][0].pop("text")
    with pytest.raises(RuntimeError, match="dropped by merge_block"):
        d2j.emit(record, page_rows, line_rows)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def test_cli_exposes_the_accretion_flags():
    help_text = d2j.build_parser().format_help()
    for flag in ("--document-json", "--out", "--doc-id", "--run-id", "--paradata-ref"):
        assert flag in help_text


def test_unsupported_format_fails_loudly(tmp_path):
    stray = tmp_path / "notes.txt"
    stray.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported input"):
        d2j.extract(str(stray))


def test_missing_optional_dependency_is_advice_not_a_traceback(tmp_path, monkeypatch):
    """Exit 2 with an install hint beats an ImportError stack for a missing optional dep."""
    pdf = tmp_path / "x.pdf"
    pdf.write_bytes(b"%PDF-1.4\n")

    def _raise(*_args, **_kwargs):
        raise d2j._missing("pdfplumber", "pdfplumber")

    monkeypatch.setattr(d2j, "extract_pdf", _raise)
    assert d2j.main([str(pdf)]) == 2


# ──────────────────────────────────────────────────────────────────────────────
# Layer A — skipped wherever requirements_digital.txt is not installed
# ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def digital_fixtures(tmp_path):
    """Generate the pinned byte-exact fixtures (minimal.pdf, garbled.pdf, minimal.docx)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "make_fixtures", Path(__file__).parent / "fixtures" / "digital" / "make_fixtures.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    blobs = module.build_all()
    for name, payload in blobs.items():
        (tmp_path / name).write_bytes(payload)
    return tmp_path


def test_pdf_extraction_reads_top_left_coordinates(digital_fixtures):
    pytest.importorskip("pdfplumber")
    doc = d2j.extract_pdf(str(digital_fixtures / "minimal.pdf"))
    assert doc.origin == d2j.ORIGIN_PDF
    assert doc.pages and doc.pages[0].lines
    x0, top, x1, bottom = doc.pages[0].lines[0].bbox
    # Top-left origin: `top` is smaller than `bottom`. The bottom-up y0/y1 pair would
    # invert this, and is the specific mistake $defs/bbox calls out.
    assert top < bottom and x0 < x1


def test_garbled_pdf_is_routed_to_ocr(digital_fixtures):
    """The fixture reproduces the real corruption; the gate must catch it end to end."""
    pytest.importorskip("pdfplumber")
    doc = d2j.normalize(d2j.extract_pdf(str(digital_fixtures / "garbled.pdf")))
    assert any(page.needs_ocr for page in doc.pages)
    assert any(line.categ == "Garbage" for line in doc.all_lines())


def test_docx_extraction_carries_no_geometry(digital_fixtures):
    pytest.importorskip("docx")
    doc = d2j.extract_docx(str(digital_fixtures / "minimal.docx"))
    assert doc.origin == d2j.ORIGIN_DOCX
    assert all(line.bbox is None for line in doc.all_lines())
