"""
tests/test_json_to_md.py — tests for api_util/json_to_md.py, the AtriumDocument
JSON -> Markdown converter (issue #13, closing the `regenerable.markdown` loop).
"""

import json

import pytest

from api_util import json_to_md


def _write_record(tmp_path, doc_id="CTX01", **blocks):
    record = {"schema_version": "1.0", "doc_id": doc_id, **blocks}
    path = tmp_path / f"{doc_id}.document.json"
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def test_convert_renders_page_sectioned_markdown_with_bbox(tmp_path):
    path = _write_record(
        tmp_path,
        pages=[{"page": "1", "canvas": {"width": 612, "height": 792, "unit": "pt"}}],
        lines=[
            {
                "page": "1",
                "line": 1,
                "text": "First line.",
                "bbox": [10, 10, 100, 20],
                "categ": "Text",
            },
            {"page": "1", "line": 2, "text": "Second line.", "categ": "Text"},
        ],
    )
    md = json_to_md.convert(path)
    assert "## Page 1" in md
    assert "DOC_META" in md
    assert "First line." in md
    assert "Second line." in md
    assert "BBOX" in md


def test_convert_drops_garbage_and_inverted_lines(tmp_path):
    path = _write_record(
        tmp_path,
        pages=[{"page": "1"}],
        lines=[
            {"page": "1", "line": 1, "text": "Good line.", "categ": "Text"},
            {"page": "1", "line": 2, "text": "Junk line.", "categ": "Garbage"},
            {"page": "1", "line": 3, "text": "Flipped line.", "categ": "Inverted"},
        ],
    )
    md = json_to_md.convert(path)
    assert "Good line." in md
    assert "Junk line." not in md
    assert "Flipped line." not in md


def test_convert_min_quality_drops_low_score_lines(tmp_path):
    path = _write_record(
        tmp_path,
        pages=[{"page": "1"}],
        lines=[
            {"page": "1", "line": 1, "text": "Clear line.", "quality_score": 0.95},
            {"page": "1", "line": 2, "text": "Noisy line.", "quality_score": 0.4},
        ],
    )
    md = json_to_md.convert(path, min_quality=0.8)
    assert "Clear line." in md
    assert "Noisy line." not in md


def test_convert_emits_needs_ocr_and_ocr_cues(tmp_path):
    path = _write_record(
        tmp_path,
        pages=[
            {"page": "1", "needs_ocr": True},
            {"page": "2", "ocr": {"engine": "tesseract", "lang": "ces"}},
        ],
        lines=[
            {"page": "1", "line": 1, "text": "Scanned page text."},
            {"page": "2", "line": 1, "text": "OCR'd page text."},
        ],
    )
    md = json_to_md.convert(path)
    assert "NEEDS_OCR" in md
    assert "OCR: engine=tesseract" in md


def test_convert_falls_back_to_content_text_when_no_lines(tmp_path, capsys):
    path = _write_record(tmp_path, content={"text": "Whole-document fallback text."})
    md = json_to_md.convert(path)
    assert "Whole-document fallback text." in md
    assert "falling back" in capsys.readouterr().err


def test_convert_raises_when_nothing_to_render(tmp_path):
    path = _write_record(tmp_path)
    with pytest.raises(ValueError, match="nothing to render"):
        json_to_md.convert(path)


def test_convert_never_reads_enrichment_block(tmp_path):
    """The self-feeding guard: enrichment is llm-enrich's OWN prior output, and
    must never be read back as source text on a second run."""
    path = _write_record(
        tmp_path,
        lines=[{"page": "1", "line": 1, "text": "Real source text."}],
        enrichment={
            "items": [{"locator": "should never appear", "extracted_keywords_en": ["bogus"]}]
        },
    )
    md = json_to_md.convert(path)
    assert "Real source text." in md
    assert "should never appear" not in md
    assert "bogus" not in md


def test_convert_rejects_unimplemented_detail_profile(tmp_path):
    path = _write_record(tmp_path, lines=[{"page": "1", "line": 1, "text": "x"}])
    with pytest.raises(NotImplementedError, match="standard"):
        json_to_md.convert(path, detail="standard")


def test_read_document_rows_shape(tmp_path):
    path = _write_record(
        tmp_path,
        pages=[{"page": "1", "canvas": {"width": 100, "height": 200}}],
        lines=[{"page": "1", "line": 1, "text": "hi", "bbox": [1, 2, 3, 4]}],
    )
    rows, pages, fallback = json_to_md.read_document_rows(path)
    assert rows == [{"page_num": 1, "line_num": 1, "text": "hi", "bbox": [1, 2, 3, 4]}]
    assert pages == {1: {"width": 100, "height": 200}}
    assert fallback is None


# ──────────────────────────────────────────────────────────────────────────────
# Issue #18 — the consumer half of the contracts the schema states
#
# Each of these pins a claim that atrium_document.schema.json makes normatively and that
# nothing in the code honoured, so the field was inert (group_id, needs_ocr_reason) or the
# rendered cue was factually wrong (unit, page label).
# ──────────────────────────────────────────────────────────────────────────────


def test_group_id_change_renders_a_paragraph_break(tmp_path):
    """The schema's `lines[].group_id` CONSUMER CONTRACT: "a group boundary renders as a
    paragraph break ... Markdown already encodes paragraphs natively as a blank line".

    `_rows_from_lines` used to build {page_num, line_num, text} (+bbox) and drop everything
    else, so the paragraph fidelity the converter extracts never reached the model — and the
    schema named a function that had no group tracking in it.
    """
    path = _write_record(
        tmp_path,
        lines=[
            {"page": "1", "line": 0, "text": "Predmluva", "group_id": "p1"},
            {"page": "1", "line": 1, "text": "same paragraph", "group_id": "p1"},
            {"page": "1", "line": 2, "text": "new paragraph", "group_id": "p2"},
        ],
    )
    body = json_to_md.convert(path).split("## Page 1", 1)[1]
    assert "Predmluva\nsame paragraph\n\nnew paragraph" in body


def test_absent_group_id_is_byte_identical_to_before(tmp_path):
    """The ALTO path has no group_id anywhere, and its output must not shift by one byte."""
    lines = [
        {"page": "1", "line": 0, "text": "one"},
        {"page": "1", "line": 1, "text": "two"},
        {"page": "2", "line": 0, "text": "three"},
    ]
    md = json_to_md.convert(_write_record(tmp_path, lines=lines))
    assert (
        md == "# CTX01\n\n## Page 1\n\none\ntwo\n<!-- PAGE_BREAK: pg_2 -->\n\n## Page 2\n\nthree\n"
    )


def test_group_boundary_at_a_page_boundary_does_not_double_space(tmp_path):
    """The first text row of a page must never emit a boundary — the `## Page N` header
    already supplies the separation, and `None` is a legitimate group_id."""
    path = _write_record(
        tmp_path,
        lines=[
            {"page": "1", "line": 0, "text": "end of page one", "group_id": "p1"},
            {"page": "2", "line": 0, "text": "start of page two", "group_id": "p2"},
        ],
    )
    assert "\n\n\nstart of page two" not in json_to_md.convert(path)


def test_non_numeric_page_labels_survive(tmp_path):
    """The schema promises `page` is a string "so 'iv' or 'A-1' survive". `int(line["page"])`
    with a bare `continue` deleted those lines instead, then convert() died with
    ValueError("nothing to render") blaming upstream stages for text they had produced."""
    path = _write_record(
        tmp_path,
        pages=[{"page": "iv", "page_index": 1}, {"page": "A-1", "page_index": 2}],
        lines=[
            {"page": "iv", "line": 0, "text": "roman front matter"},
            {"page": "A-1", "line": 0, "text": "appendix"},
        ],
    )
    md = json_to_md.convert(path)
    assert "## Page iv" in md and "## Page A-1" in md
    assert "roman front matter" in md and "appendix" in md
    # page_index, not the label, decides the order.
    assert md.index("roman front matter") < md.index("appendix")


def test_page_index_orders_pages_when_labels_do_not(tmp_path):
    path = _write_record(
        tmp_path,
        pages=[{"page": "ii", "page_index": 2}, {"page": "i", "page_index": 1}],
        lines=[
            {"page": "ii", "line": 0, "text": "second"},
            {"page": "i", "line": 0, "text": "first"},
        ],
    )
    md = json_to_md.convert(path)
    assert md.index("first") < md.index("second")


def test_canvas_unit_is_carried_into_doc_meta(tmp_path):
    """DOC_META hardcoded "px", so every digital-born PDF page (which is in POINTS) told the
    model it was that many pixels — and api_util/pdf_to_md.py emitted "pt" for the same
    document, so the two front-ends disagreed on one cue."""
    path = _write_record(
        tmp_path,
        pages=[{"page": "1", "canvas": {"width": 595, "height": 842, "unit": "pt"}}],
        lines=[{"page": "1", "line": 0, "text": "x"}],
    )
    assert "size=595x842pt" in json_to_md.convert(path)


def test_missing_unit_still_renders_px(tmp_path):
    """Legacy ALTO records carry no unit; the default must not change."""
    path = _write_record(
        tmp_path,
        pages=[{"page": "1", "canvas": {"width": 100, "height": 200}}],
        lines=[{"page": "1", "line": 0, "text": "x"}],
    )
    assert "size=100x200px" in json_to_md.convert(path)


def test_needs_ocr_reason_replaces_the_inverted_default(tmp_path):
    """needs_ocr means OPPOSITE things on the two paths. With no field to carry the
    distinction, every digital-born page rendered "no extractable text layer" — false by
    definition for a document that HAS a text layer that merely decodes to garbage."""
    path = _write_record(
        tmp_path,
        pages=[
            {
                "page": "1",
                "needs_ocr": True,
                "needs_ocr_reason": "text layer decodes to corrupt diacritics (no /ToUnicode)",
            }
        ],
        lines=[{"page": "1", "line": 0, "text": "sondI"}],
    )
    md = json_to_md.convert(path)
    assert "corrupt diacritics" in md
    assert "no extractable text layer" not in md


def test_needs_ocr_without_a_reason_keeps_the_old_default(tmp_path):
    path = _write_record(
        tmp_path,
        pages=[{"page": "1", "needs_ocr": True}],
        lines=[{"page": "1", "line": 0, "text": "x"}],
    )
    assert "no extractable text layer" in json_to_md.convert(path)


def test_page_ordinals_uses_int_labels_unchanged(tmp_path):
    """Case 1 of page_ordinals: an all-numeric document behaves exactly as before, which is
    what keeps every pre-#18 record's output stable."""
    assert json_to_md.page_ordinals([], [{"page": "10"}, {"page": "2"}]) == {"10": 10, "2": 2}


def test_page_ordinals_falls_back_to_document_order(tmp_path):
    """Case 3: non-numeric labels and no page_index — first appearance wins, pages[] first."""
    ordinals = json_to_md.page_ordinals(
        [{"page": "cover"}, {"page": "iv"}], [{"page": "iv"}, {"page": "extra"}]
    )
    assert ordinals == {"cover": 1, "iv": 2, "extra": 3}
