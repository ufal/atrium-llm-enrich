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
            {"page": "1", "line": 1, "text": "First line.", "bbox": [10, 10, 100, 20], "categ": "Text"},
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
        enrichment={"items": [{"locator": "should never appear", "extracted_keywords_en": ["bogus"]}]},
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
