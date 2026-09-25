"""
tests/test_doc_to_visual_md.py
==============================
Tests for api_util/doc_to_visual_md.py — the extension dispatcher that routes
DOCX/PDF to the right visually-rich Markdown converter.
"""

import pytest

from api_util import doc_to_visual_md


def test_is_supported_by_extension():
    assert doc_to_visual_md.is_supported("report.docx") is True
    assert doc_to_visual_md.is_supported("report.PDF") is True
    assert doc_to_visual_md.is_supported("report.txt") is False
    assert doc_to_visual_md.is_supported("report") is False


def test_convert_rejects_unsupported_extension(tmp_path):
    p = tmp_path / "report.txt"
    p.write_text("hello", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported input"):
        doc_to_visual_md.convert_to_visual_md(p)


def _stub_json_route(monkeypatch):
    """Record what reached the JSON route (Issue #18 / #10 G8) instead of converting."""
    from api_util import digital_to_json

    called = {}

    def fake_build_record(path, engine="light", **_kwargs):
        called["path"] = str(path)
        called["engine"] = engine
        return {"doc_id": "a"}

    def fake_render(record, title, min_quality=0.0, **_kwargs):
        called["record"] = record
        called["title"] = title
        called["min_quality"] = min_quality
        return f"# {title} via json route\n"

    monkeypatch.setattr(digital_to_json, "build_record", fake_build_record)
    monkeypatch.setattr(doc_to_visual_md.json_to_md, "render_record", fake_render)
    return called


@pytest.mark.parametrize("name", ["a.docx", "a.pdf"])
def test_convert_routes_pdf_and_docx_through_the_json_route(monkeypatch, tmp_path, name):
    """One route: the Markdown the model reads comes from the same record the pipeline keeps.

    Before 2026-09-25 the clients' auto-convert ran the older direct converters (which miss
    the garbled-text-layer check) while the hub E2E ran digital_to_json -> json_to_md.
    """
    called = _stub_json_route(monkeypatch)
    out = doc_to_visual_md.convert_to_visual_md(tmp_path / name, min_quality=0.3, engine="docling")
    assert out == "# a via json route\n"
    assert called["path"].endswith(name)
    assert called["engine"] == "docling"
    assert called["min_quality"] == 0.3


def test_legacy_flag_still_reaches_the_direct_docx_converter(monkeypatch, tmp_path):
    called = {}

    def fake_docx(path):
        called["docx"] = str(path)
        return "# docx md\n"

    monkeypatch.setattr(doc_to_visual_md.docx_to_md, "convert", fake_docx)
    out = doc_to_visual_md.convert_to_visual_md(tmp_path / "a.docx", legacy=True)
    assert out == "# docx md\n"
    assert called["docx"].endswith("a.docx")


def test_ocr_on_a_pdf_keeps_the_tesseract_path(monkeypatch, tmp_path):
    """`--ocr` is the only OCR in this repo until the needs_ocr hand-off takes over; the
    JSON route never OCRs (that would put OCR text under the digital-born originator)."""
    called = {}

    def fake_pdf(path, ocr=False):
        called["pdf"] = str(path)
        called["ocr"] = ocr
        return "# pdf md\n"

    monkeypatch.setattr(doc_to_visual_md.pdf_to_md, "convert", fake_pdf)
    out = doc_to_visual_md.convert_to_visual_md(tmp_path / "a.pdf", ocr=True)
    assert out == "# pdf md\n"
    assert called == {"pdf": str(tmp_path / "a.pdf"), "ocr": True}


def test_json_route_end_to_end_on_a_real_docx(tmp_path):
    """No stubs: a DOCX with a heading and a table renders both (G3)."""
    docx = pytest.importorskip("docx")
    pytest.importorskip("jsonschema")
    document = docx.Document()
    document.add_heading("Nálezová zpráva", level=1)
    document.add_paragraph("Sonda II odkryla val.")
    table = document.add_table(rows=2, cols=2)
    for (r, c), text in {
        (0, 0): "Vrstva",
        (0, 1): "Mocnost",
        (1, 0): "Ornice",
        (1, 1): "30 cm",
    }.items():
        table.cell(r, c).text = text
    path = tmp_path / "report.docx"
    document.save(str(path))

    md = doc_to_visual_md.convert_to_visual_md(path)
    assert "## Page 1" in md
    assert "### Nálezová zpráva" in md
    assert "| Vrstva | Mocnost |" in md and "| Ornice | 30 cm |" in md


def test_is_supported_by_document_json_suffix():
    assert doc_to_visual_md.is_supported("CTX01.document.json") is True
    assert doc_to_visual_md.is_supported("CTX01.categories.json") is False


def test_convert_routes_document_json(monkeypatch, tmp_path):
    called = {}

    def fake_json_to_md(path, min_quality=0.0):
        called["path"] = str(path)
        called["min_quality"] = min_quality
        return "# json md\n"

    monkeypatch.setattr(doc_to_visual_md.json_to_md, "convert", fake_json_to_md)
    out = doc_to_visual_md.convert_to_visual_md(tmp_path / "CTX01.document.json", min_quality=0.5)
    assert out == "# json md\n"
    assert called["path"].endswith("CTX01.document.json")
    assert called["min_quality"] == 0.5
