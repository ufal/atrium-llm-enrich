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


def test_convert_routes_docx(monkeypatch, tmp_path):
    called = {}

    def fake_docx(path):
        called["docx"] = str(path)
        return "# docx md\n"

    monkeypatch.setattr(doc_to_visual_md.docx_to_md, "convert", fake_docx)
    out = doc_to_visual_md.convert_to_visual_md(tmp_path / "a.docx")
    assert out == "# docx md\n"
    assert called["docx"].endswith("a.docx")


def test_convert_routes_pdf(monkeypatch, tmp_path):
    called = {}

    def fake_pdf(path, ocr=False):
        called["pdf"] = str(path)
        called["ocr"] = ocr
        return "# pdf md\n"

    monkeypatch.setattr(doc_to_visual_md.pdf_to_md, "convert", fake_pdf)
    out = doc_to_visual_md.convert_to_visual_md(tmp_path / "a.pdf")
    assert out == "# pdf md\n"
    assert called["pdf"].endswith("a.pdf")


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
