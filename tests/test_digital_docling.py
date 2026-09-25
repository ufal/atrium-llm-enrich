"""Tests for api_util/digital_docling.py — the opt-in heavy PDF engine (`--engine docling`).

The mapping reads Docling's EXPORTED dict, so it is tested here without Docling installed:
against small synthetic IRs that pin each rule, and against the real IR the Phase-0 probe
produced for `digital_born/sample.pdf` (schema 1.10.0) where that file is present. A live
conversion needs Docling's model weights (Hugging Face) and is exercised only by the slow,
skip-guarded test at the end.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from api_util import digital_docling as dd
from api_util import digital_to_json as d2j

REPO = Path(__file__).resolve().parent.parent
SAMPLE_PDF = REPO / "digital_born" / "sample.pdf"
SAMPLE_IR = REPO / "digital_born" / "sample.pdf_docling_ir.json"


def _light(lines, height=792.0):
    page = d2j.DigitalPage(page="1", page_index=1, width=612.0, height=height)
    page.lines = [
        d2j.DigitalLine(page="1", line=i, text=text, bbox=box, heading_level=None)
        for i, (text, box) in enumerate(lines)
    ]
    return d2j.DigitalDocument(
        doc_id="S",
        origin=d2j.ORIGIN_PDF,
        media_type="application/pdf",
        pages=[page],
        sha256="c" * 64,
        filename="S.pdf",
        components=["pdfplumber"],
    )


def _box(left, t, r, b, origin="BOTTOMLEFT"):
    return {"l": left, "t": t, "r": r, "b": b, "coord_origin": origin}


def _ir(texts, tables=(), body=None, furniture=()):
    """A minimal DoclingDocument dict: page 1 of 612x792, items in `body` order."""
    refs = body if body is not None else [f"#/texts/{i}" for i in range(len(texts))]
    return {
        "schema_name": "DoclingDocument",
        "pages": {"1": {"page_no": 1, "size": {"width": 612.0, "height": 792.0}}},
        "body": {"self_ref": "#/body", "children": [{"$ref": r} for r in refs]},
        "furniture": {"self_ref": "#/furniture", "children": [{"$ref": r} for r in furniture]},
        "groups": [],
        "texts": [dict(t, self_ref=f"#/texts/{i}") for i, t in enumerate(texts)],
        "tables": [dict(t, self_ref=f"#/tables/{i}") for i, t in enumerate(tables)],
        "pictures": [],
    }


def test_bottom_left_boxes_become_top_left():
    """`$defs/bbox` is top-left, y down; Docling item boxes are usually BOTTOMLEFT."""
    assert dd.top_left(_box(72, 730, 300, 718), 792.0) == [72.0, 62.0, 300.0, 74.0]
    assert dd.top_left(_box(72, 62, 300, 74, "TOPLEFT"), 792.0) == [72.0, 62.0, 300.0, 74.0]


def test_docling_order_and_labels_win_while_light_geometry_stays():
    """Docling's reading order puts the right column's heading FIRST here; the light lines
    keep their text and exact line boxes, and take the item's group, level and region."""
    light = _light(
        [
            ("Left column text", [72, 100, 250, 112]),
            ("Right heading", [324, 100, 450, 116]),
            ("7", [300, 760, 305, 769]),
        ]
    )
    ir = _ir(
        [
            {
                "label": "section_header",
                "level": 2,
                "text": "Right heading",
                "prov": [{"page_no": 1, "bbox": _box(320, 694, 460, 674)}],
            },
            {
                "label": "text",
                "text": "Left column text",
                "prov": [{"page_no": 1, "bbox": _box(70, 694, 260, 678)}],
            },
            {
                "label": "page_footer",
                "content_layer": "furniture",
                "text": "7",
                "prov": [{"page_no": 1, "bbox": _box(298, 34, 307, 21)}],
            },
        ]
    )
    doc = dd.docling_to_digital(ir, light)
    lines = doc.pages[0].lines
    assert [ln.text for ln in lines] == ["Right heading", "Left column text", "7"]
    assert lines[0].heading_level == 2 and lines[0].bbox == [324, 100, 450, 116]
    assert lines[2].region == "page_footer"
    assert [ln.line for ln in lines] == [0, 1, 2]
    assert doc.engine == "docling"
    assert set(dd.COMPONENTS) <= set(doc.components) and "pdfplumber" in doc.components


def test_unclaimed_light_lines_are_kept_in_place():
    """The heavy engine may reorganise text, never lose it."""
    light = _light([("Known", [72, 100, 200, 112]), ("Stray caption", [72, 300, 200, 312])])
    ir = _ir(
        [
            {
                "label": "text",
                "text": "Known",
                "prov": [{"page_no": 1, "bbox": _box(70, 694, 210, 678)}],
            }
        ]
    )
    lines = dd.docling_to_digital(ir, light).pages[0].lines
    assert [ln.text for ln in lines] == ["Known", "Stray caption"]
    assert lines[1].group_id is None, "left for Layer B to group"


def test_items_without_light_lines_fall_back_to_docling_text():
    light = _light([])
    ir = _ir(
        [
            {
                "label": "title",
                "text": "Only Docling saw this",
                "prov": [{"page_no": 1, "bbox": _box(72, 730, 300, 718)}],
            }
        ]
    )
    [line] = dd.docling_to_digital(ir, light).pages[0].lines
    assert line.text == "Only Docling saw this" and line.heading_level == 1
    assert line.bbox == [72.0, 62.0, 300.0, 74.0]


def test_tables_map_spans_headers_and_cell_lines():
    light = _light(
        [
            ("Nalezy", [80, 104, 130, 114]),
            ("Keramika", [80, 124, 140, 134]),
            ("12 ks", [220, 124, 250, 134]),
        ]
    )
    table = {
        "label": "table",
        "prov": [{"page_no": 1, "bbox": _box(72, 700, 400, 650)}],
        "data": {
            "num_rows": 2,
            "num_cols": 2,
            "table_cells": [
                {
                    "start_row_offset_idx": 0,
                    "end_row_offset_idx": 1,
                    "start_col_offset_idx": 0,
                    "end_col_offset_idx": 2,
                    "column_header": True,
                    "text": "Nalezy",
                    "bbox": _box(72, 100, 400, 118, "TOPLEFT"),
                },
                {
                    "start_row_offset_idx": 1,
                    "end_row_offset_idx": 2,
                    "start_col_offset_idx": 0,
                    "end_col_offset_idx": 1,
                    "text": "Keramika",
                    "bbox": _box(72, 120, 200, 138, "TOPLEFT"),
                },
                {
                    "start_row_offset_idx": 1,
                    "end_row_offset_idx": 2,
                    "start_col_offset_idx": 1,
                    "end_col_offset_idx": 2,
                    "text": "12 ks",
                    "bbox": _box(200, 120, 400, 138, "TOPLEFT"),
                },
            ],
        },
    }
    ir = _ir([], tables=[table], body=["#/tables/0"])
    doc = dd.docling_to_digital(ir, light)
    [grid] = doc.pages[0].tables
    assert (grid.n_rows, grid.n_cols) == (2, 2)
    first = grid.cells[0]
    assert (
        first["colspan"] == 2
        and first["is_header"]
        and first["bbox"] == [72.0, 100.0, 400.0, 118.0]
    )
    assert grid.cells[1]["group_id"] == first["group_id"], "a spanned position repeats the join key"
    by_group = {ln.group_id: ln.text for ln in doc.pages[0].lines}
    assert by_group[first["group_id"]] == "Nalezy" and len(doc.pages[0].lines) == 3


def test_mapped_record_passes_the_output_gate(tmp_path):
    pytest.importorskip("jsonschema")
    light = _light([("Title", [72, 62, 300, 80])])
    ir = _ir(
        [
            {
                "label": "title",
                "text": "Title",
                "prov": [{"page_no": 1, "bbox": _box(70, 732, 310, 710)}],
            }
        ]
    )
    doc = d2j.normalize(dd.docling_to_digital(ir, light))
    record, pages, lines = d2j.to_record(doc, out_dir=str(tmp_path), strict=True)
    data = d2j._gate(record, pages, lines)
    assert data["lines"][0]["style"]["heading_level"] == 1


@pytest.mark.skipif(
    not (SAMPLE_PDF.exists() and SAMPLE_IR.exists()), reason="digital_born/ sample absent"
)
def test_real_docling_ir_over_real_light_lines(tmp_path):
    """The Phase-0 probe's real IR (schema 1.10.0) over the light engine's lines for the
    same PDF: furniture and headings come through, no light text is duplicated, and the
    record validates. The IR was produced with Docling's OCR ON, so it carries text from
    inside figures a live `--engine docling` run (OCR off) would not."""
    pytest.importorskip("pdfplumber")
    pytest.importorskip("jsonschema")
    ir = json.loads(SAMPLE_IR.read_text(encoding="utf-8"))
    light = d2j.extract_pdf(str(SAMPLE_PDF))
    light_texts = [" ".join(ln.text.split()) for ln in light.all_lines()]
    doc = d2j.normalize(dd.docling_to_digital(ir, light))
    texts = [" ".join(ln.text.split()) for ln in doc.all_lines()]
    for text in set(light_texts):
        assert texts.count(text) >= light_texts.count(text), f"light text lost: {text!r}"
    assert any(ln.region == "page_footer" for ln in doc.all_lines())
    assert any(ln.heading_level for ln in doc.all_lines())
    record, pages, lines = d2j.to_record(doc, out_dir=str(tmp_path), strict=True)
    d2j._gate(record, pages, lines)


@pytest.mark.slow
def test_live_docling_conversion(tmp_path):
    """Needs Docling AND its model weights (DOCLING_ARTIFACTS_PATH or a reachable Hub)."""
    pytest.importorskip("docling")
    fixtures = tmp_path / "fx"
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "make_fixtures", REPO / "tests" / "fixtures" / "digital" / "make_fixtures.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fixtures.mkdir()
    (fixtures / "two_column.pdf").write_bytes(module.two_column_pdf())
    try:
        data = d2j.build_record(str(fixtures / "two_column.pdf"), engine="docling")
    except RuntimeError as exc:
        pytest.skip(f"Docling models unavailable here: {exc}")
    texts = [ln["text"] for ln in data["lines"]]
    assert "Sonda II odkryla val." in texts
