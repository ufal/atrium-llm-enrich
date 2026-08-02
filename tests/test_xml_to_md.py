"""
tests/test_xml_to_md.py
========================
Tests for api_util/xml_to_md.py — the TEITOK/ALTO -> Markdown/plain-text
converter that feeds the whole-document input path (BACKEND=openrouter/
ollama, .md/.txt input) via run_document_level() in llm_client_shared.py.
"""

from api_util.xml_to_md import (
    _read_alto_rows,
    _read_teitok_layout,
    convert,
    is_alto,
    read_document_rows,
    rows_to_markdown,
    rows_to_plain_text,
)

TEITOK_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<teiCorpus>
    <text>
        <pb n="1"/>
        <s text="Prvni veta na strance.">
            <tok id="w-1">Prvni</tok>
            <tok id="w-2">veta</tok>
        </s>
        <lb/>
        <s text="Druha veta.">
            <tok id="w-3">Druha</tok>
        </s>
        <pb n="2"/>
        <s text="Veta na druhe strane.">
            <tok id="w-4">Veta</tok>
        </s>
    </text>
</teiCorpus>
"""

ALTO_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<alto xmlns="http://www.loc.gov/standards/alto/ns-v4#">
  <Layout>
    <Page ID="page_1" PHYSICAL_IMG_NR="1">
      <PrintSpace>
        <TextBlock ID="block_1">
          <TextLine ID="line_1">
            <String CONTENT="Vyzkum"/>
            <String CONTENT="odhalil"/>
          </TextLine>
          <TextLine ID="line_2">
            <String CONTENT="zaklady"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
    <Page ID="page_2" PHYSICAL_IMG_NR="2">
      <PrintSpace>
        <TextBlock ID="block_2">
          <TextLine ID="line_3">
            <String CONTENT="kostela"/>
          </TextLine>
        </TextBlock>
      </PrintSpace>
    </Page>
  </Layout>
</alto>
"""

NON_XML_SAMPLE = "<?xml version=\"1.0\"?>\n<some_other_format><thing/></some_other_format>\n"


# ── is_alto ──────────────────────────────────────────────────────────────────


def test_is_alto_true_for_namespaced_alto(tmp_path):
    p = tmp_path / "sample.xml"
    p.write_text(ALTO_SAMPLE, encoding="utf-8")
    assert is_alto(p) is True


def test_is_alto_false_for_teitok(tmp_path):
    p = tmp_path / "sample.teitok.xml"
    p.write_text(TEITOK_SAMPLE, encoding="utf-8")
    assert is_alto(p) is False


def test_is_alto_false_for_unrelated_xml(tmp_path):
    p = tmp_path / "sample.xml"
    p.write_text(NON_XML_SAMPLE, encoding="utf-8")
    assert is_alto(p) is False


def test_is_alto_false_for_malformed_xml(tmp_path):
    p = tmp_path / "broken.xml"
    p.write_text("<not><valid", encoding="utf-8")
    assert is_alto(p) is False


# ── _read_alto_rows ──────────────────────────────────────────────────────────


def test_read_alto_rows_extracts_pages_lines_text(tmp_path):
    p = tmp_path / "sample.alto.xml"
    p.write_text(ALTO_SAMPLE, encoding="utf-8")
    rows = _read_alto_rows(p)

    assert [r["page_num"] for r in rows] == [1, 1, 2]
    assert [r["line_num"] for r in rows] == [1, 2, 1]
    assert rows[0]["text"] == "Vyzkum odhalil"
    assert rows[1]["text"] == "zaklady"
    assert rows[2]["text"] == "kostela"


def test_read_alto_rows_row_shape_matches_teitok_rows(tmp_path):
    """The converter's own docstring promises the ALTO reader mirrors
    teitok_read.read_teitok_rows()'s {"page_num", "line_num", "text"} shape —
    this is what lets rows_to_markdown()/rows_to_plain_text() handle both
    formats identically."""
    p = tmp_path / "sample.alto.xml"
    p.write_text(ALTO_SAMPLE, encoding="utf-8")
    rows = _read_alto_rows(p)
    assert set(rows[0].keys()) == {"page_num", "line_num", "text"}


# ── read_document_rows dispatch ──────────────────────────────────────────────


def test_read_document_rows_dispatches_teitok_by_extension(tmp_path):
    p = tmp_path / "sample.teitok.xml"
    p.write_text(TEITOK_SAMPLE, encoding="utf-8")
    rows = read_document_rows(p)
    assert len(rows) == 3
    assert rows[0]["page_num"] == 1
    assert rows[2]["page_num"] == 2


def test_read_document_rows_dispatches_alto_by_content(tmp_path):
    # Deliberately generic filename — dispatch must go by sniffing content,
    # not by filename pattern, since it doesn't end in .teitok.xml.
    p = tmp_path / "sample.xml"
    p.write_text(ALTO_SAMPLE, encoding="utf-8")
    rows = read_document_rows(p)
    assert len(rows) == 3
    assert rows[0]["text"] == "Vyzkum odhalil"


# ── rows_to_markdown ─────────────────────────────────────────────────────────


def test_rows_to_markdown_page_sections_and_title():
    rows = [
        {"page_num": 1, "line_num": 1, "text": "First line"},
        {"page_num": 1, "line_num": 2, "text": "Second line"},
        {"page_num": 2, "line_num": 1, "text": "Third line"},
    ]
    md = rows_to_markdown(rows, title="doc1")
    assert md.startswith("# doc1")
    assert "## Page 1" in md
    assert "## Page 2" in md
    # Page 2 heading must come after both page-1 lines
    assert md.index("## Page 2") > md.index("Second line")


def test_rows_to_markdown_empty_rows():
    assert rows_to_markdown([], title="doc1") == "# doc1\n"
    assert rows_to_markdown([]) == ""


def test_rows_to_markdown_skips_blank_text():
    rows = [
        {"page_num": 1, "line_num": 1, "text": "  "},
        {"page_num": 1, "line_num": 2, "text": "Real content"},
    ]
    md = rows_to_markdown(rows, title="doc1")
    # The blank-text row contributes no content line of its own — output is
    # exactly the title, one page heading, and the single real content line.
    assert md == "# doc1\n\n## Page 1\n\nReal content\n"


# ── rows_to_plain_text ───────────────────────────────────────────────────────


def test_rows_to_plain_text_inserts_blank_line_on_page_break():
    rows = [
        {"page_num": 1, "line_num": 1, "text": "Page one line"},
        {"page_num": 2, "line_num": 1, "text": "Page two line"},
    ]
    txt = rows_to_plain_text(rows)
    assert "Page one line\n\nPage two line" in txt


def test_rows_to_plain_text_no_leading_blank_for_first_page():
    rows = [{"page_num": 1, "line_num": 1, "text": "Only line"}]
    txt = rows_to_plain_text(rows)
    assert txt == "Only line\n"


# ── _read_teitok_layout ──────────────────────────────────────────────────────
# Previously zero test coverage despite backing both `xml_to_md.py --format
# layout` and (transitively, via rows_to_layout_markdown) json_to_md.py's
# renderer -- the two converters issue #13's TODO calls out by name
# ("Make sure JSON-2-MD and TEITOK-2-MD pipelines work correctly ... with
# ... complex XMLs of pages").

TEITOK_LAYOUT_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<TEI>
<facsimile>
  <surface id="CTX1.surface1" lrx="2480" lry="3508"><graphic url="CTX1-1.png"/></surface>
</facsimile>
<text><body><div>
<pb n="1"/>
<s text="Prvni veta."><tok bbox="100 200 300 240">Prvni</tok></s>
<figure bbox="50 50 500 600" type="illustration"/>
</div></body></text>
</TEI>
"""


def test_read_teitok_layout_happy_path(tmp_path):
    p = tmp_path / "sample.teitok.xml"
    p.write_text(TEITOK_LAYOUT_SAMPLE, encoding="utf-8")
    rows, pages = _read_teitok_layout(p)

    assert rows == [
        {"page_num": 1, "line_num": 1, "text": "Prvni veta.", "bbox": [100, 200, 300, 240]}
    ]
    assert pages[1]["width"] == 2480
    assert pages[1]["height"] == 3508
    assert pages[1]["figures"] == [{"bbox": [50, 50, 500, 600], "type": "illustration"}]


def test_read_teitok_layout_survives_non_numeric_pb_n(tmp_path):
    """Same crash as read_teitok_rows() (int() on a roman-numeral <pb n>),
    independently reachable through the layout reader's own <pb> handling."""
    p = tmp_path / "roman.teitok.xml"
    p.write_text(
        '<TEI><text><body><div><pb n="I"/><s text="Front matter."/>'
        '<pb n="1"/><s text="Content."/></div></body></text></TEI>',
        encoding="utf-8",
    )
    rows, _pages = _read_teitok_layout(p)  # must not raise
    assert [r["text"] for r in rows] == ["Front matter.", "Content."]


def test_read_teitok_layout_survives_name_misclose(tmp_path):
    """<name>...</n> used to raise ET.ParseError before the layout reader
    was routed through teitok_read.parse_teitok()'s repair."""
    p = tmp_path / "misclose.teitok.xml"
    p.write_text(
        '<TEI><text><body><div><pb n="1"/>'
        '<s text="Zaklady kostela"><name>kostela</n></s>'
        "</div></body></text></TEI>",
        encoding="utf-8",
    )
    rows, _pages = _read_teitok_layout(p)  # must not raise
    assert rows[0]["text"] == "Zaklady kostela"


def test_read_teitok_layout_aligns_surfaces_when_pages_dont_start_at_one(tmp_path):
    """Regression for the silent data-loss bug: surface dimensions were
    aligned by treating a page's own <pb n="..."> label as an array index
    (surface #1 -> pages[1], #2 -> pages[2], ...), so any document whose
    page numbering doesn't start at 1 -- a continuation volume, an excerpt
    of a larger bound archive -- silently lost its canvas width/height.
    Surfaces must align by encounter order against the order <pb> elements
    introduce pages instead."""
    p = tmp_path / "continuation.teitok.xml"
    p.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
        <TEI>
        <facsimile>
          <surface lrx="2480" lry="3508"/>
          <surface lrx="2490" lry="3510"/>
        </facsimile>
        <text><body><div>
        <pb n="47"/>
        <s text="Continuation page forty-seven."/>
        <pb n="48"/>
        <s text="Continuation page forty-eight."/>
        </div></body></text>
        </TEI>
        """,
        encoding="utf-8",
    )
    _rows, pages = _read_teitok_layout(p)

    assert pages[47]["width"] == 2480
    assert pages[47]["height"] == 3508
    assert pages[48]["width"] == 2490
    assert pages[48]["height"] == 3510


def test_convert_layout_reflects_correct_page_dims_for_continuation_numbering(tmp_path):
    """End-to-end through convert(fmt="layout") -- the DOC_META cue for a
    continuation-numbered page must carry that page's own real dimensions,
    not a phantom page's or none at all."""
    p = tmp_path / "continuation.teitok.xml"
    p.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
        <TEI>
        <facsimile>
          <surface lrx="2480" lry="3508"/>
        </facsimile>
        <text><body><div>
        <pb n="47"/>
        <s text="Continuation page forty-seven."/>
        </div></body></text>
        </TEI>
        """,
        encoding="utf-8",
    )
    md = convert(p, fmt="layout")
    assert "## Page 47" in md
    assert "DOC_META: size=2480x3508px" in md


# ── convert() end-to-end ─────────────────────────────────────────────────────


def test_convert_markdown_uses_doc_id_as_title(tmp_path):
    p = tmp_path / "CTX195603828.teitok.xml"
    p.write_text(TEITOK_SAMPLE, encoding="utf-8")
    md = convert(p, fmt="markdown")
    assert md.startswith("# CTX195603828")
    assert "## Page 1" in md
    assert "## Page 2" in md


def test_convert_text_has_no_markdown_headings(tmp_path):
    p = tmp_path / "CTX195603828.teitok.xml"
    p.write_text(TEITOK_SAMPLE, encoding="utf-8")
    txt = convert(p, fmt="text")
    assert "#" not in txt
    assert "Prvni veta na strance." in txt
