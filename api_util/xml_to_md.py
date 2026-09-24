"""
api_util/xml_to_md.py — TEITOK / ALTO → Markdown converter.

Renders a whole XML document (TEITOK ``*.teitok.xml`` or raw ALTO) to
Markdown or plain text, so entire documents can be fed to LLMs as a single
prompt (local or as an OpenRouter file/text attachment) — document-level
input, complementing the existing line-level CSV/TEITOK row reader in
llm_client_shared.read_input_rows() / llm_utils.read_input_rows().

Builds on teitok_read.py (TEITOK) and a small, dependency-free ALTO reader
below, following teitok_read.read_teitok_rows()'s row shape
({"page_num", "line_num", "text"}) so both formats feed the same renderer.

Note: the ALTO parser of atrium-nlp-enrich's TEITOK writer (``teitok_alto.py``,
not vendored here — this repo only reads TEITOK) is intentionally NOT reused: it
is tightly coupled to the CoNLL-U+NER merge pipeline and returns bbox/image
metadata this converter doesn't need. ``_read_alto_rows`` below extracts only
String/TextLine text, in the same namespace-agnostic parsing style.
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from api_util import layout_md as L  # noqa: E402
from api_util.teitok_read import (  # noqa: E402
    _events,
    _join_tokens,
    _Pages,
    _token_records,
    doc_id_from_path,
    parse_teitok,
    read_teitok_rows,
    sentence_text,
)


def _local_tag(elem: ET.Element) -> str:
    """Strip an XML namespace URI from a tag, e.g. '{ns}Page' -> 'Page'."""
    return elem.tag.split("}")[-1]


def is_alto(path: str | Path) -> bool:
    """Best-effort ALTO detection: peek at the root tag/namespace."""
    try:
        for _event, elem in ET.iterparse(str(path), events=("start",)):
            return _local_tag(elem).lower() == "alto" or "alto" in elem.tag.lower()
    except ET.ParseError:
        return False
    return False


def _read_alto_rows(path: str | Path) -> List[dict]:
    """
    Parses raw ALTO XML (Page > PrintSpace > TextBlock > TextLine > String).
    Returns: list of dicts [{"page_num": int, "line_num": int, "text": str}],
    matching teitok_read.read_teitok_rows()'s row shape.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    rows: List[dict] = []

    page_num = 0
    for page_elem in root.iter():
        if _local_tag(page_elem) != "Page":
            continue
        page_num += 1
        try:
            page_num = int(page_elem.get("PHYSICAL_IMG_NR", page_num))
        except (TypeError, ValueError):
            pass

        line_num = 0
        for line_elem in page_elem.iter():
            if _local_tag(line_elem) != "TextLine":
                continue
            line_num += 1

            words = [
                s.get("CONTENT", "")
                for s in line_elem.iter()
                if _local_tag(s) == "String" and s.get("CONTENT")
            ]
            text = " ".join(w for w in words if w).strip()
            if text:
                rows.append({"page_num": page_num, "line_num": line_num, "text": text})

    return rows


def read_document_rows(path: str | Path) -> List[dict]:
    """Reads rows from either a TEITOK or a raw ALTO XML document."""
    path = Path(path)
    if path.name.lower().endswith(".teitok.xml"):
        return read_teitok_rows(path)
    if is_alto(path):
        return _read_alto_rows(path)
    # Fall back to TEITOK parsing — read_teitok_rows() is namespace-agnostic
    # and will simply return [] if the structure doesn't match, rather than
    # raising, so this is a safe default rather than a silent misdetection.
    return read_teitok_rows(path)


def _parse_bbox_attr(value: str | None) -> list | None:
    """Parse a TEITOK ``bbox="x1 y1 x2 y2"`` attribute into ``[x1, y1, x2, y2]``."""
    if not value:
        return None
    parts = value.split()
    if len(parts) != 4:
        return None
    try:
        return [int(float(p)) for p in parts]
    except (ValueError, TypeError):
        return None


def _alto_box(elem: ET.Element) -> list | None:
    """[left, top, right, bottom] from an ALTO element's HPOS/VPOS/WIDTH/HEIGHT."""
    try:
        h = float(elem.get("HPOS", "") or "")
        v = float(elem.get("VPOS", "") or "")
        w = float(elem.get("WIDTH", "") or "")
        ht = float(elem.get("HEIGHT", "") or "")
    except (ValueError, TypeError):
        return None
    return [int(h), int(v), int(h + w), int(v + ht)]


def _read_alto_layout(path: str | Path) -> tuple:
    """ALTO → (rows, pages) with coordinates.

    rows: [{"page_num", "line_num", "text", "bbox"}] (bbox = per-TextLine box).
    pages: {page_num: {"width", "height", "figures": [{"bbox", "type"}]}} —
    the page canvas size and any Illustration/GraphicalElement regions.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    rows: List[dict] = []
    pages: dict = {}

    page_num = 0
    for page_elem in root.iter():
        if _local_tag(page_elem) != "Page":
            continue
        page_num += 1
        try:
            page_num = int(page_elem.get("PHYSICAL_IMG_NR", page_num))
        except (TypeError, ValueError):
            pass

        def _int(v):
            try:
                return int(float(v))
            except (ValueError, TypeError):
                return None

        pages.setdefault(page_num, {"width": None, "height": None, "figures": []})
        pages[page_num]["width"] = _int(page_elem.get("WIDTH"))
        pages[page_num]["height"] = _int(page_elem.get("HEIGHT"))

        line_num = 0
        for elem in page_elem.iter():
            tag = _local_tag(elem)
            if tag == "TextLine":
                line_num += 1
                words = [
                    s.get("CONTENT", "")
                    for s in elem.iter()
                    if _local_tag(s) == "String" and s.get("CONTENT")
                ]
                text = " ".join(w for w in words if w).strip()
                if text:
                    rows.append(
                        {
                            "page_num": page_num,
                            "line_num": line_num,
                            "text": text,
                            "bbox": _alto_box(elem),
                        }
                    )
            elif tag in ("Illustration", "GraphicalElement"):
                box = _alto_box(elem)
                if box:
                    pages[page_num]["figures"].append({"bbox": box, "type": tag})

    return rows, pages


def _union(boxes: List[list]) -> list | None:
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    return [
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    ]


def _bbox_origin(root: ET.Element) -> str | None:
    """What atrium-nlp-enrich's writer says its boxes are measured from
    (``<application ident="atrium-nlp-enrich"><desc>bbox origin: page|printspace</desc>``),
    or None for a document it did not write."""
    for el in root.iter():
        if _local_tag(el) == "application" and el.get("ident") == "atrium-nlp-enrich":
            for desc in el:
                text = "".join(desc.itertext()) if _local_tag(desc) == "desc" else ""
                if text.strip().startswith("bbox origin:"):
                    return text.split(":", 1)[1].strip() or None
    return None


def _read_teitok_layout(path: str | Path) -> tuple:
    """TEITOK → (rows, pages) with coordinates.

    The rows are ``teitok_read.read_teitok_rows()``'s -- the same page and line numbering
    (the vendored reader's private ``_Pages``/``_events`` helpers, pinned with it), one row per
    ``<s>`` or per page part of an ``<s>`` that contains a ``<pb/>`` (nlp-enrich writes one
    there when a sentence runs over a page break) -- plus a ``bbox`` aggregated from the
    ``<tok bbox>`` of that row, so a box never spans two pages. A page labelled otherwise than
    its number (``pb@n="I"``) gives its rows a ``page_label``. ``pages`` holds the canvas size
    from ``<surface lrx lry>`` (in document order), the figures from ``<figure bbox type>``,
    and ``origin="printspace"`` when the writer says its boxes are PrintSpace-relative.
    Documents without ``<s>`` (flexiconv output) get teitok_read's line/block rows, without
    a bbox.
    """
    root = parse_teitok(path)
    rows: List[dict] = []
    pages: dict = {}
    surface_dims: List[tuple] = []
    for elem in root.iter():
        if _local_tag(elem) == "surface":
            try:
                surface_dims.append(
                    (int(float(elem.get("lrx", "") or "")), int(float(elem.get("lry", "") or "")))
                )
            except (ValueError, TypeError):
                surface_dims.append((None, None))

    def page(num: int) -> dict:
        return pages.setdefault(num, {"width": None, "height": None, "figures": []})

    pager = _Pages()
    page(pager.num)
    page_order: List[int] = []
    records = _token_records(root)
    has_sentences = False
    sent = None

    def new_part():
        return {"at": pager.snapshot(), "line0": pager.line, "line": None, "toks": []}

    def add_row(text, part, toks):
        at = part["at"]
        row = {
            "page_num": at["page_num"],
            "line_num": part["line"] if part["line"] is not None else part["line0"],
            "text": text,
            "bbox": _union([_parse_bbox_attr(t.get("bbox")) for t in toks]),
        }
        if at["page_label"] and at["page_label"] != str(at["page_num"]):
            row["page_label"] = at["page_label"]
        rows.append(row)

    for kind, value in _events(root):
        if kind == "tok":
            if sent is not None:
                part = sent["parts"][-1]
                if part["line"] is None:
                    part["line"] = pager.line
                part["toks"].append(value)
            pager.content()
            continue
        if kind == "text":
            continue
        tag = _local_tag(value)
        if kind == "start":
            if tag == "pb":
                pager.pb(value)
                page(pager.num)
                page_order.append(pager.num)
                if sent is not None:
                    sent["parts"].append(new_part())
            elif tag == "lb":
                pager.lb()
            elif tag == "figure":
                box = _parse_bbox_attr(value.get("bbox"))
                if box:
                    page(pager.num)["figures"].append({"bbox": box, "type": value.get("type", "")})
            elif tag == "s" and sent is None:
                has_sentences = True
                sent = {"elem": value, "parts": [new_part()]}
        elif sent is not None and value is sent["elem"]:
            parts = [p for p in sent["parts"] if p["toks"]]
            if len(parts) > 1:
                for part in parts:
                    text = _join_tokens([records[t] for t in part["toks"]])
                    if text:
                        add_row(text, part, part["toks"])
            else:
                part = parts[0] if parts else sent["parts"][0]
                text = sentence_text(value)
                if text:
                    add_row(text, part, part["toks"])
                    pager.content()
            sent = None

    if not has_sentences:
        rows = []
        for row in read_teitok_rows(path):
            out = {k: row[k] for k in ("page_num", "line_num", "text")}
            out["bbox"] = None
            label = row.get("page_label")
            if label and label != str(row["page_num"]):
                out["page_label"] = label
            rows.append(out)
            page(row["page_num"])

    # Surfaces are written one-per-page, in the same document order that
    # <pb> elements introduce pages — align positionally against THAT order,
    # not against the page's own `n` label. A label-as-index assumption
    # (surface #1 -> pages[1], surface #2 -> pages[2], ...) silently drops
    # dimensions whenever numbering doesn't start at 1 or isn't contiguous
    # (continuation volumes, roman-numeral front matter, an appendix
    # restarting the count) — the surface data lands on a phantom page key
    # instead of the real one. Fall back to the single implicit page when
    # the document has no <pb> at all.
    if not page_order:
        page_order = [pager.num]
    for i, (w, h) in enumerate(surface_dims):
        if i < len(page_order):
            target_page = page_order[i]
            pages[target_page]["width"] = w
            pages[target_page]["height"] = h

    # BBOX_ORIGIN=printspace: the boxes are not page boxes; say so in every page's DOC_META
    # (atrium-llm-enrich#13, P5.4). The default, "page", is what the cues mean anyway.
    if _bbox_origin(root) == "printspace":
        for meta in pages.values():
            meta["origin"] = "printspace"

    return rows, pages


def read_document_layout(path: str | Path) -> tuple:
    """Reads (rows, pages) with coordinates from a TEITOK or ALTO document."""
    path = Path(path)
    if path.name.lower().endswith(".teitok.xml"):
        return _read_teitok_layout(path)
    if is_alto(path):
        return _read_alto_layout(path)
    return _read_teitok_layout(path)


def rows_to_layout_markdown(rows: List[dict], pages: dict, title: str = "") -> str:
    """Renders coordinate-bearing rows as visually-rich, page-sectioned Markdown.

    Emits the same layout_md cue vocabulary as the PDF/DOCX converters —
    ``## Page N`` + ``<!-- PAGE_BREAK -->``, ``<!-- DOC_META -->`` (canvas size),
    ``<!-- BBOX -->`` per line, and ``![figure]`` placeholders — so TEITOK/ALTO
    input lands on the one annotated-Markdown schema (issue #11).

    A page's meta dict may also carry ``needs_ocr`` (bool), ``needs_ocr_reason``
    (str), ``ocr`` (``{"engine": ..., "lang": ...}``) and ``unit`` (the canvas
    unit, default ``px``) — populated only by json_to_md.py, which reads them
    straight off the AtriumDocument record's ``pages[]`` block; no other caller
    sets them today, so this is purely additive for existing ones.

    A row may additionally carry ``group_id`` and ``page_label`` (issue #18):

    * ``group_id`` — the source-structural unit the line came from (a DOCX
      paragraph, a table cell, a PDF text block). A change of value emits a
      **blank line**, which is Markdown's own paragraph primitive, so the
      grouping needs no new ``layout_md`` cue. This is the consumer half of the
      contract stated in ``atrium_document.schema.json``'s ``lines[].group_id``:
      without it the field was inert and the converter's paragraph fidelity never
      reached the model.
    * ``page_label`` — the page's real label when it is not simply ``str(page_num)``,
      so ``iv`` or ``A-1`` is what appears in ``## Page …`` and in the PAGE_BREAK
      cue the citation format keys off, rather than a synthetic ordinal.

    Rows without either key render exactly as before, so the ALTO/TEITOK path is
    byte-for-byte unchanged.
    """
    pages = pages or {}
    parts: List[str] = [f"# {title}"] if title else []
    current_page = None
    # A distinct sentinel, because `None` is a legitimate group_id (the ALTO path): the first
    # text row of a page must never emit a boundary, whatever its group is.
    no_group = object()
    current_group: object = no_group

    for row in rows:
        page = row.get("page_num")
        label = row.get("page_label", page)
        if page != current_page:
            if current_page is not None:
                parts.append(L.page_break(label))
            parts.append(f"\n## Page {label}\n")
            meta = pages.get(page, {})
            w, h = meta.get("width"), meta.get("height")
            origin = {"origin": meta["origin"]} if meta.get("origin") else {}
            if w and h:
                parts.append(L.doc_meta(size=f"{w}x{h}{meta.get('unit', 'px')}", **origin))
            elif origin:
                parts.append(L.doc_meta(**origin))
            for fig in meta.get("figures", []):
                parts.append(L.image(fig.get("type", "figure"), "", fig.get("bbox")))
            if meta.get("needs_ocr"):
                parts.append(
                    L.needs_ocr(
                        label, reason=meta.get("needs_ocr_reason", "no extractable text layer")
                    )
                )
            ocr_meta = meta.get("ocr")
            if ocr_meta:
                parts.append(
                    L.ocr_meta(engine=ocr_meta.get("engine", "unknown"), lang=ocr_meta.get("lang"))
                )
            current_page = page
            current_group = no_group

        text = str(row.get("text", "")).strip()
        if not text:
            continue
        group = row.get("group_id")
        if current_group is not no_group and group != current_group:
            parts.append("")
        current_group = group
        box = row.get("bbox")
        parts.append(f"{L.bbox(box)}\n{text}" if box else text)

    return "\n".join(parts).strip() + "\n"


def rows_to_markdown(rows: List[dict], title: str = "") -> str:
    """Renders {page_num, line_num, text} rows as page-sectioned Markdown."""
    if not rows:
        return f"# {title}\n" if title else ""

    parts: List[str] = [f"# {title}"] if title else []
    current_page = None
    for row in rows:
        page = row.get("page_num")
        if page != current_page:
            parts.append(f"\n## Page {page}\n")
            current_page = page
        text = str(row.get("text", "")).strip()
        if text:
            parts.append(text)

    return "\n".join(parts).strip() + "\n"


def rows_to_plain_text(rows: List[dict]) -> str:
    """Renders rows as plain text, one line per row, page breaks as blank lines."""
    parts: List[str] = []
    current_page = None
    for row in rows:
        page = row.get("page_num")
        if current_page is not None and page != current_page:
            parts.append("")
        current_page = page
        text = str(row.get("text", "")).strip()
        if text:
            parts.append(text)
    return "\n".join(parts).strip() + "\n"


def convert(path: str | Path, fmt: str = "markdown") -> str:
    """Convert a TEITOK/ALTO XML document to 'markdown', 'text', or 'layout'.

    'layout' emits visually-rich Markdown carrying the layout_md cue vocabulary
    (page dimensions, bounding boxes, page breaks, figures) — the same schema as
    the PDF/DOCX converters.
    """
    path = Path(path)
    if fmt == "layout":
        rows, pages = read_document_layout(path)
        return rows_to_layout_markdown(rows, pages, title=doc_id_from_path(path))
    rows = read_document_rows(path)
    if fmt == "text":
        return rows_to_plain_text(rows)
    return rows_to_markdown(rows, title=doc_id_from_path(path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_file", type=Path)
    parser.add_argument("--format", choices=["markdown", "text", "layout"], default="markdown")
    parser.add_argument(
        "--output", type=Path, default=None, help="Write to file instead of stdout."
    )
    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    rendered = convert(args.input_file, fmt=args.format)
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"-> {args.output}")
    else:
        print(rendered)
