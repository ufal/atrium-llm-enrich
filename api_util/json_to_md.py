"""
api_util/json_to_md.py — AtriumDocument JSON → visually-rich Markdown.

Closes the loop `regenerable.markdown` currently leaves open: llm-enrich already
records the annotated Markdown as a *recipe* rather than a stored path — e.g.
``{"from": "<doc_id>.document.json", "converter": "json_to_md@1.0", "detail":
"full"}`` — but until now nothing could actually execute that recipe from just
the JSON. A consumer holding only the AtriumDocument record (the FAIR search
artifact issue #13 settled on) had a dangling pointer, not a regenerable one.

Renders through the SAME cue vocabulary as the TEITOK/ALTO/PDF/DOCX front-ends
(``api_util/layout_md.py``'s ``CUE_SCHEMA``, via ``xml_to_md.rows_to_layout_markdown``)
so the JSON path produces the one canonical LLM diet, not a second dialect. It can
also do two things no other converter can, because only the record carries the
data for them:

  * emit ``NEEDS_OCR``/``OCR`` cues straight from ``pages[].needs_ocr``/``pages[].ocr``;
  * drop heuristically-bad lines via ``lines[].categ``/``quality_score`` before
    they ever reach the model, via ``--min-quality``.

Two hard constraints, both load-bearing:

  1. Reads ONLY ``pages``/``lines``/``content``. NEVER ``enrichment`` — that block
     is llm-enrich's OWN prior output; reading it back would feed the model its
     own earlier answer on a second run over the same document.
  2. No silent structureless dumps. The fallback ladder is ``lines[]`` (page-
     sectioned, cue-annotated) -> ``content.text`` (unsectioned, and it says so
     on stderr) -> ``ValueError`` naming exactly what's missing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from api_util import layout_md as L  # noqa: E402
from api_util import xml_to_md  # noqa: E402
from atrium_document import load_document  # noqa: E402
from atrium_vocab import UNTRUSTWORTHY_LINE_CATEGORIES  # noqa: E402

#: Line categories excluded from the rendered text by default — text this pipeline has
#: already judged untrustworthy, which would only degrade the LLM's read of the page
#: rather than inform it.
#:
#: The hub's semantic set, `atrium_vocab.UNTRUSTWORTHY_LINE_CATEGORIES` — ("Garbage",
#: "Inverted", "Trash") — which spans both originators of `lines[].categ`:
#: `api_util/digital_to_json.py` writes Garbage/Inverted on the digital-born path
#: (`atrium_vocab.LINE_CATEGORY_ORIGINATORS["digital-convert"]`), alto-postprocess writes
#: {Clear, Empty, Noisy, Non-text, Trash} on the OCR path.
#:
#: FIXED — defect V-1 of the hub's `docs/skos_strategy.md` §6 (atrium-alto-postprocess#31,
#: 2026-09-25). This was frozenset({"Garbage", "Inverted"}), the digital-convert set only, so
#: on the OCR path it matched nothing and every `Trash` line reached the model. Filtering
#: `Trash` changes what the model is shown on every OCR document — the point of the fix.
DROP_CATEGORIES = frozenset(UNTRUSTWORTHY_LINE_CATEGORIES)

#: The only implemented profile today. "standard"/"minimal" are the down-profiles
#: from issue #13 §B, still deferred across the whole MD front-end — accepted
#: here for CLI/API parity with the schema's regenerable.detail enum, but not
#: silently downgraded to "full" if requested.
IMPLEMENTED_DETAIL_LEVELS = frozenset({"full"})

#: `lines[].style.region` values (written by `api_util/digital_to_json.py`; see
#: `api_util/digital_ir.py`) and where each renders on its page: running header first,
#: then the body, then footnotes, then the running footer. Absent = body, which is every
#: line on the ALTO path — its order is unchanged.
REGION_RANK = {"page_header": 0, None: 1, "footnote": 2, "page_footer": 3}

#: Headings sit INSIDE the `## Page N` sections, so a level-1 heading renders as `###`. A
#: `##` heading would compete with the page markers the citation locator reads (#11).
HEADING_OFFSET = 2


def _label(value: object) -> Optional[str]:
    """A page label as the schema stores it — a string — or None when absent."""
    if value is None:
        return None
    label = str(value).strip()
    return label or None


def page_ordinals(pages: List[dict], lines: List[dict]) -> Dict[str, int]:
    """
    Map each page LABEL to the integer the renderer sorts and sections on.

    The renderer needs an int key; the schema deliberately keeps ``page`` a string "so 'iv'
    or 'A-1' survive". Reconciling those used to be a bare ``int(line["page"])`` with a
    ``continue`` on failure, so a document with roman-numeral front matter — precisely the
    digital-born archival material Issue #18 exists to ingest — lost those lines with no
    diagnostic, and then died with ``ValueError("nothing to render")`` blaming upstream
    stages for text they had in fact produced.

    Three cases, in order:

    1. **Every label is a plain non-negative integer** — use ``int(label)``. This is the ALTO
       path and every record written before ``page_index`` existed; output is unchanged.
    2. **Some label is not** — use ``pages[].page_index`` when every listed page declares
       one, since that is the field the schema names as the ordering key.
    3. **Neither** — fall back to first-appearance order, ``pages[]`` first (it is in
       document order) and then any page only ``lines[]`` mentions.
    """
    labels: List[str] = []
    for source in (pages, lines):
        for item in source:
            label = _label(item.get("page")) if isinstance(item, dict) else None
            if label and label not in labels:
                labels.append(label)

    if labels and all(label.isdigit() for label in labels):
        return {label: int(label) for label in labels}

    declared: Dict[str, int] = {}
    for p in pages:
        label = _label(p.get("page"))
        idx = p.get("page_index")
        if label and isinstance(idx, int) and not isinstance(idx, bool):
            declared[label] = idx
    listed = [lbl for lbl in ({_label(p.get("page")) for p in pages} - {None}) if lbl]
    if listed and len(declared) == len(listed) and len(set(declared.values())) == len(declared):
        ordinals = dict(declared)
        nxt = max(declared.values())
        for label in labels:
            if label not in ordinals:
                nxt += 1
                ordinals[label] = nxt
        return ordinals

    return {label: i + 1 for i, label in enumerate(labels)}


def _rows_from_lines(
    lines: List[dict], min_quality: float, ordinals: Optional[Dict[str, int]] = None
) -> List[dict]:
    ordinals = ordinals if ordinals is not None else page_ordinals([], lines)
    rows: List[dict] = []
    for line in lines:
        if line.get("categ") in DROP_CATEGORIES:
            continue
        score = line.get("quality_score")
        if score is not None and score < min_quality:
            continue
        text = str(line.get("text") or "").strip()
        if not text:
            continue
        label = _label(line.get("page"))
        if label is None:
            continue
        page_num = ordinals.get(label)
        if page_num is None:
            print(
                f"[json_to_md] line on page {label!r} has no position in the document's page "
                f"order — rendering it after the known pages. Populate pages[].page_index.",
                file=sys.stderr,
            )
            page_num = max(ordinals.values(), default=0) + 1
            ordinals[label] = page_num
        row = {"page_num": page_num, "line_num": line.get("line", 0), "text": text}
        # Only added when it carries information the ordinal does not, so a numeric-label
        # record (the whole ALTO path) produces exactly the rows it produced before.
        if label != str(page_num):
            row["page_label"] = label
        if line.get("bbox"):
            row["bbox"] = line["bbox"]
        # Issue #18 §1c: the paragraph grouping the converter went to the trouble of
        # extracting has to reach the renderer, or the schema field is a no-op with extra
        # steps. Absent on the ALTO path, where it stays absent from the row too.
        if line.get("group_id") is not None:
            row["group_id"] = line["group_id"]
        # Issue #18, the layout-cue box: `style` (digital-convert only) was stored and never
        # rendered. Kept on the row for `_decorate()`; absent on the ALTO path.
        style = line.get("style")
        if isinstance(style, dict) and style:
            row["style"] = style
        rows.append(row)
    rows.sort(key=_row_order)
    return rows


def _region(row: dict) -> Optional[str]:
    region = (row.get("style") or {}).get("region")
    return region if region in REGION_RANK else None


def _row_order(row: dict) -> Tuple[int, int, int]:
    return (row["page_num"], REGION_RANK[_region(row)], row.get("line_num", 0))


def _decorate(row: dict) -> str:
    """A body line's Markdown: heading marks, or whole-line emphasis.

    Only for body lines — page furniture and footnotes get their own cues, and a heading
    style on them would be noise. Emphasis is whole-line because `style` is per line; a
    line that is only partly bold carries no `bold` flag and renders plain.
    """
    style = row.get("style") or {}
    text = row["text"]
    level = style.get("heading_level")
    if isinstance(level, int) and not isinstance(level, bool) and level >= 1:
        return "#" * min(level + HEADING_OFFSET, 6) + " " + text
    if "*" in text:  # emphasis marks around text that has its own would garble it
        return text
    if style.get("bold") and style.get("italic"):
        return f"***{text}***"
    if style.get("bold"):
        return f"**{text}**"
    if style.get("italic"):
        return f"*{text}*"
    return text


def _note_id(group_id: object, fallback: int) -> str:
    """`fn12` → `12`, `en3` → `e3`: the label a `[^…]` definition shows."""
    value = str(group_id or "")
    if value.startswith("fn") and value[2:]:
        return value[2:]
    if value.startswith("en") and value[2:]:
        return "e" + value[2:]
    return str(fallback)


def _assemble_regions(rows: List[dict]) -> List[dict]:
    """Per page: headers inside HEADER cues, decorated body, footnotes as `[^n]:`
    definitions, footers inside FOOTER cues. Rows without `style` pass through unchanged."""
    out: List[dict] = []
    index = 0
    while index < len(rows):
        page = rows[index]["page_num"]
        page_rows = []
        while index < len(rows) and rows[index]["page_num"] == page:
            page_rows.append(rows[index])
            index += 1
        for region in ("page_header", None, "footnote", "page_footer"):
            members = [r for r in page_rows if _region(r) == region]
            if not members:
                continue
            if region is None:
                for row in members:
                    if "style" in row:
                        row = dict(row, text=_decorate(row))
                    out.append(row)
                continue
            first = dict(members[0])
            first.pop("bbox", None)
            first["group_id"] = f"__{region}__"
            if region == "footnote":
                notes: Dict[str, List[str]] = {}
                for n, row in enumerate(members, 1):
                    notes.setdefault(_note_id(row.get("group_id"), n), []).append(row["text"])
                first["text"] = "\n".join(L.footnote_def(k, " ".join(v)) for k, v in notes.items())
            else:
                start, end = (
                    (L.header_start(), L.header_end())
                    if region == "page_header"
                    else (L.footer_start(), L.footer_end())
                )
                first["text"] = "\n".join([start, *(r["text"] for r in members), end])
            out.append(first)
    return out


def _cell_groups(table: dict) -> List[Tuple[int, int, Optional[str], List[Tuple[str, int]]]]:
    """(row, col, group_id, explicit line refs) per cell entry of a `tables[]` item."""
    cells = []
    for cell in table.get("cells") or []:
        if not isinstance(cell, dict):
            continue
        refs = [
            (str(ref.get("page")), int(ref.get("line", 0)))
            for ref in (cell.get("lines") or [])
            if isinstance(ref, dict) and ref.get("page") is not None
        ]
        cells.append((int(cell.get("row", 0)), int(cell.get("col", 0)), cell.get("group_id"), refs))
    return cells


def _assemble_tables(rows: List[dict], tables: List[dict]) -> List[dict]:
    """Replace each table's cell lines with ONE row holding the GFM table.

    The `tables` block carries grid SHAPE only; cell text lives once in `lines[]`, joined by
    `cells[].group_id` (or an explicit `cells[].lines`, which wins). A merged area repeats
    its owner's group at every position it covers; only the first position shows the text.
    The table lands where its first cell line was. A table whose cells join to no rendered
    line (the ALTO path writes no cell group_ids) is left alone, and so are its lines.
    """
    for table in tables or []:
        if not isinstance(table, dict):
            continue
        cells = _cell_groups(table)
        if not cells:
            continue
        by_group: Dict[str, List[dict]] = {}
        for row in rows:
            if row.get("group_id") is not None:
                by_group.setdefault(str(row["group_id"]), []).append(row)
        by_ref = {(r.get("page_label", str(r["page_num"])), r.get("line_num")): r for r in rows}
        n_rows = max([int(table.get("n_rows") or 0)] + [r + 1 for r, _, _, _ in cells])
        n_cols = max([int(table.get("n_cols") or 0)] + [c + 1 for _, c, _, _ in cells])
        grid = [["" for _ in range(n_cols)] for _ in range(n_rows)]
        used: List[dict] = []
        shown = set()
        for r, c, group, refs in cells:
            members = (
                [by_ref[ref] for ref in refs if ref in by_ref]
                if refs
                else by_group.get(str(group), [])
            )
            key = tuple(refs) if refs else group
            if not members or key in shown:
                continue
            shown.add(key)
            grid[r][c] = " ".join(m["text"] for m in members)
            used.extend(members)
        if not used:
            continue
        header = any(
            isinstance(cell, dict) and cell.get("row") == 0 and cell.get("is_header")
            for cell in table.get("cells") or []
        )
        anchor = min(used, key=_row_order)
        boxes = [m["bbox"] for m in used if m.get("bbox")]
        block = {
            "page_num": anchor["page_num"],
            "line_num": anchor.get("line_num", 0),
            "text": L.md_table(grid, header=header),
            "group_id": f"__table__{table.get('table_id', '')}",
        }
        if "page_label" in anchor:
            block["page_label"] = anchor["page_label"]
        if boxes:
            block["bbox"] = [
                min(b[0] for b in boxes),
                min(b[1] for b in boxes),
                max(b[2] for b in boxes),
                max(b[3] for b in boxes),
            ]
        used_ids = {id(m) for m in used}
        position = next(i for i, row in enumerate(rows) if row is anchor)
        rows = (
            [row for row in rows[:position] if id(row) not in used_ids]
            + [block]
            + [row for row in rows[position:] if id(row) not in used_ids]
        )
    return rows


def _pages_meta(pages: List[dict], ordinals: Optional[Dict[str, int]] = None) -> Dict[int, dict]:
    ordinals = ordinals if ordinals is not None else page_ordinals(pages, [])
    meta: Dict[int, dict] = {}
    for p in pages:
        label = _label(p.get("page"))
        page_num = ordinals.get(label) if label else None
        if page_num is None:
            continue
        canvas = p.get("canvas") or {}
        entry: Dict[str, object] = {}
        if canvas.get("width") and canvas.get("height"):
            entry["width"] = canvas["width"]
            entry["height"] = canvas["height"]
            # The DOC_META cue used to hardcode "px". A digital-born PDF page is in points,
            # so every such cue told the model a page was 595x842 PIXELS — and api_util/
            # pdf_to_md.py already emitted "pt" for the same document, so the two front-ends
            # disagreed on one cue. Absent unit still renders "px", unchanged.
            if canvas.get("unit"):
                entry["unit"] = canvas["unit"]
        if p.get("needs_ocr"):
            entry["needs_ocr"] = True
            # Without this the renderer's default fires and every digital-born page reads
            # "no extractable text layer" — the opposite of what needs_ocr means on that
            # path, where the text layer exists and merely decodes to garbage.
            if p.get("needs_ocr_reason"):
                entry["needs_ocr_reason"] = p["needs_ocr_reason"]
        if p.get("ocr"):
            entry["ocr"] = p["ocr"]
        meta[page_num] = entry
    return meta


def rows_from_record(
    record: Dict[str, Any], min_quality: float = 0.0
) -> Tuple[List[dict], Dict[int, dict], Optional[str]]:
    """``(rows, pages, fallback_text)`` for an AtriumDocument record already in memory —
    ``pages``/``lines``/``tables``/``content`` only, never ``enrichment``.

    Beyond the TEITOK/ALTO row shape ``xml_to_md.rows_to_layout_markdown`` expects:

    * a line's ``style`` becomes Markdown — heading marks, whole-line emphasis, the
      ``HEADER``/``FOOTER`` cue blocks and ``[^n]:`` footnote definitions (Issue #18);
    * a table's cell lines become one GFM table row (``tables[].cells[].group_id``);
    * a page with ``needs_ocr`` and no renderable line still gets its ``## Page N`` section
      and its ``NEEDS_OCR`` cue (G1): an empty-text row is what makes the renderer open the
      section, and it prints nothing else for it.

    ``fallback_text`` is set only when ``lines[]`` yielded nothing usable.
    """
    lines = record.get("lines") or []
    page_block = record.get("pages") or []
    # One ordinal map shared by both, so a line and its page's metadata cannot end up under
    # different keys — which is what would happen if each derived its own.
    ordinals = page_ordinals(page_block, lines)
    rows = _rows_from_lines(lines, min_quality, ordinals)
    pages = _pages_meta(page_block, ordinals)
    real = bool(rows)
    if rows:
        rows = _assemble_tables(rows, record.get("tables") or [])
        rows = _assemble_regions(rows)
    with_rows = {r["page_num"] for r in rows}
    for p in page_block:
        label = _label(p.get("page")) if isinstance(p, dict) else None
        page_num = ordinals.get(label) if label else None
        if page_num is None or page_num in with_rows or not p.get("needs_ocr"):
            continue
        placeholder = {"page_num": page_num, "line_num": -1, "text": ""}
        if label != str(page_num):
            placeholder["page_label"] = label
        rows.append(placeholder)
    rows.sort(key=lambda r: r["page_num"])  # stable: keeps the order within each page
    fallback_text = None
    if not real:
        fallback_text = (record.get("content") or {}).get("text")
    return rows, pages, fallback_text


def read_document_rows(
    doc_json_path: str | Path, min_quality: float = 0.0
) -> Tuple[List[dict], Dict[int, dict], Optional[str]]:
    """
    Read ``(rows, pages, fallback_text)`` off an AtriumDocument JSON —
    ``pages``/``lines``/``tables``/``content`` only, never ``enrichment``.

    ``rows``/``pages`` are in exactly the shape ``xml_to_md.rows_to_layout_markdown``
    expects (mirroring ``read_document_layout``'s TEITOK/ALTO shape), so the
    renderer is shared rather than re-implemented. ``fallback_text`` is set only
    when ``lines[]`` yielded nothing usable.
    """
    return rows_from_record(load_document(str(doc_json_path)), min_quality=min_quality)


def render_record(
    record: Dict[str, Any], title: str, detail: str = "full", min_quality: float = 0.0
) -> str:
    """Render a record already in memory — what ``convert()`` does after loading the file.

    ``api_util/doc_to_visual_md.py`` renders PDF/DOCX through this, straight from
    ``digital_to_json.build_record()``, so the Markdown the model reads and the record that
    is stored come from one conversion (#10 G8).
    """
    if detail not in IMPLEMENTED_DETAIL_LEVELS:
        raise NotImplementedError(
            f"detail={detail!r} is not implemented yet (only {sorted(IMPLEMENTED_DETAIL_LEVELS)} — "
            f"the standard/minimal down-profiles are a deferred issue #13 item)."
        )

    rows, pages, fallback_text = rows_from_record(record, min_quality=min_quality)
    doc_id = title

    if fallback_text and fallback_text.strip():
        print(
            f"[json_to_md] {doc_id}: no usable lines[] in the record (alto-postprocess's "
            f"quality/categ pass may not have run, or --min-quality={min_quality} dropped "
            f"everything) — falling back to unsectioned content.text.",
            file=sys.stderr,
        )
        return f"# {doc_id}\n\n{fallback_text.strip()}\n"

    if rows:
        return xml_to_md.rows_to_layout_markdown(rows, pages, title=doc_id)

    raise ValueError(
        f"{doc_id}: record has neither a usable lines[] nor content.text — nothing to "
        f"render. Has any pipeline stage populated this document's text yet?"
    )


def convert(doc_json_path: str | Path, detail: str = "full", min_quality: float = 0.0) -> str:
    """
    Convert an AtriumDocument JSON (``<doc_id>.document.json``) to page-sectioned,
    cue-annotated Markdown.

    Raises ``NotImplementedError`` for a ``detail`` profile that isn't built yet
    (only ``"full"`` is), and ``ValueError`` when the record has neither
    ``lines[]`` nor ``content.text`` to render — never a silent empty/garbled dump.
    """
    if detail not in IMPLEMENTED_DETAIL_LEVELS:
        raise NotImplementedError(
            f"detail={detail!r} is not implemented yet (only {sorted(IMPLEMENTED_DETAIL_LEVELS)} — "
            f"the standard/minimal down-profiles are a deferred issue #13 item)."
        )

    record_path = Path(doc_json_path)
    doc_id = record_path.name
    if doc_id.lower().endswith(".document.json"):
        doc_id = doc_id[: -len(".document.json")]

    return render_record(
        load_document(str(doc_json_path)), title=doc_id, detail=detail, min_quality=min_quality
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("document_json", type=Path)
    parser.add_argument(
        "--output", type=Path, default=None, help="Write to file instead of stdout."
    )
    parser.add_argument("--detail", choices=["full", "standard", "minimal"], default="full")
    parser.add_argument(
        "--min-quality", type=float, default=0.0, help="Drop lines below this quality_score."
    )
    args = parser.parse_args()

    if not args.document_json.exists():
        print(f"Document JSON not found: {args.document_json}", file=sys.stderr)
        sys.exit(1)

    try:
        rendered = convert(args.document_json, detail=args.detail, min_quality=args.min_quality)
    except (ValueError, NotImplementedError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)

    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"-> {args.output}")
    else:
        print(rendered)
