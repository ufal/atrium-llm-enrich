"""
api_util/digital_docling.py — the heavyweight PDF engine of `digital_to_json.py` (`--engine docling`).

Opt-in, for when accuracy on a complex layout is worth a torch stack: Docling (MIT code;
layout model `docling-layout-heron`, Apache-2.0; table model TableFormer,
CDLA-Permissive-2.0) decides the page's STRUCTURE — reading order across columns, headings
and their levels, captions, page headers/footers, footnotes, and tables with or without
ruling. `pip install -r requirements_digital_docling.txt`. The light engine
(`api_util/digital_pdf.py`) stays the default: no models, no network, MIT/BSD/Apache.

Why a refinement and not a replacement. Docling's items are BLOCKS (a paragraph, a table
cell), each with one bbox; `lines[]` is a line-level plane, and the #18 plan's selling point
for the PDF path is exact native coordinates per line. So the light engine runs first and
Docling's structure is laid over its lines: a light line whose centre falls inside a Docling
item takes that item's reading position, `group_id` and style; an item no light line falls
into keeps Docling's own text with the block bbox; a light line no item claims is kept, in
its place by position, so the heavy engine can reorganise text but never lose it. Layer B
(decode sanity, grouping of the leftovers) then runs unchanged.

OCR stays off (`do_ocr=False`): a page without a text layer is flagged `needs_ocr` and
handed to the OCR originator (alto-postprocess), exactly as on the light path. Docling
writing OCR text into a `digital-born-*` record would put OCR output under the wrong
originator.

The mapping reads Docling's exported dict (`DoclingDocument.export_to_dict()`), not its
objects, so a saved IR (`digital_born/sample.pdf_docling_ir.json`, schema 1.x) and a live
conversion share one code path and the tests need no Docling install.

Offline use: point `DOCLING_ARTIFACTS_PATH` at a directory prepared with
`docling-tools models download` (the Dockerfile's `digital-docling` stage does this at build
time); without it Docling fetches the weights from Hugging Face on first use.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

from api_util.digital_ir import (
    REGION_FOOTER,
    REGION_FOOTNOTE,
    REGION_HEADER,
    DigitalDocument,
    DigitalLine,
    DigitalPage,
    DigitalTable,
    missing_dependency,
    renumber_lines,
)

ARTIFACTS_ENV = "DOCLING_ARTIFACTS_PATH"

#: `para_config.txt` components a Docling run adds to the light engine's.
COMPONENTS: Tuple[str, ...] = (
    "docling",
    "docling-parse",
    "docling-layout-heron",
    "docling-tableformer",
)

_REGION_BY_LABEL = {
    "page_header": REGION_HEADER,
    "page_footer": REGION_FOOTER,
    "footnote": REGION_FOOTNOTE,
}
_TABLE_LABELS = frozenset({"table", "document_index"})
#: Slack when testing a light line's centre against a Docling box: the two parsers round
#: glyph boxes differently, by a point or two.
_TOLERANCE = 2.0


def convert_with_docling(path: str) -> Dict[str, Any]:
    """Run Docling's standard PDF pipeline with OCR off; return the exported dict."""
    try:
        from docling.datamodel.base_models import InputFormat  # noqa: PLC0415
        from docling.datamodel.pipeline_options import PdfPipelineOptions  # noqa: PLC0415
        from docling.document_converter import DocumentConverter, PdfFormatOption  # noqa: PLC0415
    except ImportError as exc:
        raise missing_dependency("docling", "docling", "requirements_digital_docling.txt") from exc

    options = PdfPipelineOptions()
    options.do_ocr = False
    options.do_table_structure = True
    artifacts = os.environ.get(ARTIFACTS_ENV)
    if artifacts:
        options.artifacts_path = artifacts
    try:
        converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
        )
        result = converter.convert(path, raises_on_error=True)
    except Exception as exc:
        # The light engine has already read this PDF, so a failure here is Docling's
        # environment, and in practice its model weights: without DOCLING_ARTIFACTS_PATH it
        # fetches them from the Hugging Face Hub on first use. Advice, not a stack trace —
        # the same contract as any other missing dependency (exit 2).
        raise RuntimeError(
            f"Docling could not convert {os.path.basename(path)} "
            f"({type(exc).__name__}: {str(exc)[:200]}). Its layout and table models are "
            f"needed: run `docling-tools models download layout tableformer -o DIR` where "
            f"the Hugging Face Hub is reachable and set {ARTIFACTS_ENV}=DIR (the "
            f"digital-docling image does this at build time), or use --engine light."
        ) from exc
    return result.document.export_to_dict()


# ── reading the exported dict ─────────────────────────────────────────────────


def _resolve(ir: Dict[str, Any], ref: str) -> Optional[Dict[str, Any]]:
    """`#/texts/3` → the item."""
    parts = ref.lstrip("#/").split("/")
    if len(parts) != 2 or not parts[1].isdigit():
        return None
    items = ir.get(parts[0]) or []
    index = int(parts[1])
    return items[index] if 0 <= index < len(items) else None


def iter_items(ir: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
    """Every content item in Docling's reading order: the body tree, then the furniture tree.

    Groups (lists, inline groups) are transparent; a picture's or table's children (its
    caption, text inside a figure) follow it. Each item is yielded once even when two
    trees reference it.
    """
    seen = set()

    def walk(node: Dict[str, Any]) -> Iterator[Dict[str, Any]]:
        for child in node.get("children") or []:
            item = _resolve(ir, child.get("$ref", "")) if isinstance(child, dict) else None
            if item is None or item.get("self_ref") in seen:
                continue
            seen.add(item.get("self_ref"))
            if not item.get("self_ref", "").startswith("#/groups/"):
                yield item
            yield from walk(item)

    for root in ("body", "furniture"):
        yield from walk(ir.get(root) or {})


def _page_height(ir: Dict[str, Any], page_no: int) -> float:
    page = (ir.get("pages") or {}).get(str(page_no)) or {}
    return float((page.get("size") or {}).get("height") or 0.0)


def top_left(box: Dict[str, Any], height: float) -> List[float]:
    """A Docling bbox as `$defs/bbox`: [x0, top, x1, bottom], origin top-left, y down."""
    left, t, right, b = (float(box.get(k, 0.0)) for k in ("l", "t", "r", "b"))
    if str(box.get("coord_origin", "TOPLEFT")).upper() == "BOTTOMLEFT":
        t, b = height - t, height - b
    top, bottom = min(t, b), max(t, b)
    return [round(min(left, right), 3), round(top, 3), round(max(left, right), 3), round(bottom, 3)]


def _centre_in(line: DigitalLine, box: List[float]) -> bool:
    if not line.bbox:
        return False
    cx = (line.bbox[0] + line.bbox[2]) / 2
    cy = (line.bbox[1] + line.bbox[3]) / 2
    return (
        box[0] - _TOLERANCE <= cx <= box[2] + _TOLERANCE
        and box[1] - _TOLERANCE <= cy <= box[3] + _TOLERANCE
    )


def _heading_level(item: Dict[str, Any]) -> Optional[int]:
    label = item.get("label")
    if label == "title":
        return 1
    if label == "section_header":
        try:
            level = int(item.get("level") or 1)
        except (TypeError, ValueError):
            level = 1
        return max(1, min(level, 6))
    return None


# ── the refinement ────────────────────────────────────────────────────────────


def _claim(pool: List[DigitalLine], box: List[float]) -> List[DigitalLine]:
    """Take the light lines whose centre lies in `box` out of `pool`, top to bottom."""
    taken = [line for line in pool if _centre_in(line, box)]
    for line in taken:
        pool.remove(line)
    return sorted(taken, key=lambda ln: (ln.bbox[1], ln.bbox[0]))


def _table(
    item: Dict[str, Any], number: int, page: DigitalPage, pool: List[DigitalLine], height: float
) -> Tuple[DigitalTable, List[DigitalLine]]:
    data = item.get("data") or {}
    group = f"tbl{number}"
    grid = DigitalTable(
        table_id=f"t{number}",
        page=page.page,
        n_rows=int(data.get("num_rows") or 0),
        n_cols=int(data.get("num_cols") or 0),
        group_id=group,
    )
    lines: List[DigitalLine] = []
    owners: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for cell in data.get("table_cells") or []:
        r0, c0 = int(cell.get("start_row_offset_idx", 0)), int(cell.get("start_col_offset_idx", 0))
        r1 = int(cell.get("end_row_offset_idx", r0 + 1))
        c1 = int(cell.get("end_col_offset_idx", c0 + 1))
        cell_group = f"{group}-r{r0}c{c0}"
        entry: Dict[str, Any] = {
            "row": r0,
            "col": c0,
            "is_header": bool(cell.get("column_header")),
            "group_id": cell_group,
        }
        if r1 - r0 > 1:
            entry["rowspan"] = r1 - r0
        if c1 - c0 > 1:
            entry["colspan"] = c1 - c0
        box = top_left(cell["bbox"], height) if cell.get("bbox") else None
        if box:
            entry["bbox"] = box
            claimed = _claim(pool, box)
        else:
            claimed = []
        text = str(cell.get("text") or "").strip()
        if not claimed and text:
            claimed = [DigitalLine(page=page.page, line=0, text=text, bbox=box)]
        for line in claimed:
            line.group_id = cell_group
            line.heading_level = None
            line.region = None
        lines.extend(claimed)
        for r in range(r0, max(r1, r0 + 1)):
            for c in range(c0, max(c1, c0 + 1)):
                owners[(r, c)] = entry
    for (r, c), owner in sorted(owners.items()):
        if (r, c) == (owner["row"], owner["col"]):
            grid.cells.append(owner)
        else:
            grid.cells.append(
                {"row": r, "col": c, "is_header": owner["is_header"], "group_id": owner["group_id"]}
            )
    grid.n_rows = max(grid.n_rows, max((r for r, _ in owners), default=-1) + 1)
    grid.n_cols = max(grid.n_cols, max((c for _, c in owners), default=-1) + 1)
    return grid, lines


def docling_to_digital(ir: Dict[str, Any], light: DigitalDocument) -> DigitalDocument:
    """Lay Docling's structure over the light engine's lines. See the module docstring.

    `light` supplies everything Docling does not describe — the page census (text layer,
    images, labels, sizes) and the line geometry — and is modified in place and returned.
    """
    pages = {page.page_index: page for page in light.pages}
    pools = {index: list(page.lines) for index, page in pages.items()}
    ordered: Dict[int, List[DigitalLine]] = {index: [] for index in pages}
    tables: Dict[int, List[DigitalTable]] = {index: [] for index in pages}
    table_count = 0

    for number, item in enumerate(iter_items(ir)):
        label = item.get("label")
        if label == "picture":
            continue
        for prov in item.get("prov") or []:
            page_no = int(prov.get("page_no") or 0)
            page = pages.get(page_no)
            if page is None or not prov.get("bbox"):
                continue
            height = _page_height(ir, page_no) or page.height
            box = top_left(prov["bbox"], height)
            if label in _TABLE_LABELS:
                grid, lines = _table(item, table_count, page, pools[page_no], height)
                table_count += 1
                if grid.n_rows and grid.n_cols and grid.cells:
                    tables[page_no].append(grid)
                    ordered[page_no].extend(lines)
                continue
            claimed = _claim(pools[page_no], box)
            if not claimed:
                text = str(item.get("text") or "")
                span = prov.get("charspan")
                if span and len(item.get("prov") or []) > 1:
                    text = text[int(span[0]) : int(span[1])]
                text = " ".join(text.split())
                if not text:
                    continue
                claimed = [DigitalLine(page=page.page, line=0, text=text, bbox=box)]
            for line in claimed:
                line.group_id = f"d{number}"
                line.heading_level = _heading_level(item)
                line.region = _REGION_BY_LABEL.get(label)
                line.column = 0
            ordered[page_no].extend(claimed)

    for index, page in pages.items():
        lines = ordered[index]
        # Light lines no Docling item claimed: kept, each before the first ordered line
        # that starts below it — text is never dropped for want of a layout label.
        for stray in sorted(pools[index], key=lambda ln: ln.bbox[1] if ln.bbox else 0.0):
            stray.group_id = None
            top = stray.bbox[1] if stray.bbox else float("inf")
            at = next(
                (i for i, ln in enumerate(lines) if ln.bbox and ln.bbox[1] > top + _TOLERANCE),
                len(lines),
            )
            lines.insert(at, stray)
        page.lines = lines
        page.tables = tables[index]

    light.engine = "docling"
    light.use(*COMPONENTS)
    renumber_lines(light)
    return light


def refine_with_docling(path: str, light: DigitalDocument) -> DigitalDocument:
    """`digital_to_json.extract()`'s entry point for `--engine docling`."""
    return docling_to_digital(convert_with_docling(path), light)
