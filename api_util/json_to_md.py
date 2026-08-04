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
from typing import Dict, List, Optional, Tuple

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from api_util import xml_to_md  # noqa: E402
from atrium_document import load_document  # noqa: E402

#: Line categories excluded from the rendered text by default — heuristic
#: OCR/layout noise (see alto-postprocess's compute_quality_score) that would
#: only degrade the LLM's read of the page, not inform it.
DROP_CATEGORIES = frozenset({"Garbage", "Inverted"})

#: The only implemented profile today. "standard"/"minimal" are the down-profiles
#: from issue #13 §B, still deferred across the whole MD front-end — accepted
#: here for CLI/API parity with the schema's regenerable.detail enum, but not
#: silently downgraded to "full" if requested.
IMPLEMENTED_DETAIL_LEVELS = frozenset({"full"})


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
        rows.append(row)
    rows.sort(key=lambda r: (r["page_num"], r["line_num"]))
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


def read_document_rows(
    doc_json_path: str | Path, min_quality: float = 0.0
) -> Tuple[List[dict], Dict[int, dict], Optional[str]]:
    """
    Read ``(rows, pages, fallback_text)`` off an AtriumDocument JSON —
    ``pages``/``lines``/``content`` only, never ``enrichment``.

    ``rows``/``pages`` are in exactly the shape ``xml_to_md.rows_to_layout_markdown``
    expects (mirroring ``read_document_layout``'s TEITOK/ALTO shape), so the
    renderer is shared rather than re-implemented. ``fallback_text`` is set only
    when ``lines[]`` yielded nothing usable.
    """
    record = load_document(str(doc_json_path))
    lines = record.get("lines") or []
    page_block = record.get("pages") or []
    # One ordinal map shared by both, so a line and its page's metadata cannot end up under
    # different keys — which is what would happen if each derived its own.
    ordinals = page_ordinals(page_block, lines)
    rows = _rows_from_lines(lines, min_quality, ordinals)
    pages = _pages_meta(page_block, ordinals)
    fallback_text = None
    if not rows:
        fallback_text = (record.get("content") or {}).get("text")
    return rows, pages, fallback_text


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

    rows, pages, fallback_text = read_document_rows(doc_json_path, min_quality=min_quality)

    if rows:
        return xml_to_md.rows_to_layout_markdown(rows, pages, title=doc_id)

    if fallback_text and fallback_text.strip():
        print(
            f"[json_to_md] {doc_id}: no usable lines[] in the record (alto-postprocess's "
            f"quality/categ pass may not have run, or --min-quality={min_quality} dropped "
            f"everything) — falling back to unsectioned content.text.",
            file=sys.stderr,
        )
        return f"# {doc_id}\n\n{fallback_text.strip()}\n"

    raise ValueError(
        f"{doc_id}: record has neither a usable lines[] nor content.text — nothing to "
        f"render. Has any pipeline stage populated this document's text yet?"
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
