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


def _rows_from_lines(lines: List[dict], min_quality: float) -> List[dict]:
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
        try:
            page_num = int(line["page"])
        except (KeyError, TypeError, ValueError):
            continue
        row = {"page_num": page_num, "line_num": line.get("line", 0), "text": text}
        if line.get("bbox"):
            row["bbox"] = line["bbox"]
        rows.append(row)
    rows.sort(key=lambda r: (r["page_num"], r["line_num"]))
    return rows


def _pages_meta(pages: List[dict]) -> Dict[int, dict]:
    meta: Dict[int, dict] = {}
    for p in pages:
        try:
            page_num = int(p["page"])
        except (KeyError, TypeError, ValueError):
            continue
        canvas = p.get("canvas") or {}
        entry: Dict[str, object] = {}
        if canvas.get("width") and canvas.get("height"):
            entry["width"] = canvas["width"]
            entry["height"] = canvas["height"]
        if p.get("needs_ocr"):
            entry["needs_ocr"] = True
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
    rows = _rows_from_lines(lines, min_quality)
    pages = _pages_meta(record.get("pages") or [])
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
