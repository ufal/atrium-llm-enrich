#!/usr/bin/env python3
"""
digital_born/probe_pdf_geometry.py — the PDF-geometry A/B harness (Issue #18 §2).

Phase 0 exploration, not shipped code. Replaces `probe_pymupdf.py`, which only printed
PyMuPDF output and so could not answer the question that actually matters.

WHY THIS EXISTS
---------------
Accretion rule 5 merges component licences as a most-restrictive union into
`provenance.license` on every document record, and that field goes to FAIR catalogue
export. Declaring PyMuPDF (AGPL-3.0) as a component of a `digital-convert` run therefore
sets `provenance.license = AGPL-3.0` for EVERY digital-born document. Docling already owns
structural grouping, so PyMuPDF's only unique contribution is precise span geometry — and
pdfplumber (MIT, already vendored) appears to supply the same numbers.

"Appears to" is the whole point. `digital_born/README.md` documented this comparison, with
this filename and these flags, before either existed; the protocol was unrunnable. Now it
runs, so the licence decision is made on measured deltas rather than on an argument.

USAGE
-----
    python digital_born/probe_pdf_geometry.py --engine pdfplumber --pdf sample.pdf > plumber.json
    python digital_born/probe_pdf_geometry.py --engine pymupdf    --pdf sample.pdf > mupdf.json
    python digital_born/probe_pdf_geometry.py --compare plumber.json mupdf.json

Run it on `sample.pdf`, NOT on tests/fixtures/digital/minimal.pdf — the fixture is
single-rotation Helvetica, exactly the easy case where the two libraries agree.

Neither engine is a hard dependency. pdfplumber is in requirements_digital.txt; PyMuPDF is
deliberately absent from it, so install it in a scratch venv (`pip install pymupdf`) only
for the duration of the comparison. A missing engine exits 2 with an explanation rather
than a traceback.

COORDINATE CONVENTION
---------------------
Both extractors emit `atrium_document.schema.json`'s `$defs/bbox` convention —
[x_min, y_min, x_max, y_max], origin TOP-LEFT, y increasing downwards, in points — so the
numbers are directly comparable and directly writable to `lines[].bbox`.

  * pdfplumber gives `top`/`bottom` already top-down. `y0`/`y1` are the bottom-up pair and
    are NOT used here; writing those to a record is the mistake this note exists to prevent.
  * PyMuPDF's `get_text("dict")` is top-down natively.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List, Optional, Tuple

#: Two lines are treated as "the same line" for comparison when their extracted text is
#: identical after whitespace normalisation. Text, not position, is the join key on
#: purpose: position is the thing under measurement.
Line = Dict[str, Any]


def _norm(text: str) -> str:
    return " ".join(str(text).split())


def _round(box: Tuple[float, float, float, float]) -> List[float]:
    return [round(float(v), 3) for v in box]


# ── engines ──────────────────────────────────────────────────────────────────


def extract_pdfplumber(pdf_path: str, max_pages: Optional[int]) -> List[Line]:
    import pdfplumber

    out: List[Line] = []
    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages[:max_pages] if max_pages else pdf.pages
        for p_idx, page in enumerate(pages, start=1):
            # use_text_flow keeps reading order rather than sorting by position, which is
            # what a converter wants for lines[].line numbering.
            words = page.extract_words(use_text_flow=True, extra_attrs=["fontname", "size"])
            rows: Dict[float, List[dict]] = {}
            for w in words:
                # Bucket by rounded `top` — pdfplumber returns words, not lines.
                rows.setdefault(round(float(w["top"]), 1), []).append(w)
            for line_idx, (_top, group) in enumerate(sorted(rows.items())):
                group.sort(key=lambda w: float(w["x0"]))
                text = _norm(" ".join(w["text"] for w in group))
                if not text:
                    continue
                out.append(
                    {
                        "page": p_idx,
                        "line": line_idx,
                        "text": text,
                        "bbox": _round(
                            (
                                min(float(w["x0"]) for w in group),
                                min(float(w["top"]) for w in group),
                                max(float(w["x1"]) for w in group),
                                max(float(w["bottom"]) for w in group),
                            )
                        ),
                        "font": group[0].get("fontname"),
                        "size": round(float(group[0].get("size", 0)), 2),
                    }
                )
    return out


def extract_pymupdf(pdf_path: str, max_pages: Optional[int]) -> List[Line]:
    import fitz  # PyMuPDF — AGPL-3.0, see the module docstring

    out: List[Line] = []
    with fitz.open(pdf_path) as doc:
        limit = min(max_pages, len(doc)) if max_pages else len(doc)
        for p_idx in range(limit):
            page = doc[p_idx]
            line_idx = 0
            for block in page.get_text("dict").get("blocks", []):
                if block.get("type") != 0:  # 0 = text, 1 = image
                    continue
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    text = _norm("".join(s.get("text", "") for s in spans))
                    if not text:
                        continue
                    out.append(
                        {
                            "page": p_idx + 1,
                            "line": line_idx,
                            "text": text,
                            "bbox": _round(tuple(line["bbox"])),
                            "font": spans[0].get("font") if spans else None,
                            "size": round(float(spans[0].get("size", 0)), 2) if spans else 0.0,
                            # PyMuPDF's unique claim: the block grouping a converter would
                            # otherwise have to infer from inter-line gaps.
                            "block": block.get("number"),
                        }
                    )
                    line_idx += 1
    return out


ENGINES = {"pdfplumber": extract_pdfplumber, "pymupdf": extract_pymupdf}


def inter_line_gaps(lines: List[Line]) -> List[dict]:
    """
    Vertical gap between consecutive lines on a page, per page.

    This is the measurement the licence decision actually turns on. PyMuPDF's unique
    contribution over pdfplumber is its block API; if the gap *within* a paragraph and the
    gap *between* paragraphs are cleanly separable from geometry alone, then block
    boundaries are derivable without it and the AGPL dependency buys nothing. On
    tests/fixtures/digital/minimal.pdf that is 2.0 pt vs 34.0 pt — the claim
    digital_born/README.md makes, now produced by the tool rather than quoted at it.
    """
    out: List[dict] = []
    by_page: Dict[int, List[Line]] = {}
    for line in lines:
        by_page.setdefault(line["page"], []).append(line)
    for page in sorted(by_page):
        rows = sorted(by_page[page], key=lambda r: r["bbox"][1])
        gaps = [
            round(nxt["bbox"][1] - cur["bbox"][3], 3)
            for cur, nxt in zip(rows, rows[1:], strict=False)
        ]
        out.append(
            {
                "page": page,
                "gaps_pt": gaps,
                "min_pt": min(gaps) if gaps else None,
                "max_pt": max(gaps) if gaps else None,
                "separable": bool(gaps) and max(gaps) - min(gaps) > 4.0,
            }
        )
    return out


# ── comparison ───────────────────────────────────────────────────────────────


def compare(a: dict, b: dict) -> dict:
    """Join two extractions on normalised line text and report the coordinate deltas."""
    left = {(_norm(r["text"]), r["page"]): r for r in a["lines"]}
    right = {(_norm(r["text"]), r["page"]): r for r in b["lines"]}
    shared = sorted(set(left) & set(right), key=lambda k: (k[1], k[0]))

    deltas: List[dict] = []
    worst = 0.0
    for key in shared:
        la, lb = left[key], right[key]
        per_edge = [abs(x - y) for x, y in zip(la["bbox"], lb["bbox"], strict=False)]
        m = max(per_edge) if per_edge else 0.0
        worst = max(worst, m)
        deltas.append(
            {
                "page": key[1],
                "text": key[0][:60],
                "max_edge_delta_pt": round(m, 3),
                f"{a['engine']}_bbox": la["bbox"],
                f"{b['engine']}_bbox": lb["bbox"],
                "font_agrees": (la.get("font") or "").split("+")[-1]
                == (lb.get("font") or "").split("+")[-1],
            }
        )

    deltas.sort(key=lambda d: -d["max_edge_delta_pt"])
    verdict = (
        "DROP PYMUPDF — deltas are below the 0.5 pt threshold; keep the stack MIT-only "
        "and leave pymupdf out of requirements_digital.txt and para_config.txt."
        if worst < 0.5
        else "DELTAS MATTER — take the AGPL-3.0 question to the maintainers WITH THESE "
        "NUMBERS, and note that accepting it sets provenance.license = AGPL-3.0 for every "
        "digital-born document."
    )
    # Keyed by position, not by engine name: comparing two dumps from the same engine (a
    # useful sanity check — it must report 0.0) would otherwise collapse both counts into
    # one dict key and silently hide half the report.
    return {
        "engines": {"a": a["engine"], "b": b["engine"]},
        "pdf": a.get("pdf"),
        "lines": {"a": len(a["lines"]), "b": len(b["lines"]), "matched": len(shared)},
        "only_in_a": [t for t, _p in sorted(set(left) - set(right))][:20],
        "only_in_b": [t for t, _p in sorted(set(right) - set(left))][:20],
        "max_edge_delta_pt": round(worst, 3),
        "threshold_pt": 0.5,
        "verdict": verdict,
        "worst_20": deltas[:20],
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Extract or compare per-line PDF geometry (Issue #18 §2 licence A/B).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--engine", choices=sorted(ENGINES), help="extract with this engine")
    ap.add_argument("--pdf", help="input PDF (required with --engine)")
    ap.add_argument("--max-pages", type=int, default=None, help="limit pages (default: all)")
    ap.add_argument(
        "--compare",
        nargs=2,
        metavar=("A.json", "B.json"),
        help="compare two extraction dumps produced by --engine",
    )
    args = ap.parse_args(argv)

    if args.compare:
        with open(args.compare[0], encoding="utf-8") as fh:
            a = json.load(fh)
        with open(args.compare[1], encoding="utf-8") as fh:
            b = json.load(fh)
        report = compare(a, b)
        json.dump(report, sys.stdout, indent=2, ensure_ascii=False)
        print()
        print(f"\n{report['verdict']}", file=sys.stderr)
        return 0

    if not args.engine or not args.pdf:
        ap.error("give either --engine ENGINE --pdf FILE, or --compare A.json B.json")

    try:
        lines = ENGINES[args.engine](args.pdf, args.max_pages)
    except ImportError:
        print(
            f"{args.engine} is not installed. pdfplumber is in requirements_digital.txt; "
            f"pymupdf is deliberately NOT (AGPL-3.0 — see that file), so install it in a "
            f"scratch venv just for this comparison: pip install pymupdf",
            file=sys.stderr,
        )
        return 2

    json.dump(
        {
            "engine": args.engine,
            "pdf": args.pdf,
            "bbox_convention": "[x_min, y_min, x_max, y_max], origin top-left, y down, points",
            "inter_line_gaps": inter_line_gaps(lines),
            "lines": lines,
        },
        sys.stdout,
        indent=2,
        ensure_ascii=False,
    )
    print()
    print(f"{args.engine}: {len(lines)} lines from {args.pdf}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
