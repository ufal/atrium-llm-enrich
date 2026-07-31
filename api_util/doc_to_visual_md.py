"""
api_util/doc_to_visual_md.py — DOCX / PDF → visually-rich Markdown (dispatcher).

Single entry point for issue #10's document front-end: routes an input file to
the right converter by extension and returns page-sectioned Markdown enriched
with the visual-layout cues from ``layout_md.py`` (page borders, bounding boxes,
fonts, alignment, headers/footers, tables, …).

Meant to be used exactly like ``xml_to_md.py`` / ``flexiconv_convert.py``:
**pre-convert, then run**. Land the resulting ``.md`` in ``INPUT_DIR`` and the
document-level pipeline (``run_document_level()`` in llm_client_shared.py, for
BACKEND=openrouter / ollama) picks it up with no dispatch change — the HTML-
comment cues are inert text that passes straight through to the LLM.

    python3 api_util/doc_to_visual_md.py report.docx --output INPUT_DIR/report.md
    python3 api_util/doc_to_visual_md.py report.pdf  --output INPUT_DIR/report.md

Scope (first pass): DOCX and digital-born PDF. Scanned / curve-only PDF pages are
marked with a ``NEEDS_OCR`` cue rather than transcribed; the OCR path is a
benchmark-gated follow-up (hub ``atrium-project#22``).

A fourth source joined the dispatcher later: ``*.document.json``, an
AtriumDocument record (issue #13's JSON plane), via ``json_to_md.py`` — the
regenerable-recipe path for a consumer that holds only the JSON, not a stored
TEITOK/PDF/DOCX file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from api_util import docx_to_md, json_to_md, pdf_to_md  # noqa: E402

SUPPORTED_EXTENSIONS = frozenset({".docx", ".pdf"})

#: Not a simple extension — checked separately (see is_supported/convert_to_visual_md).
_DOCUMENT_JSON_SUFFIX = ".document.json"


def is_supported(path: str | Path) -> bool:
    """Whether this file has a visual-MD converter — by extension, or as an
    AtriumDocument record (``*.document.json``)."""
    name = str(path).lower()
    return name.endswith(_DOCUMENT_JSON_SUFFIX) or Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def convert_to_visual_md(path: str | Path, ocr: bool = False, min_quality: float = 0.0) -> str:
    """Convert a DOCX, PDF, or AtriumDocument JSON to visually-rich Markdown.

    ``ocr`` (PDF only) transcribes scanned / curve-only pages with Tesseract
    instead of flagging them ``NEEDS_OCR``. ``min_quality`` (JSON only) drops
    lines below that ``quality_score`` before rendering. Raises ``ValueError``
    for unsupported extensions or an unrenderable record, and the converter's
    own ``*NotInstalled`` error when a backing library is missing.
    """
    name = str(path).lower()
    if name.endswith(_DOCUMENT_JSON_SUFFIX):
        return json_to_md.convert(path, min_quality=min_quality)
    ext = Path(path).suffix.lower()
    if ext == ".docx":
        return docx_to_md.convert(path)
    if ext == ".pdf":
        return pdf_to_md.convert(path, ocr=ocr)
    raise ValueError(
        f"Unsupported input '{ext or '(none)'}'. "
        f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}, {_DOCUMENT_JSON_SUFFIX}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input_file", type=Path)
    parser.add_argument(
        "--output", type=Path, default=None, help="Write to file instead of stdout."
    )
    parser.add_argument(
        "--ocr", action="store_true", help="PDF only: transcribe text-less pages with Tesseract."
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.0,
        help="AtriumDocument JSON only: drop lines below this quality_score.",
    )
    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    try:
        rendered = convert_to_visual_md(args.input_file, ocr=args.ocr, min_quality=args.min_quality)
    except (ValueError, NotImplementedError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)
    except (docx_to_md.DocxNotInstalled, pdf_to_md.PdfPlumberNotInstalled) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"-> {args.output}")
    else:
        print(rendered)
