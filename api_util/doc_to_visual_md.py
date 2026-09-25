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

**One route (Issue #18 / #10 G8, 2026-09-25).** ``.pdf`` and ``.docx`` now go through
``digital_to_json.build_record()`` (in memory, schema-validated) and then
``json_to_md.render_record()`` — the same conversion the hub's born-digital smoke and any
stored ``*.document.json`` go through, so the Markdown the model reads can no longer differ
from the record that is kept. Before, the clients' auto-convert ran the older direct
converters (``docx_to_md`` / ``pdf_to_md``), which miss the garbled-text-layer check, while
the E2E ran the JSON route.

The older converters stay reachable, deprecated:

* ``ocr=True`` on a PDF still runs ``pdf_to_md`` — its opt-in Tesseract path is the only
  OCR in this repo until the ``needs_ocr`` hand-off to the OCR originator (alto-postprocess)
  takes over (#10 plan §10 Phase 3);
* ``legacy=True`` (CLI ``--legacy``) runs ``docx_to_md`` / ``pdf_to_md`` for A/B checks.

``engine`` picks the JSON route's PDF engine: ``light`` (pdfplumber, the default) or
``docling`` (requirements_digital_docling.txt).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from api_util import docx_to_md, json_to_md, pdf_to_md  # noqa: E402
from api_util.digital_ir import DigitalInputError  # noqa: E402

SUPPORTED_EXTENSIONS = frozenset({".docx", ".pdf"})

#: Bumped whenever the Markdown a given input renders to changes, so a cached rendering from
#: an older converter is not served as current (`llm_client_shared.prepare_document_input`).
CONVERTER_VERSION = "2026-09-25.json-route"

#: Not a simple extension — checked separately (see is_supported/convert_to_visual_md).
_DOCUMENT_JSON_SUFFIX = ".document.json"


def is_supported(path: str | Path) -> bool:
    """Whether this file has a visual-MD converter — by extension, or as an
    AtriumDocument record (``*.document.json``)."""
    name = str(path).lower()
    return name.endswith(_DOCUMENT_JSON_SUFFIX) or Path(path).suffix.lower() in SUPPORTED_EXTENSIONS


def convert_to_visual_md(
    path: str | Path,
    ocr: bool = False,
    min_quality: float = 0.0,
    engine: str = "light",
    legacy: bool = False,
) -> str:
    """Convert a DOCX, PDF, or AtriumDocument JSON to visually-rich Markdown.

    ``.pdf``/``.docx`` take the JSON route (see the module docstring); ``ocr`` (PDF only)
    and ``legacy`` select the deprecated direct converters instead. ``min_quality`` drops
    lines below that ``quality_score`` before rendering (JSON route and records). Raises
    ``ValueError`` for an unsupported input or an unrenderable record (``DigitalInputError``
    is one), ``RuntimeError`` with install advice when a backing library is missing.
    """
    name = str(path).lower()
    if name.endswith(_DOCUMENT_JSON_SUFFIX):
        return json_to_md.convert(path, min_quality=min_quality)
    ext = Path(path).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported input '{ext or '(none)'}'. "
            f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}, {_DOCUMENT_JSON_SUFFIX}."
        )
    if legacy or (ocr and ext == ".pdf"):
        return docx_to_md.convert(path) if ext == ".docx" else pdf_to_md.convert(path, ocr=ocr)

    from api_util import digital_to_json  # noqa: PLC0415  (keeps this module's import light)

    record = digital_to_json.build_record(str(path), engine=engine)
    return json_to_md.render_record(record, title=Path(path).stem, min_quality=min_quality)


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
        help="Drop lines below this quality_score (JSON route and AtriumDocument JSON).",
    )
    parser.add_argument(
        "--engine",
        choices=["light", "docling"],
        default="light",
        help="PDF engine of the JSON route: light (pdfplumber) or docling (heavy, opt-in).",
    )
    parser.add_argument(
        "--legacy",
        action="store_true",
        help="Deprecated direct converters (docx_to_md / pdf_to_md) instead of the JSON route.",
    )
    args = parser.parse_args()

    if not args.input_file.exists():
        print(f"Input file not found: {args.input_file}", file=sys.stderr)
        sys.exit(1)

    try:
        rendered = convert_to_visual_md(
            args.input_file,
            ocr=args.ocr,
            min_quality=args.min_quality,
            engine=args.engine,
            legacy=args.legacy,
        )
    except DigitalInputError as exc:
        print(f"{exc.reason}: {exc}", file=sys.stderr)
        sys.exit(2)
    except (ValueError, NotImplementedError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(2)
    except (docx_to_md.DocxNotInstalled, pdf_to_md.PdfPlumberNotInstalled, RuntimeError) as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
        print(f"-> {args.output}")
    else:
        print(rendered)
