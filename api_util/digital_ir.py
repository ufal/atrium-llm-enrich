"""
api_util/digital_ir.py — the internal representation shared by the digital-born adapters.

`digital_to_json.py` (Layers B–D and the CLI), `digital_pdf.py` (light PDF engine),
`digital_docx.py` (DOCX) and `digital_docling.py` (heavy PDF engine) all build or read these
dataclasses. They live in their own module rather than in `digital_to_json.py` because that
file is also run as a script: an adapter importing `api_util.digital_to_json` while it runs
as `__main__` would load a SECOND copy of it — two sets of dataclasses, and the module-level
registry check printing twice. `digital_to_json` re-exports every name here, so
`d2j.DigitalLine` and friends keep working for existing callers and tests.

No optional dependency is imported here.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: `lines[].style.region` values this converter writes (Issue #18, 2026-09-25).
#:
#: `lines[].style` is an object only `digital-convert` writes, so a `region` key inside it
#: needed no new ownership grant, and was additive under the hub's versioning rule 1. The
#: canonical schema declares it since 2026-09-25 as a CLOSED enum of exactly these three
#: values (hub `docs/document_schema.md`, changelog 2026-09-25): a new value here must be
#: added there first, or the output gate refuses the record. Absent means body text. Its one
#: consumer is `api_util/json_to_md.py`.
REGION_HEADER = "page_header"
REGION_FOOTER = "page_footer"
REGION_FOOTNOTE = "footnote"
REGIONS: Tuple[str, ...] = (REGION_HEADER, REGION_FOOTER, REGION_FOOTNOTE)

#: `DigitalPage.text_layer` — the per-page census the PDF adapters fill in, mirroring
#: atrium-alto-postprocess `text_formats.classify_text_layer` (#31) so the two tools read a
#: page the same way:
#:
#:   digital — an embedded text layer that is (so far) believed;
#:   none    — a PDF page with no extractable text: a scan, curves-for-letters, or a page
#:             that draws nothing a parser can see — OCR work either way (`images` and
#:             `vector_paths` say which, in the reason);
#:   ocr     — the text layer is INVISIBLE text: a prior OCR run, whose originator is
#:             alto-postprocess (`ocr:pdf-text-layer`), not this converter;
#:   blank   — an empty DOCX page (two breaks in a row); there is nothing to re-acquire.
TEXT_LAYER_DIGITAL = "digital"
TEXT_LAYER_NONE = "none"
TEXT_LAYER_OCR = "ocr"
TEXT_LAYER_BLANK = "blank"


class DigitalInputError(ValueError):
    """An input this converter must not turn into a record, with a stable reason code.

    A `ValueError`, so callers that already catch the "unsupported input" error keep working.
    The codes match atrium-alto-postprocess's `ingest_report.csv` reasons where the two
    overlap, so an operator sees one vocabulary across both tools.
    """

    #: reason -> CLI exit code. 3 = not something this converter takes; 4 = broken file.
    EXIT_CODES: Dict[str, int] = {
        "unsupported": 3,
        "legacy_office_unsupported": 3,
        "ocr_text_layer": 3,
        "encrypted": 4,
        "corrupt": 4,
        "zip_limits_exceeded": 4,
    }

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason

    @property
    def exit_code(self) -> int:
        return self.EXIT_CODES.get(self.reason, 3)


def missing_dependency(
    dependency: str, module: str, requirements: str = "requirements_digital.txt"
) -> RuntimeError:
    """The one error a missing optional import turns into: advice, not an ImportError stack."""
    return RuntimeError(
        f"{dependency} is required to convert this input but is not installed. "
        f"Install the converter stack: pip install -r {requirements} "
        f"(missing import: {module})"
    )


def sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def font_flags(fontname: str) -> Tuple[bool, bool]:
    lowered = (fontname or "").lower()
    return ("bold" in lowered or "black" in lowered or "heavy" in lowered), (
        "italic" in lowered or "oblique" in lowered
    )


def is_inverted(matrix: Any) -> bool:
    """True when a glyph's transformation matrix mirrors or flips it.

    pdfplumber exposes the text matrix as `(a, b, c, d, e, f)`. A negative `a` mirrors
    horizontally and a negative `d` flips vertically; either makes the extracted string
    unreliable as *text* however cleanly it decodes, which is what `Inverted` records.
    """
    if not matrix or len(matrix) < 4:
        return False
    try:
        return float(matrix[0]) < 0 or float(matrix[3]) < 0
    except (TypeError, ValueError):
        return False


@dataclass
class DigitalLine:
    """One text line, before it becomes a `lines[]` row."""

    page: str
    line: int
    text: str
    bbox: Optional[List[float]] = None
    font: str = ""
    size: float = 0.0
    bold: bool = False
    italic: bool = False
    heading_level: Optional[int] = None
    inverted: bool = False
    group_id: Optional[str] = None
    categ: Optional[str] = None
    quality_score: Optional[float] = None
    lang: Optional[str] = None
    #: `style.region` — page furniture or a footnote; None for body text.
    region: Optional[str] = None
    #: Reading-order column on its page (PDF only). Internal: a column change is a
    #: paragraph boundary even when the vertical gap says otherwise.
    column: int = 0


@dataclass
class DigitalTable:
    """One table grid. `cells[].group_id` is the join key back into `lines[]`."""

    table_id: str
    page: str
    caption: str = ""
    n_rows: int = 0
    n_cols: int = 0
    group_id: Optional[str] = None
    cells: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DigitalPage:
    page: str
    page_index: int
    width: float = 0.0
    height: float = 0.0
    unit: str = "pt"
    lines: List[DigitalLine] = field(default_factory=list)
    tables: List[DigitalTable] = field(default_factory=list)
    needs_ocr: bool = False
    needs_ocr_reason: str = ""
    quality_score: Optional[float] = None
    quality_band: Optional[str] = None
    #: Per-page census (PDF). DOCX pages are always `digital`.
    text_layer: str = TEXT_LAYER_DIGITAL
    images: int = 0
    vector_paths: int = 0
    text_objects: int = 0
    invisible_text_objects: int = 0

    @property
    def has_geometry(self) -> bool:
        return any(line.bbox for line in self.lines)


@dataclass
class DigitalDocument:
    doc_id: str
    origin: str
    media_type: str
    pages: List[DigitalPage] = field(default_factory=list)
    sha256: str = ""
    filename: str = ""
    reading_order: str = "layout"
    #: Which engine produced the structure: `light` (pdfplumber / python-docx) or `docling`.
    engine: str = "light"
    #: `para_config.txt` component names this run actually used — the licence union's input.
    components: List[str] = field(default_factory=list)

    def all_lines(self) -> List[DigitalLine]:
        return [line for page in self.pages for line in page.lines]

    def use(self, *names: str) -> None:
        for name in names:
            if name not in self.components:
                self.components.append(name)


def renumber_lines(document: DigitalDocument) -> None:
    """Number every page's lines 0..n-1 in their final reading order, in place.

    `(page, line)` is the key `merge_block()` and every downstream consumer address a line
    by, so it has to be unique per page and follow reading order — which only the adapter's
    final assembly (columns, furniture, footnotes) knows.
    """
    for page in document.pages:
        for number, line in enumerate(page.lines):
            line.page = page.page
            line.line = number
