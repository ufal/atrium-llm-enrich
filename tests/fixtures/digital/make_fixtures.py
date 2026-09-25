#!/usr/bin/env python3
"""
tests/fixtures/digital/make_fixtures.py

Issue #18 §5 — generates the minimal, deterministic digital-born fixtures the golden
tests compare against.

Why generated rather than committed binaries: `digital_born/sample.pdf` and `sample.docx`
are ~2-3 MB real-world documents. They are fine for exploration and useless as golden
fixtures — an `expected.json` diff over a few hundred lines of real layout is unreviewable,
and nobody can tell an intentional change from a regression. These are a few hundred bytes
each, every glyph position is chosen deliberately, and the bytes are reproducible, so a
golden diff is readable by a human in a code review.

Why hand-rolled rather than reportlab / docx-js / python-docx:

  1. **Byte determinism.** Every generator worth using stamps a creation date and a
     document ID. `git diff` on a regenerated fixture has to be empty or the fixture is
     not a fixture. Here there is no date, no /ID, and the zip entries carry a fixed
     1980-01-01 timestamp.
  2. **Exact control of the font dictionary.** `garbled.pdf` has to declare
     /WinAnsiEncoding over Central-European bytes with no /ToUnicode map, because that is
     the specific construction Issue #10's research pass found in the wild — non-embedded
     WinAnsi Helvetica producing systematic (not random) diacritic corruption across
     *every* text parser. No high-level API lets you build a deliberately broken font
     dict, which is exactly what needs testing.

For anything that is not a golden fixture, use reportlab / python-docx as normal.

Usage:
    python tests/fixtures/digital/make_fixtures.py            # write fixtures + manifest
    python tests/fixtures/digital/make_fixtures.py --verify    # fail on drift
    python tests/fixtures/digital/make_fixtures.py --outdir /tmp/fx   # scratch bytes only

The sha256 manifest is committed at `tests/fixtures/MANIFEST.json` and its path is
INDEPENDENT of `--outdir`: the bytes are scratch, the manifest is the repo's canonical
record of what they must be. Override it only with an explicit `--manifest`.

The four fixtures and what each one is for:

    minimal.pdf   2 pages, 3 text blocks, clean ASCII-safe Latin. The happy path: exact
                  bboxes, a real page break, and blocks that must land as distinct
                  `lines[].group_id` values. Its text is deliberately CONTENTLESS
                  ("block one, line one") so a bbox diff is readable — which is exactly
                  why it is the wrong input for a semantic stage; see enrichable.pdf.
    enrichable.pdf
                  1 page of diacritic-free Czech archaeological prose. The born-digital
                  E2E's llm-enrich input (atrium-project#49): a fixture the semantic
                  stage can actually find something in, so a green digital smoke means
                  more than "the plumbing connects". Diacritic-free is load-bearing —
                  see the builder's docstring.
    garbled.pdf   1 page, Czech text, /WinAnsiEncoding declared over cp1250 bytes, no
                  /ToUnicode. Text extraction SUCCEEDS and returns wrong characters — the
                  case that must trip the decode-sanity check and set
                  `pages[].needs_ocr = true` rather than pass silently downstream.
    minimal.docx  Heading + 2 paragraphs + a 2x2 table + an explicit page break + a
                  paragraph after it. Structural truth with no reliable geometry, i.e.
                  §3's "do not fabricate bounding boxes" rule under test.

Added 2026-09-25 for the #10 §9 parity pass (gap register G1–G7), one fixture per failure
the measurement found on the JSON route:

    image_only.pdf  a text page, an image-only page and an empty page — both needs_ocr, with
                    reasons that tell them apart.
    two_column.pdf  two columns under a full-width bold title, a running header, a
                    page-number footer, kerned words without spaces, roman page labels.
    ocr_layer.pdf   page images under an invisible text layer — OCR output, not born-digital.
    table.pdf       a ruled table pdfplumber's line strategy finds.
    rich.docx       headings by style name and outline level, emphasis via a character style,
                    tracked changes, footnote, text box, header/footer, three kinds of page
                    break, a merged cell.
    bare.docx       a package without the main part's content-type Override.

The four fixtures above keep their bytes (and their hub-pinned sha256s): the new PDFs use
`_build_pdf_ex`, and `_build_pdf` is unchanged except for sharing `_serialize_pdf`.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

#: Fixed zip timestamp. Any real clock value makes the DOCX non-reproducible.
ZIP_EPOCH: Tuple[int, int, int, int, int, int] = (1980, 1, 1, 0, 0, 0)

#: Pinned so the DOCX bytes do not depend on the machine that generated them.
#:
#: `zipfile.ZipInfo.__init__` sets `create_system = 0` on win32 and `3` (Unix) everywhere
#: else, and that byte lands in every local and central directory header. So the sha256 in
#: MANIFEST.json was silently platform-specific: a contributor on Windows regenerating the
#: fixture would see `--verify` fail with a "drift" that is not a content change at all.
#: 3 (Unix) matches what CI and every developer machine here already produced, so the
#: existing hash is unaffected on those platforms.
ZIP_CREATE_SYSTEM = 3

#: STORED, not DEFLATED.
#:
#: DEFLATE output is not specified byte-for-byte — it depends on the zlib version and build
#: options linked into the interpreter. A fixture whose whole premise is "git diff on a
#: regenerated fixture has to be empty" cannot be compressed with a codec whose output may
#: legitimately change under a zlib upgrade. The six XML parts total a few KB, so storing
#: them uncompressed costs nothing and removes the entire class of drift. This DOES change
#: the fixture's bytes and size once, which is why MANIFEST.json is regenerated in the same
#: commit — python-docx, Word and every conformant reader accept stored entries.
ZIP_COMPRESSION = zipfile.ZIP_STORED

#: Letter, in points. Matches pages[].canvas {width, height, unit: "pt"}.
PAGE_W, PAGE_H = 612, 792


# ── PDF ──────────────────────────────────────────────────────────────────────


def _pdf_escape(raw: bytes) -> bytes:
    """Escape a PDF literal string. Order matters: backslash first."""
    out = raw.replace(b"\\", b"\\\\")
    out = out.replace(b"(", b"\\(").replace(b")", b"\\)")
    return out


def _text_block(x: int, y: int, lines: List[bytes], leading: int = 14) -> bytes:
    """One BT/ET block. Separate blocks at separate y positions are what a text-block
    extractor groups into separate `group_id`s, so the block boundaries here ARE the
    assertion."""
    parts = [b"BT", b"/F1 12 Tf", f"{x} {y} Td".encode("ascii"), f"{leading} TL".encode("ascii")]
    for i, line in enumerate(lines):
        if i:
            parts.append(b"T*")
        parts.append(b"(" + _pdf_escape(line) + b") Tj")
    parts.append(b"ET")
    return b"\n".join(parts)


def _build_pdf(pages: List[bytes], font_obj: bytes) -> bytes:
    """
    Assemble a PDF from per-page content streams, computing the xref offsets.

    Object layout: 1 catalog, 2 page tree, then (page, contents) per page, then the font.
    No /Info, no /ID, no dates — that is what keeps the output reproducible.
    """
    n_pages = len(pages)
    font_num = 3 + 2 * n_pages
    kid_nums = [3 + 2 * i for i in range(n_pages)]

    objects: Dict[int, bytes] = {
        1: b"<< /Type /Catalog /Pages 2 0 R >>",
        2: (
            b"<< /Type /Pages /Kids ["
            + b" ".join(f"{n} 0 R".encode("ascii") for n in kid_nums)
            + f"] /Count {n_pages} >>".encode("ascii")
        ),
        font_num: font_obj,
    }

    for i, stream in enumerate(pages):
        page_num = 3 + 2 * i
        contents_num = page_num + 1
        objects[page_num] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] "
            f"/Resources << /Font << /F1 {font_num} 0 R >> >> "
            f"/Contents {contents_num} 0 R >>".encode("ascii")
        )
        objects[contents_num] = _stream_obj(stream)

    return _serialize_pdf(objects)


def _stream_obj(stream: bytes, dictionary: bytes = b"") -> bytes:
    """A stream object: `<< [dictionary] /Length n >> stream … endstream`."""
    head = b"<< " + (dictionary + b" " if dictionary else b"")
    return head + f"/Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream"


def _serialize_pdf(objects: Dict[int, bytes]) -> bytes:
    """Write numbered objects, the xref table and the trailer. Object 1 is the catalog."""
    out = bytearray(b"%PDF-1.4\n")
    # A binary comment marks the file as non-ASCII so naive tools stop "fixing" line endings.
    out += b"%\xe2\xe3\xcf\xd3\n"

    offsets: Dict[int, int] = {}
    for num in sorted(objects):
        offsets[num] = len(out)
        out += f"{num} 0 obj\n".encode("ascii") + objects[num] + b"\nendobj\n"

    xref_at = len(out)
    highest = max(objects)
    out += f"xref\n0 {highest + 1}\n".encode("ascii")
    out += b"0000000000 65535 f \n"
    for num in range(1, highest + 1):
        out += f"{offsets[num]:010d} 00000 n \n".encode("ascii")
    out += f"trailer\n<< /Size {highest + 1} /Root 1 0 R >>\nstartxref\n{xref_at}\n%%EOF\n".encode(
        "ascii"
    )
    return bytes(out)


#: A base-14 Type1 font with no /Encoding override and no /ToUnicode. Fine for ASCII.
FONT_CLEAN = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"

#: The Issue #10 construction: WinAnsi declared, no /ToUnicode, and the content stream
#: carries cp1250 (Central European) bytes. A parser trusting the declaration decodes
#: 0xEC as U+00EC 'i-grave' instead of U+011B 'e-caron'. Extraction SUCCEEDS and lies.
FONT_MISDECLARED = (
    b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>"
)


def minimal_pdf() -> bytes:
    """Two pages, three text blocks, no diacritics — the clean happy path."""
    page1 = b"\n".join(
        [
            _text_block(
                72,
                720,
                [
                    b"Excavation report, block one, line one.",
                    b"Excavation report, block one, line two.",
                ],
            ),
            _text_block(72, 660, [b"Block two is a separate paragraph."]),
        ]
    )
    page2 = _text_block(72, 720, [b"Page two, block three, only line."])
    return _build_pdf([page1, page2], FONT_CLEAN)


def enrichable_pdf() -> bytes:
    """One page of real archaeological content, for the LLM stage rather than the geometry one.

    Added for atrium-project#49. The born-digital E2E (atrium-project's
    e2e-digital-smoke.yml) fed `minimal.pdf` to llm-enrich, and `minimal.pdf` says
    "Excavation report, block one, line one." — placeholder chosen to make BLOCK BOUNDARIES
    reviewable, with no site, find, method, period or material anywhere in it. The
    document-level prompt's own instruction for that input is "If the document has no
    archaeologically relevant passages, return an empty items list", so run 34090340995's
    `records enriched: 0` was the model being RIGHT. The smoke was asserting a semantic
    outcome against a fixture built to pin bboxes.

    So this is a separate fixture rather than a rewrite of `minimal.pdf`: that one's bytes
    are a geometry golden and there is no reason to disturb them, and one fixture doing two
    unrelated jobs is how the confusion started.

    NO DIACRITICS, and that is a hard constraint, not a style choice. FONT_CLEAN is base-14
    Helvetica with no /Encoding override and no /ToUnicode, i.e. StandardEncoding. Czech
    text written through it comes back from every extractor as mojibake — measured, not
    assumed:

        "Zpráva o sondě číslo 3."  ->  'ZprÆva o sond(cid:236) Ł(cid:237)slo 3.'

    and decode-sanity does NOT flag it (needs_ocr stayed False), so such a fixture would
    quietly break the happy-path contract instead of failing loudly. Diacritic-free Czech
    is what the corpus's own legacy digitisations look like anyway, it round-trips
    byte-exactly through StandardEncoding, and it keeps the vocabulary in the language the
    TEATER/AMCR terms are actually written in. Getting diacritics in here needs a
    /ToUnicode CMap, which is a deliberate change to `_build_pdf`, not a change to a string.
    """
    return _build_pdf(
        [
            _text_block(
                72,
                720,
                [
                    b"Zprava o zachrannem archeologickem vyzkumu.",
                    b"Lokalita: hradiste u Horni Mezi, okres Beroun.",
                    b"Sonda II odkryla cast valoveho telesa.",
                    b"Mocnost kulturni vrstvy cinila 40 cm.",
                    b"Nalezeny zlomky keramiky z raneho stredoveku,",
                    b"zelezne hreby a mazanice z vypalene hliny.",
                ],
            )
        ],
        FONT_CLEAN,
    )


def garbled_pdf() -> bytes:
    """One page of Czech in cp1250 bytes under a /WinAnsiEncoding declaration.

    The exact mojibake differs per parser (Issue #10 observed `sondě` -> `sondI`; a strict
    WinAnsi table gives `sondì`). What is stable, and what the fixture is actually pinning,
    is the CLASS of failure: extraction returns a plausible-looking string with
    systematically wrong diacritics, so nothing short of a decode-sanity check notices.
    """
    czech = [
        "Zpráva o sondě číslo 3.",
        "Nalezeny hřeby a zlomky keramiky.",
        "Vrstva ornice měla mocnost 30 cm.",
    ]
    body = [line.encode("cp1250") for line in czech]
    return _build_pdf([_text_block(72, 720, body)], FONT_MISDECLARED)


# ── PDF: the #10 §9 parity fixtures (Issue #18, 2026-09-25) ──────────────────
#
# Built with `_build_pdf_ex` rather than `_build_pdf`, which stays byte-for-byte what it was:
# the four fixtures above are sha256-pinned in the hub's digital smoke as well as here.

#: Bold base-14 face — what a PDF heading usually is, and what `_font_flags` reads.
FONT_BOLD = b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"

#: A 2x2 8-bit grey image. Its bytes are irrelevant; what matters is that a page DRAWS an
#: image, which is how a scan (or a scan under an OCR text layer) presents to a parser.
IMAGE_XOBJECT = _stream_obj(
    b"\x00\x80\x80\xff",
    b"/Type /XObject /Subtype /Image /Width 2 /Height 2 /ColorSpace /DeviceGray "
    b"/BitsPerComponent 8",
)

#: Draw /Im1 over the whole page — the layout of every scanned page.
FULL_PAGE_IMAGE = f"q {PAGE_W} 0 0 {PAGE_H} 0 0 cm /Im1 Do Q".encode("ascii")


def _tj(parts: List[object]) -> bytes:
    """A TJ array: strings and kerning numbers. A negative number moves the pen RIGHT, so
    `[(Sonda) -400 (II)] TJ` puts a visible gap between the words with NO space character in
    the text — the construction that made the old char-joining extractor glue words."""
    items = []
    for part in parts:
        if isinstance(part, bytes):
            items.append(b"(" + _pdf_escape(part) + b")")
        else:
            items.append(str(part).encode("ascii"))
    return b"[" + b" ".join(items) + b"] TJ"


def _text_at(
    x: float,
    y: float,
    lines: List[object],
    font: str = "F1",
    size: int = 12,
    leading: int = 14,
    render: int = 0,
) -> bytes:
    """One BT/ET block with a chosen font, size and text render mode (3 = invisible, the
    mode an OCR engine writes its text layer in). A line is `bytes` (Tj) or a list (TJ)."""
    parts = [b"BT"]
    if render:
        parts.append(f"{render} Tr".encode("ascii"))
    parts += [f"/{font} {size} Tf".encode("ascii"), f"{x} {y} Td".encode("ascii")]
    parts.append(f"{leading} TL".encode("ascii"))
    for i, line in enumerate(lines):
        if i:
            parts.append(b"T*")
        parts.append(_tj(line) if isinstance(line, list) else b"(" + _pdf_escape(line) + b") Tj")
    parts.append(b"ET")
    return b"\n".join(parts)


def _build_pdf_ex(pages: List[bytes], page_labels: bytes = b"") -> bytes:
    """Like `_build_pdf`, with two fonts (/F1 Helvetica, /F2 Helvetica-Bold), one image
    (/Im1) available on every page, and an optional /PageLabels number tree.

    Object layout: 1 catalog, 2 page tree, (page, contents) per page, then F1, F2, Im1.
    """
    n_pages = len(pages)
    f1, f2, im = 3 + 2 * n_pages, 4 + 2 * n_pages, 5 + 2 * n_pages
    kids = b" ".join(f"{3 + 2 * i} 0 R".encode("ascii") for i in range(n_pages))
    catalog = b"<< /Type /Catalog /Pages 2 0 R" + (
        b" /PageLabels " + page_labels if page_labels else b""
    )
    objects: Dict[int, bytes] = {
        1: catalog + b" >>",
        2: b"<< /Type /Pages /Kids [" + kids + f"] /Count {n_pages} >>".encode("ascii"),
        f1: FONT_CLEAN,
        f2: FONT_BOLD,
        im: IMAGE_XOBJECT,
    }
    resources = (
        f"/Resources << /Font << /F1 {f1} 0 R /F2 {f2} 0 R >> /XObject << /Im1 {im} 0 R >> >>"
    )
    for i, stream in enumerate(pages):
        page_num = 3 + 2 * i
        objects[page_num] = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {PAGE_W} {PAGE_H}] {resources} "
            f"/Contents {page_num + 1} 0 R >>".encode("ascii")
        )
        objects[page_num + 1] = _stream_obj(stream)
    return _serialize_pdf(objects)


def image_only_pdf() -> bytes:
    """G1: a text page, a page that is ONE image and nothing else, and a page that draws nothing.

    The image-only page is the scanned-plate case — a born-digital report with a scanned
    figure page. It has no text layer, so it must come out `needs_ocr` with the reason "no
    extractable text layer", not scored Clear/1.0 and dropped. The empty page is flagged too
    (alto-postprocess #31 classifies any text-less page as `none`), with a reason that says it
    draws nothing a parser can see — that is how the two are told apart.
    """
    text = _text_at(72, 720, [b"Zprava o vyzkumu, textova strana."])
    return _build_pdf_ex([text, FULL_PAGE_IMAGE, b""])


def two_column_pdf() -> bytes:
    """G2 and the layout cues: two pages of two-column text under a full-width title.

    * a running header repeated on both pages and a page-number footer — `style.region`;
    * a bold 16 pt title that crosses the gutter — a full-width band read first, and a
      heading by size;
    * a left column with two paragraphs, then a right column — column-major reading order,
      the right column never glued onto the left one's lines;
    * one line written as a TJ array with kerning gaps and no space characters — words
      must still come out separated;
    * /PageLabels with lower-case roman numerals — the labels are `i` and `ii`, the
      physical positions 1 and 2.
    """
    header = _text_at(72, 765, [b"Zprava o vyzkumu Horni Mez"], size=9)

    def footer(n: int) -> bytes:
        return _text_at(300, 40, [str(n).encode("ascii")], size=9)

    title = _text_at(72, 715, [b"Hradiste u Horni Mezi: zachranny vyzkum"], font="F2", size=16)
    left1 = _text_at(
        72,
        680,
        [
            [b"Sonda", -400, b"II", -400, b"odkryla", -400, b"val."],
            b"Vrstva ornice byla tenka.",
            b"Pod ni lezela hlina.",
        ],
    )
    left2 = _text_at(72, 620, [b"Druhy odstavec vlevo.", b"Konec leveho sloupce."])
    right = _text_at(
        324,
        680,
        [b"Pravy sloupec zacina zde.", b"Nalezy keramiky a mazanice.", b"Konec praveho sloupce."],
    )
    page1 = b"\n".join([header, title, left1, left2, right, footer(1)])
    page2 = b"\n".join(
        [
            header,
            _text_at(72, 715, [b"Strana dve, levy sloupec.", b"Dalsi radek vlevo."]),
            _text_at(324, 715, [b"Strana dve, pravy sloupec.", b"Dalsi radek vpravo."]),
            footer(2),
        ]
    )
    return _build_pdf_ex([page1, page2], page_labels=b"<< /Nums [0 << /S /r >>] >>")


def ocr_layer_pdf() -> bytes:
    """G7: two pages that are a page image with an INVISIBLE text layer (render mode 3).

    That is how OCRmyPDF, ABBYY and Acrobat write a searchable scan. The text extracts
    perfectly well, which is why the old converter stamped it `digital-born-pdf` — but it is
    OCR output, and its originator is alto-postprocess (`ocr:pdf-text-layer`).
    """

    def page(text: bytes) -> bytes:
        return FULL_PAGE_IMAGE + b"\n" + _text_at(72, 720, [text], render=3)

    return _build_pdf_ex([page(b"Naskenovana strana jedna."), page(b"Naskenovana strana dve.")])


def table_pdf() -> bytes:
    """A ruled 3x2 table between two paragraphs — pdfplumber's line-strategy table finder.

    The legacy `pdf_to_md` emitted tables found this way; the JSON route has to as well, or
    switching the auto-convert would lose them. Rules are drawn as stroked path segments.
    """
    cols = [72, 192, 312, 432]
    rows = [660, 640, 620]
    rules = [b"0.5 w"]
    for y in rows:
        rules.append(f"{cols[0]} {y} m {cols[-1]} {y} l S".encode("ascii"))
    for x in cols:
        rules.append(f"{x} {rows[-1]} m {x} {rows[0]} l S".encode("ascii"))
    cells = [
        ["Vrstva", "Mocnost", "Nalezy"],
        ["Ornice", "30 cm", "keramika"],
    ]
    text = []
    for r, row in enumerate(cells):
        for c, value in enumerate(row):
            text.append(_text_at(cols[c] + 4, rows[r + 1] + 6, [value.encode("ascii")], size=10))
    page = b"\n".join(
        [
            _text_at(72, 720, [b"Tabulka vrstev je nize."]),
            *rules,
            *text,
            _text_at(72, 580, [b"Text pod tabulkou."]),
        ]
    )
    return _build_pdf_ex([page])


# ── DOCX ─────────────────────────────────────────────────────────────────────

W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
XML_DECL = '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'

CONTENT_TYPES = XML_DECL + (
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
    '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
    '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
    "</Types>"
)

ROOT_RELS = XML_DECL + (
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>'
    '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>'
    "</Relationships>"
)

DOC_RELS = XML_DECL + (
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    "</Relationships>"
)

STYLES = XML_DECL + (
    f'<w:styles xmlns:w="{W_NS}">'
    '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>'
    '<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/>'
    '<w:basedOn w:val="Normal"/><w:pPr><w:outlineLvl w:val="0"/></w:pPr>'
    '<w:rPr><w:b/><w:sz w:val="32"/></w:rPr></w:style>'
    "</w:styles>"
)

#: Fixed dates. dcterms values are the DOCX equivalent of a PDF /CreationDate.
CORE_PROPS = XML_DECL + (
    "<cp:coreProperties "
    'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
    'xmlns:dc="http://purl.org/dc/elements/1.1/" '
    'xmlns:dcterms="http://purl.org/dc/terms/" '
    'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
    "<dc:title>Zpráva o sondě</dc:title>"
    "<dc:creator>ATRIUM test fixture</dc:creator>"
    '<dcterms:created xsi:type="dcterms:W3CDTF">1980-01-01T00:00:00Z</dcterms:created>'
    '<dcterms:modified xsi:type="dcterms:W3CDTF">1980-01-01T00:00:00Z</dcterms:modified>'
    "</cp:coreProperties>"
)


def _p(text: str, style: str | None = None) -> str:
    ppr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
    return f'<w:p>{ppr}<w:r><w:t xml:space="preserve">{text}</w:t></w:r></w:p>'


def _cell(text: str, width: int = 4535) -> str:
    return (
        f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/></w:tcPr>'
        f'<w:p><w:r><w:t xml:space="preserve">{text}</w:t></w:r></w:p></w:tc>'
    )


def _document_xml() -> str:
    table = (
        "<w:tbl>"
        '<w:tblPr><w:tblW w:w="9070" w:type="dxa"/></w:tblPr>'
        '<w:tblGrid><w:gridCol w:w="4535"/><w:gridCol w:w="4535"/></w:tblGrid>'
        f"<w:tr>{_cell('Vrstva')}{_cell('Mocnost')}</w:tr>"
        f"<w:tr>{_cell('Ornice')}{_cell('30 cm')}</w:tr>"
        "</w:tbl>"
    )
    page_break = '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'
    body = (
        _p("Zpráva o sondě", style="Heading1")
        # Two runs in ONE paragraph: the adapter must emit these as one group_id, not two.
        + '<w:p><w:r><w:t xml:space="preserve">První odstavec, věta jedna. </w:t></w:r>'
        + '<w:r><w:t xml:space="preserve">První odstavec, věta dvě.</w:t></w:r></w:p>'
        + _p("Druhý odstavec s hřeby a kamením.")
        + table
        + page_break
        + _p("Třetí odstavec, už na druhé straně.")
        + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/></w:sectPr>'
    )
    return XML_DECL + f'<w:document xmlns:w="{W_NS}"><w:body>{body}</w:body></w:document>'


def minimal_docx() -> bytes:
    """A .docx is a zip of XML parts; fixed entry order, fixed timestamps, a pinned
    create_system byte and no compression make it byte-reproducible, which no high-level
    writer will do for you."""
    parts = [
        ("[Content_Types].xml", CONTENT_TYPES),
        ("_rels/.rels", ROOT_RELS),
        ("docProps/core.xml", CORE_PROPS),
        ("word/_rels/document.xml.rels", DOC_RELS),
        ("word/document.xml", _document_xml()),
        ("word/styles.xml", STYLES),
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=ZIP_COMPRESSION) as zf:
        for name, text in parts:
            info = zipfile.ZipInfo(name, date_time=ZIP_EPOCH)
            info.compress_type = ZIP_COMPRESSION
            info.create_system = ZIP_CREATE_SYSTEM
            info.external_attr = 0o600 << 16
            zf.writestr(info, text.encode("utf-8"))
    return buf.getvalue()


# ── DOCX: the #10 §9 parity fixtures (Issue #18, 2026-09-25) ─────────────────

R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
MC_NS = "http://schemas.openxmlformats.org/markup-compatibility/2006"
WPS_NS = "http://schemas.microsoft.com/office/word/2010/wordprocessingShape"
V_NS = "urn:schemas-microsoft-com:vml"
_CT = "application/vnd.openxmlformats-officedocument.wordprocessingml."
_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/"

RICH_CONTENT_TYPES = XML_DECL + (
    '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
    '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
    '<Default Extension="xml" ContentType="application/xml"/>'
    f'<Override PartName="/word/document.xml" ContentType="{_CT}document.main+xml"/>'
    f'<Override PartName="/word/styles.xml" ContentType="{_CT}styles+xml"/>'
    f'<Override PartName="/word/footnotes.xml" ContentType="{_CT}footnotes+xml"/>'
    f'<Override PartName="/word/header1.xml" ContentType="{_CT}header+xml"/>'
    f'<Override PartName="/word/footer1.xml" ContentType="{_CT}footer+xml"/>'
    '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>'
    "</Types>"
)

RICH_DOC_RELS = XML_DECL + (
    '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
    f'<Relationship Id="rId1" Type="{_REL}styles" Target="styles.xml"/>'
    f'<Relationship Id="rId2" Type="{_REL}footnotes" Target="footnotes.xml"/>'
    f'<Relationship Id="rId3" Type="{_REL}header" Target="header1.xml"/>'
    f'<Relationship Id="rId4" Type="{_REL}footer" Target="footer1.xml"/>'
    f'<Relationship Id="rId5" Type="{_REL}hyperlink" Target="https://example.org/katalog" TargetMode="External"/>'
    "</Relationships>"
)

#: Title by NAME (no outline level), a localized heading by NAME ("Nadpis 2", as a Czech
#: template defines it), and a character style that carries bold — heading and emphasis
#: detection has to follow the style chain, not just the paragraph's own properties.
RICH_STYLES = XML_DECL + (
    f'<w:styles xmlns:w="{W_NS}">'
    '<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>'
    '<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/>'
    '<w:basedOn w:val="Normal"/><w:rPr><w:sz w:val="56"/></w:rPr></w:style>'
    '<w:style w:type="paragraph" w:styleId="Nadpis2"><w:name w:val="Nadpis 2"/>'
    '<w:basedOn w:val="Normal"/><w:rPr><w:b/></w:rPr></w:style>'
    '<w:style w:type="character" w:styleId="Strong"><w:name w:val="Strong"/>'
    "<w:rPr><w:b/></w:rPr></w:style>"
    "</w:styles>"
)

RICH_FOOTNOTES = XML_DECL + (
    f'<w:footnotes xmlns:w="{W_NS}">'
    '<w:footnote w:type="separator" w:id="-1"><w:p><w:r><w:separator/></w:r></w:p></w:footnote>'
    '<w:footnote w:type="continuationSeparator" w:id="0"><w:p><w:r><w:continuationSeparator/>'
    "</w:r></w:p></w:footnote>"
    '<w:footnote w:id="1"><w:p><w:r><w:footnoteRef/></w:r>'
    '<w:r><w:t xml:space="preserve"> Katalog nálezů je uložen v archivu.</w:t></w:r></w:p></w:footnote>'
    "</w:footnotes>"
)

RICH_HEADER = XML_DECL + (
    f'<w:hdr xmlns:w="{W_NS}"><w:p><w:r><w:t>Archeologický ústav – nálezová zpráva</w:t></w:r>'
    "</w:p></w:hdr>"
)

#: "Strana " + a PAGE field. The field CODE (`instrText`) must not become text; its cached
#: RESULT ("1") is what a reader shows, so that is what the footer line carries.
RICH_FOOTER = XML_DECL + (
    f'<w:ftr xmlns:w="{W_NS}"><w:p><w:r><w:t xml:space="preserve">Strana </w:t></w:r>'
    '<w:r><w:fldChar w:fldCharType="begin"/></w:r><w:r><w:instrText xml:space="preserve"> PAGE </w:instrText></w:r>'
    '<w:r><w:fldChar w:fldCharType="separate"/></w:r><w:r><w:t>1</w:t></w:r>'
    '<w:r><w:fldChar w:fldCharType="end"/></w:r></w:p></w:ftr>'
)


def _rich_document_xml() -> str:
    """Everything the #10 §9 measurement found missing on the DOCX side, in one body.

    Pages: pageBreakBefore starts page 2; a nextPage section break starts page 3; Word's
    `lastRenderedPageBreak` inside a paragraph starts page 4 (the `auto` break mode shared
    with alto-postprocess #31), splitting that paragraph's two lines across the boundary.
    """
    textbox = (
        '<w:r><mc:AlternateContent><mc:Choice Requires="wps"><w:drawing><wps:wsp><wps:txbx>'
        "<w:txbxContent><w:p><w:r><w:t>Poznámka v rámečku.</w:t></w:r></w:p></w:txbxContent>"
        "</wps:txbx></wps:wsp></w:drawing></mc:Choice><mc:Fallback><w:pict><v:shape><v:textbox>"
        "<w:txbxContent><w:p><w:r><w:t>Poznámka v rámečku.</w:t></w:r></w:p></w:txbxContent>"
        "</v:textbox></v:shape></w:pict></mc:Fallback></mc:AlternateContent></w:r>"
    )
    merged_table = (
        "<w:tbl>"
        '<w:tblPr><w:tblW w:w="9070" w:type="dxa"/></w:tblPr>'
        '<w:tblGrid><w:gridCol w:w="4535"/><w:gridCol w:w="4535"/></w:tblGrid>'
        '<w:tr><w:tc><w:tcPr><w:tcW w:w="9070" w:type="dxa"/><w:gridSpan w:val="2"/></w:tcPr>'
        "<w:p><w:r><w:t>Nálezy</w:t></w:r></w:p></w:tc></w:tr>"
        f"<w:tr>{_cell('Keramika')}{_cell('12 ks')}</w:tr>"
        "</w:tbl>"
    )
    section_1 = (
        '<w:sectPr><w:headerReference w:type="default" r:id="rId3"/>'
        '<w:footerReference w:type="default" r:id="rId4"/>'
        '<w:type w:val="nextPage"/><w:pgSz w:w="11906" w:h="16838"/></w:sectPr>'
    )
    body = (
        _p("Hradiště u Horní Mezi", style="Title")
        + _p("Průběh výzkumu", style="Nadpis2")
        + '<w:p><w:r><w:rPr><w:rStyle w:val="Strong"/></w:rPr><w:t>Sonda II</w:t></w:r></w:p>'
        # Tracked insertion kept, tracked deletion dropped, hyperlink text kept, and a
        # footnote reference that must not turn into text of its own.
        + '<w:p><w:r><w:t xml:space="preserve">Nalezena </w:t></w:r>'
        + '<w:ins w:id="1" w:author="A" w:date="1980-01-01T00:00:00Z"><w:r><w:t>bronzová spona</w:t></w:r></w:ins>'
        + '<w:del w:id="2" w:author="A" w:date="1980-01-01T00:00:00Z"><w:r><w:delText>železný nůž</w:delText></w:r></w:del>'
        + '<w:r><w:t xml:space="preserve">, viz </w:t></w:r>'
        + '<w:hyperlink r:id="rId5"><w:r><w:t>katalog</w:t></w:r></w:hyperlink>'
        + '<w:r><w:t>.</w:t></w:r><w:r><w:footnoteReference w:id="1"/></w:r></w:p>'
        # A soft line break: two lines, ONE paragraph group.
        + "<w:p><w:r><w:t>První řádek</w:t><w:br/><w:t>Druhý řádek</w:t></w:r></w:p>"
        # An outline level set directly on the paragraph (level 2 = heading 3).
        + '<w:p><w:pPr><w:outlineLvl w:val="2"/></w:pPr><w:r><w:t>Dílčí závěr</w:t></w:r></w:p>'
        + f"<w:p><w:r><w:t>Kotva rámečku.</w:t></w:r>{textbox}</w:p>"
        + "<w:p><w:pPr><w:pageBreakBefore/></w:pPr><w:r><w:t>Druhá strana začíná zde.</w:t></w:r></w:p>"
        + merged_table
        + f"<w:p><w:pPr>{section_1}</w:pPr><w:r><w:t>Konec první sekce.</w:t></w:r></w:p>"
        + '<w:p><w:r><w:t xml:space="preserve">Text třetí strany </w:t></w:r>'
        + "<w:r><w:lastRenderedPageBreak/><w:t>pokračuje na čtvrté.</w:t></w:r></w:p>"
        + _p("Závěr.")
        + '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/></w:sectPr>'
    )
    return XML_DECL + (
        f'<w:document xmlns:w="{W_NS}" xmlns:r="{R_NS}" xmlns:mc="{MC_NS}" '
        f'xmlns:wps="{WPS_NS}" xmlns:v="{V_NS}"><w:body>{body}</w:body></w:document>'
    )


def _zip_parts(parts: List[Tuple[str, str]]) -> bytes:
    """The deterministic zip writer `minimal_docx` uses, for the fixtures below."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=ZIP_COMPRESSION) as zf:
        for name, text in parts:
            info = zipfile.ZipInfo(name, date_time=ZIP_EPOCH)
            info.compress_type = ZIP_COMPRESSION
            info.create_system = ZIP_CREATE_SYSTEM
            info.external_attr = 0o600 << 16
            zf.writestr(info, text.encode("utf-8"))
    return buf.getvalue()


def rich_docx() -> bytes:
    """G3/G5 and the layout cues on the DOCX side: title and localized heading by style
    name, a direct outline level, bold through a character style, a tracked insertion and
    deletion, a hyperlink, a footnote, a soft line break, a text box (Choice + VML
    Fallback — its text must appear once), pageBreakBefore, a merged table cell, a
    nextPage section break with a header and a PAGE-field footer, and Word's
    lastRenderedPageBreak."""
    return _zip_parts(
        [
            ("[Content_Types].xml", RICH_CONTENT_TYPES),
            ("_rels/.rels", ROOT_RELS),
            ("docProps/core.xml", CORE_PROPS),
            ("word/_rels/document.xml.rels", RICH_DOC_RELS),
            ("word/document.xml", _rich_document_xml()),
            ("word/styles.xml", RICH_STYLES),
            ("word/footnotes.xml", RICH_FOOTNOTES),
            ("word/header1.xml", RICH_HEADER),
            ("word/footer1.xml", RICH_FOOTER),
        ]
    )


def bare_docx() -> bytes:
    """G4: minimal.docx's parts with NO Override for word/document.xml.

    The main part then falls back to `<Default Extension="xml">`, i.e. application/xml —
    what alto-postprocess's `CTX000000010.docx` looks like. Word, LibreOffice and mammoth
    open it (the package relationship names the main part); python-docx refuses it with
    "not a Word file".
    """
    content_types = CONTENT_TYPES.replace(
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-'
        'officedocument.wordprocessingml.document.main+xml"/>',
        "",
    )
    assert content_types != CONTENT_TYPES
    return _zip_parts(
        [
            ("[Content_Types].xml", content_types),
            ("_rels/.rels", ROOT_RELS),
            ("docProps/core.xml", CORE_PROPS),
            ("word/_rels/document.xml.rels", DOC_RELS),
            ("word/document.xml", _document_xml()),
            ("word/styles.xml", STYLES),
        ]
    )


# ── driver ───────────────────────────────────────────────────────────────────

BUILDERS = {
    "minimal.pdf": minimal_pdf,
    "enrichable.pdf": enrichable_pdf,
    "garbled.pdf": garbled_pdf,
    "minimal.docx": minimal_docx,
    "image_only.pdf": image_only_pdf,
    "two_column.pdf": two_column_pdf,
    "ocr_layer.pdf": ocr_layer_pdf,
    "table.pdf": table_pdf,
    "rich.docx": rich_docx,
    "bare.docx": bare_docx,
}

NOTES = {
    "minimal.pdf": "2 pages, 3 text blocks, no diacritics — happy path for bbox + group_id",
    "enrichable.pdf": "1 page, diacritic-free Czech archaeology — the LLM stage's input",
    "garbled.pdf": "WinAnsi declared over cp1250, no /ToUnicode — must trip decode-sanity",
    "minimal.docx": "heading + 2 paragraphs + 2x2 table + explicit page break",
    "image_only.pdf": "text page + image-only page + blank page — G1 needs_ocr",
    "two_column.pdf": "2 pages, 2 columns, title, running header, page-number footer, roman labels",
    "ocr_layer.pdf": "page image + invisible (3 Tr) text on both pages — G7 OCR layer",
    "table.pdf": "ruled 3x2 table between two paragraphs",
    "rich.docx": "title/localized headings, ins/del, footnote, text box, header/footer, 4 pages",
    "bare.docx": "minimal.docx without the document.xml Override — G4 lenient open",
}


MANIFEST_NAME = "MANIFEST.json"

#: Fixture BYTES are not committed — only this generator and the manifest are. That is a
#: deliberate trade (no binaries in review), but it means the sha256s are inert unless
#: something regenerates and checks them; tests/test_digital_fixtures.py is what does.
DEFAULT_OUTDIR = Path(__file__).resolve().parent

#: THE canonical manifest: one fixed, committed, repo-relative path — `tests/fixtures/`,
#: alongside the other committed fixtures, NOT inside `digital/`.
#:
#: The original defect was that the generator wrote `outdir/MANIFEST.json` while the
#: committed manifest lived one directory up, so running the generator created a second
#: manifest git had never seen and left the committed one stale forever, read by nothing.
#: There are two ways to close that gap — move the committed file down, or make the
#: generator write where the committed file already is. This is the second, and it is the
#: better one for a reason beyond convenience:
#:
#: `--outdir` is a SCRATCH parameter. `--outdir /tmp/fx` means "put the bytes somewhere I can
#: poke at them", and it should never relocate the repo's canonical record of what those bytes
#: must be — otherwise `--verify --outdir /tmp/fx` silently verifies against a manifest it
#: just wrote, which is not a check at all. So the manifest path is now independent of
#: `--outdir` and overridable only by an explicit `--manifest`.
CANONICAL_MANIFEST = DEFAULT_OUTDIR.parent / MANIFEST_NAME


def build_all() -> Dict[str, bytes]:
    """Every fixture, generated in memory. The one source of truth for both modes."""
    return {name: build() for name, build in BUILDERS.items()}


def manifest_for(blobs: Dict[str, bytes]) -> Dict[str, Dict[str, object]]:
    return {
        name: {
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
            "note": NOTES[name],
        }
        for name, data in blobs.items()
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Generate the deterministic digital-born golden fixtures for Issue #18.",
        epilog=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--outdir",
        default=str(DEFAULT_OUTDIR),
        help="where to write/read the fixture BYTES. Scratch parameter — it does not move "
        "the canonical manifest (use --manifest for that).",
    )
    ap.add_argument(
        "--manifest",
        default=str(CANONICAL_MANIFEST),
        help=f"path to the committed manifest of sha256s (default: {CANONICAL_MANIFEST}).",
    )
    ap.add_argument(
        "--verify",
        action="store_true",
        help="regenerate in memory and fail on any drift from the manifest OR the on-disk bytes",
    )
    args = ap.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    blobs = build_all()
    manifest = manifest_for(blobs)
    manifest_path = Path(args.manifest)

    if args.verify:
        drifted: List[str] = []
        # 1. Against the committed manifest. This is the check --verify advertised and did
        #    not perform: it only ever compared regenerated bytes to on-disk bytes, so with
        #    the fixtures absent (they are not committed) it compared nothing at all and the
        #    pinned sha256s were never read by any code path.
        if manifest_path.exists():
            recorded = json.loads(manifest_path.read_text(encoding="utf-8"))
            for name, entry in manifest.items():
                was = recorded.get(name)
                if was is None:
                    drifted.append(f"{name}: absent from {MANIFEST_NAME}")
                elif was.get("sha256") != entry["sha256"]:
                    drifted.append(
                        f"{name}: sha256 {was.get('sha256', '?')[:16]}… recorded, "
                        f"{entry['sha256'][:16]}… generated ({was.get('bytes')} -> {entry['bytes']} bytes)"
                    )
            for name in set(recorded) - set(manifest):
                drifted.append(f"{name}: in {MANIFEST_NAME} but no builder produces it")
        else:
            drifted.append(f"{manifest_path} is missing — nothing to verify against")

        # 2. Against any fixture bytes that happen to be on disk.
        for name, data in blobs.items():
            target = outdir / name
            if target.exists() and target.read_bytes() != data:
                drifted.append(f"{name}: on-disk bytes differ from the generator")

        if drifted:
            print("FIXTURE DRIFT:", file=sys.stderr)
            for line in drifted:
                print(f"  - {line}", file=sys.stderr)
            print(
                "\nIf the change is intended, regenerate and review the golden diffs "
                "deliberately:\n  python tests/fixtures/digital/make_fixtures.py",
                file=sys.stderr,
            )
            return 1
        print(f"fixtures and {MANIFEST_NAME} match the generator")
        return 0

    for name, data in blobs.items():
        (outdir / name).write_bytes(data)
        print(f"  {name:<14} {len(data):>6} bytes  {manifest[name]['sha256'][:16]}…")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(blobs)} fixtures to {outdir}")
    print(f"wrote {MANIFEST_NAME} to {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
