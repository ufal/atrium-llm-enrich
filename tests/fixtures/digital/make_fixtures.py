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
    python tests/fixtures/digital/make_fixtures.py --verify    # fail if bytes changed
    python tests/fixtures/digital/make_fixtures.py --outdir /tmp/fx

The three fixtures and what each one is for:

    minimal.pdf   2 pages, 3 text blocks, clean ASCII-safe Latin. The happy path: exact
                  bboxes, a real page break, and blocks that must land as distinct
                  `lines[].group_id` values.
    garbled.pdf   1 page, Czech text, /WinAnsiEncoding declared over cp1250 bytes, no
                  /ToUnicode. Text extraction SUCCEEDS and returns wrong characters — the
                  case that must trip the decode-sanity check and set
                  `pages[].needs_ocr = true` rather than pass silently downstream.
    minimal.docx  Heading + 2 paragraphs + a 2x2 table + an explicit page break + a
                  paragraph after it. Structural truth with no reliable geometry, i.e.
                  §3's "do not fabricate bounding boxes" rule under test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

#: Fixed zip timestamp. Any real clock value makes the DOCX non-reproducible.
ZIP_EPOCH: Tuple[int, int, int, int, int, int] = (1980, 1, 1, 0, 0, 0)

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
        objects[contents_num] = (
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii") + stream + b"\nendstream"
        )

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
            _text_block(72, 720, [b"Excavation report, block one, line one.",
                                  b"Excavation report, block one, line two."]),
            _text_block(72, 660, [b"Block two is a separate paragraph."]),
        ]
    )
    page2 = _text_block(72, 720, [b"Page two, block three, only line."])
    return _build_pdf([page1, page2], FONT_CLEAN)


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
    "<w:rPr><w:b/><w:sz w:val=\"32\"/></w:rPr></w:style>"
    "</w:styles>"
)

#: Fixed dates. dcterms values are the DOCX equivalent of a PDF /CreationDate.
CORE_PROPS = XML_DECL + (
    '<cp:coreProperties '
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
    page_break = "<w:p><w:r><w:br w:type=\"page\"/></w:r></w:p>"
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
    """A .docx is a zip of XML parts; fixed entry order + fixed timestamps make it
    byte-reproducible, which no high-level writer will do for you."""
    import io

    parts = [
        ("[Content_Types].xml", CONTENT_TYPES),
        ("_rels/.rels", ROOT_RELS),
        ("docProps/core.xml", CORE_PROPS),
        ("word/_rels/document.xml.rels", DOC_RELS),
        ("word/document.xml", _document_xml()),
        ("word/styles.xml", STYLES),
    ]
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, text in parts:
            info = zipfile.ZipInfo(name, date_time=ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            zf.writestr(info, text.encode("utf-8"))
    return buf.getvalue()


# ── driver ───────────────────────────────────────────────────────────────────

BUILDERS = {
    "minimal.pdf": minimal_pdf,
    "garbled.pdf": garbled_pdf,
    "minimal.docx": minimal_docx,
}

NOTES = {
    "minimal.pdf": "2 pages, 3 text blocks, no diacritics — happy path for bbox + group_id",
    "garbled.pdf": "WinAnsi declared over cp1250, no /ToUnicode — must trip decode-sanity",
    "minimal.docx": "heading + 2 paragraphs + 2x2 table + explicit page break",
}


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    ap.add_argument("--outdir", default=str(Path(__file__).resolve().parent))
    ap.add_argument(
        "--verify",
        action="store_true",
        help="regenerate in memory and fail if the on-disk bytes differ",
    )
    args = ap.parse_args(argv)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Dict[str, object]] = {}
    drifted: List[str] = []

    for name, build in BUILDERS.items():
        data = build()
        digest = hashlib.sha256(data).hexdigest()
        manifest[name] = {"sha256": digest, "bytes": len(data), "note": NOTES[name]}
        target = outdir / name
        if args.verify:
            if not target.exists() or target.read_bytes() != data:
                drifted.append(name)
            continue
        target.write_bytes(data)
        print(f"  {name:<14} {len(data):>6} bytes  {digest[:16]}…")

    if args.verify:
        if drifted:
            print(f"FIXTURE DRIFT: {', '.join(drifted)}", file=sys.stderr)
            print("Regenerate and review the golden diffs deliberately.", file=sys.stderr)
            return 1
        print("fixtures match the generator")
        return 0

    (outdir / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {len(BUILDERS)} fixtures + MANIFEST.json to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
