#!/usr/bin/env python3
"""
api_util/digital_to_json.py — digital-born PDF/DOCX → `atrium_document` JSON.

The `digital-convert` originator (hub issue #18 §1a, plan §2). `BLOCK_OWNERS` gives the
positional plane — `pages`, `content`, `lines`, `tables` — two possible originators, and
which one applies is fixed per document by `source.origin`:

    ocr:* / vlm:* / ABBYY-ALTO   → alto-postprocess
    digital-born* / pdf / docx   → digital-convert   ← this module

The two are mutually exclusive per document, so nothing here is a second opinion on an
ALTO document; it is the *only* writer of the plane for a born-digital one.

Four layers, per `agent_dev_logs/plans/18.plan.md` §2:

  A. **Extraction** — into the internal `DigitalDocument` (`api_util/digital_ir.py`).
     Two engines, chosen with `--engine`:

       * `light` (default): `api_util/digital_pdf.py` (pdfplumber + pypdfium2) for PDF,
         `api_util/digital_docx.py` (python-docx + lxml) for DOCX. MIT/BSD/Apache, no
         models, no network — `requirements_digital.txt`.
       * `docling` (opt-in, PDF only): `api_util/digital_docling.py` — Docling's layout and
         table models decide structure and reading order, the light engine's pdfplumber
         lines keep line-level geometry. For complex layouts, when accuracy is worth a
         torch stack — `requirements_digital_docling.txt`.

     Imported lazily, so the base install and the fast lane never pay for either.
  B. **Normalization** — paragraph grouping from geometry, and the decode-sanity gate
     that turns "born-digital" into "born-digital and actually readable".
  C. **Serialization** — into `DocumentRecord` as `program="digital-convert"`, writing
     only this program's declared field grants, with the licence union of the components
     this run actually used (accretion rule 5).
  D. **Validation & export** — the §1b round-trip assertion *and*
     `validate_document()`. If either fails, **no `doc.json` is emitted**.

## Born-digital ≠ trustworthy text

The reason layer B exists at all. Issue #10's research pass found PDFs whose text layer
extracts *successfully* and *plausibly*, and is wrong: non-embedded WinAnsi Helvetica with
no `/ToUnicode` map, so CP1250 (Central European) bytes are read as CP1252 (Western).

    intended : Zpráva o sondě číslo 3. Nalezeny hřeby, vrstva ornice měla
    extracted: Zpráva o sondì èíslo 3. Nalezeny høeby, vrstva ornice mìla

No exception, no encoding error, no length change — and `Zpráva` survives intact, because
`á` sits at the same codepoint in both encodings. *Partially* correct output is what makes
this dangerous: a downstream reader has no signal that anything went wrong.

`CP1250_MISREADS` is derived from the two codecs at import time rather than typed out by
hand, so it is exactly the set of byte values where the two disagree *and* the CP1250
reading is a Czech letter — provably complete, and self-documenting about why `á`/`é` are
absent from it.

Detection feeds two outputs, and deliberately not a third:

  * `lines[].categ` = **`"Garbage"`** — the exact spelling `api_util/json_to_md.py`'s
    `DROP_CATEGORIES` filters on. `categ` is an open string in the schema, so a synonym
    would silently disable the filter instead of failing validation.
  * `pages[].needs_ocr` + `pages[].needs_ocr_reason` — the documented digital→OCR
    hand-off. Setting `needs_ocr` is what *authorises* `alto-postprocess` to re-originate
    this record's plane, even though `source.origin` stays `digital-born-*` (which remains
    truthful: it describes how the ORIGINAL was acquired). `pages[].ocr` is never granted
    to this program, so "was this OCR'd" stays answerable from the record.
  * **Not** a silent repair. `decode_sanity()` returns the recovered string because it is
    the single most useful thing a human triaging the document can see, but this converter
    REPORTS; routing policy lives outside it. Rewriting text under the reader's feet would
    make `source.sha256` describe a document whose text no longer matches it.

The same hand-off carries three more page verdicts (Issue #18, 2026-09-25 — the #10 §9 gap
register): a page with no text layer but an image or vector paths on it (a scanned plate,
text drawn as curves), a text layer of U+FFFD/control characters (a subset font without
/ToUnicode), and a text layer that is a prior OCR run's invisible text over a page image.
When OCR-layer pages are at least half of the text-bearing pages, the document is not
born-digital at all and is refused (exit 3): its originator is alto-postprocess, as
`ocr:pdf-text-layer` — the rule atrium-alto-postprocess `default_source_origin` applies.

## Layout cues (the #18 mapping table)

  * pages — PDF pages (with their `/PageLabels`), DOCX pages from explicit breaks, section
    breaks and Word's `lastRenderedPageBreak` (the rules alto-postprocess #31 uses);
  * `lines[].group_id` — PDF text blocks (per column), DOCX paragraphs, table cells;
  * `lines[].style` — `bold`, `italic`, `heading_level` (PDF: by size; DOCX: outline level
    or style name) and `region` ∈ {page_header, page_footer, footnote};
  * `tables[]` — DOCX tables and ruled PDF tables (Docling: any table), shape only;
  * `lines[].bbox` + `pages[].canvas` — PDF only.

## Coordinates

`$defs/bbox` is normative: **origin top-left, y increasing downwards**, in the unit named by
that page's `canvas.unit`. For pdfplumber that means `top`/`bottom` — never `y0`/`y1`, which
are PDF user space (origin bottom-left) and are the specific mistake the schema description
calls out. DOCX has no page geometry without rendering, so a DOCX document carries no
`bbox` and therefore no `canvas` either.

## Exit codes

    0  record written
    2  a dependency is missing (advice printed, no traceback)
    3  not an input this converter takes: unsupported format, legacy .doc, an OCR-layer PDF
    4  the file is broken: corrupt, encrypted, over the ZIP limits

Usage:
    python api_util/digital_to_json.py report.pdf --document-json-out report.document.json
    python api_util/digital_to_json.py report.docx --document-json prev.json --out out.json
    python api_util/digital_to_json.py report.pdf --engine docling --paradata-dir paradata/
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import unicodedata
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

# The vendored hub canonical modules live at the repo root; this file is in api_util/.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from api_util.digital_ir import (  # noqa: E402  (re-exported: `d2j.DigitalLine` etc.)
    REGION_FOOTER,
    REGION_FOOTNOTE,
    REGION_HEADER,
    REGIONS,
    TEXT_LAYER_BLANK,
    TEXT_LAYER_DIGITAL,
    TEXT_LAYER_NONE,
    TEXT_LAYER_OCR,
    DigitalDocument,
    DigitalInputError,
    DigitalLine,
    DigitalPage,
    DigitalTable,
    font_flags,
    is_inverted,
    missing_dependency,
    sha256_file,
)
from atrium_document import (  # noqa: E402
    DocumentRecord,
    canonical_doc_id,
    validate_document,
)

__all__ = [
    "REGION_FOOTER",
    "REGION_FOOTNOTE",
    "REGION_HEADER",
    "REGIONS",
    "TEXT_LAYER_BLANK",
    "TEXT_LAYER_DIGITAL",
    "TEXT_LAYER_NONE",
    "TEXT_LAYER_OCR",
    "DigitalDocument",
    "DigitalInputError",
    "DigitalLine",
    "DigitalPage",
    "DigitalTable",
]

#: Kept for callers of the pre-split names.
_missing = missing_dependency
_sha256 = sha256_file
_font_flags = font_flags
_is_inverted = is_inverted

PROGRAM = "digital-convert"

#: `source.origin` values this converter writes. Both match an `ORIGIN_ORIGINATORS` prefix,
#: which is what routes the positional plane to this program. An origin the table has not
#: been taught makes `_assert_origin_consistent()` ABSTAIN rather than refuse — silently
#: switching §1a off for that document — so these strings are not cosmetic.
ORIGIN_PDF = "digital-born-pdf"
ORIGIN_DOCX = "digital-born-docx"

#: `--engine` values. `light` is the default and the only one the fast lane exercises.
ENGINES: Tuple[str, ...] = ("light", "docling")

#: `--docx-page-breaks` values — alto-postprocess `[TEXT_INGEST].PAGE_BREAKS`, same meaning.
DOCX_PAGE_BREAK_MODES: Tuple[str, ...] = ("auto", "explicit", "none")

# ---------------------------------------------------------------------------
# Line-category labels  (hub registry: atrium_vocab.LINE_CATEGORY_ORIGINATORS)
# ---------------------------------------------------------------------------
# The two load-bearing `categ` spellings this converter writes, and the only ones it
# ever writes. `lines[].categ`'s other authorised originator, alto-postprocess, emits a
# DISJOINT five-value set ({Clear, Empty, Noisy, Non-text, Trash}) — an OCR verdict over
# a rendered image, where these two are a decode-sanity verdict over an embedded text
# layer. The disjointness is deliberate and is not drift to reconcile; a consumer that
# filters this field must handle BOTH sets (see defect V-1 in the hub's
# docs/skos_strategy.md §6, and the note over `json_to_md.DROP_CATEGORIES`).
#
# The strings are TAKEN FROM the hub registry rather than typed here, so the declaration
# and the emission cannot drift apart unnoticed — but never at the cost of changing what
# this converter emits. `atrium_vocab.py` is a vendored copy and is legitimately absent
# in some execution contexts (a bare api_util/ next to a notebook, an image built before
# the vendor step), and a stale or re-ordered copy must not silently re-spell the
# contract. So the literals below are the floor: the registry is used only when it
# agrees with them, and a disagreement is REPORTED and then ignored. That follows the
# house idiom (atrium_document.py's origin check, alto-postprocess's text_util.py):
# abstain with a NOTE on stderr, never fatal. Silence means agreement.
_CATEG_FALLBACK: Tuple[str, ...] = ("Garbage", "Inverted")

try:
    from atrium_vocab import LINE_CATEGORY_ORIGINATORS as _VOCAB_LINE_CATEGORY_ORIGINATORS
except ImportError:  # registry not vendored here — abstain, do not guess
    _declared: Tuple[str, ...] = ()
else:
    _declared = tuple(sorted(_VOCAB_LINE_CATEGORY_ORIGINATORS.get(PROGRAM, ())))

if _declared and _declared != _CATEG_FALLBACK:
    print(
        f"[digital_to_json] NOTE - line-category drift: this module emits "
        f"{list(_CATEG_FALLBACK)} but atrium_vocab declares {list(_declared)} for "
        f"originator {PROGRAM!r}. Emission is unchanged; reconcile the registry or this "
        f"file (see defect V-1 in the hub's docs/skos_strategy.md).",
        file=sys.stderr,
    )

#: Sorted, so the comparison above and the unpacking here are order-independent — the
#: registry sorts its own union the same way (`atrium_vocab.LINE_CATEGORIES`).
CATEGORIES_EMITTED: Tuple[str, ...] = _declared if _declared == _CATEG_FALLBACK else _CATEG_FALLBACK

#: The two load-bearing `categ` spellings (`json_to_md.DROP_CATEGORIES`).
CATEG_GARBAGE, CATEG_INVERTED = CATEGORIES_EMITTED

#: Below this decode-sanity ratio a line is `Garbage` — the density signal, for a line
#: whose diacritics are mostly wrong.
QUALITY_GARBAGE_BELOW = 0.90

#: ...and this many confusable characters condemn a line regardless of its length.
#:
#: A ratio ALONE provably cannot catch the real corruption. The observed line
#: "Zpráva o sondì èíslo 3. Nalezeny høeby, vrstva ornice mìla" carries 4 misreads across
#: 46 letters, which scores 0.913 — comfortably above any ratio cut loose enough not to
#: condemn clean text. Czech simply does not contain `ì`, `è`, `ø` or `ù`, so their presence
#: is not a density question: ONE may be a foreign word in a quotation, but TWO in a single
#: line is a systematic decode fault, which is exactly the shape of this failure.
#:
#: The two rules are kept separate rather than folded into one score because they answer
#: different questions — "how corrupt is this line?" (which `quality_score` must keep
#: reporting as a 0–1 axis) and "is this line trustworthy at all?".
GARBAGE_MIN_HITS = 2

#: Share of a line's visible characters that are U+FFFD or control / format / private-use /
#: unassigned code points above which the text layer does not decode at all — a subset font
#: without /ToUnicode. The threshold legacy `pdf_to_md` and alto-postprocess #31 use; the
#: JSON route had no such check, so switching the auto-convert to it would have regressed.
GARBLE_CHAR_SHARE = 0.15

#: The mojibake verdict needs positive evidence the line is Czech read through the wrong
#: code page, not merely one of the four misread letters — which are ordinary French and
#: Italian letters (`è`, `ù`, `ì`, `ò`). Ported from atrium-alto-postprocess
#: `text_formats.mojibake_line` (#31, Phase 4) so both tools judge a line alike: a letter
#: CP1250 and CP1252 share that Czech uses (so it survives the misread), and no letter only
#: French uses. "très complète" was Garbage here before.
MOJIBAKE_SHARED_LETTERS = frozenset("áíúýšžÁÍÚÝŠŽ")
MOJIBAKE_FOREIGN_LETTERS = frozenset("àêûœÀÊÛŒ")

#: Page-level band cutoffs, matching the vocabulary alto-postprocess already uses so a
#: consumer does not have to know which originator wrote the page.
BAND_CLEAR_AT = 0.90
BAND_NOISY_AT = 0.50

#: Paragraph splitting. Measured on tests/fixtures/digital/minimal.pdf: 2.0 pt between
#: lines of one paragraph, 34.0 pt between paragraphs (digital_born/README.md). A ratio
#: is used rather than an absolute, because the gap scales with the font.
PARAGRAPH_GAP_RATIO = 1.8

#: The `ocr` text-layer verdict needs at least half of the text-bearing pages before the
#: whole DOCUMENT is refused — alto-postprocess `default_source_origin`'s rule.
OCR_LAYER_DOCUMENT_SHARE = 0.5


def _czech_letters() -> frozenset:
    return frozenset("áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ")


def _build_cp1250_misreads() -> Dict[str, str]:
    """Characters a CP1250 byte turns into when read as CP1252, where CP1250 meant Czech.

    Derived from the codecs, not hand-typed: for every high byte, decode it both ways and
    keep the pair only when the readings differ AND the CP1250 reading is a Czech letter.
    That makes the table provably complete for this failure mode, and explains its own
    omissions — `á` (0xE1) and `é` (0xE9) decode identically under both codecs, which is
    exactly why "Zpráva" survives a mis-decode while "sondě" does not.
    """
    czech = _czech_letters()
    table: Dict[str, str] = {}
    for byte in range(0x80, 0x100):
        try:
            western = bytes([byte]).decode("cp1252")
            eastern = bytes([byte]).decode("cp1250")
        except UnicodeDecodeError:  # undefined in one of the codecs
            continue
        if western != eastern and eastern in czech:
            table[western] = eastern
    return table


CP1250_MISREADS: Dict[str, str] = _build_cp1250_misreads()


# ──────────────────────────────────────────────────────────────────────────────
# Layer B — normalization and the decode-sanity gate  (pure; no optional deps)
# ──────────────────────────────────────────────────────────────────────────────


@dataclass
class DecodeReport:
    """What `decode_sanity()` concluded about one string."""

    score: float
    suspicious: int
    letters: int
    recovered: Optional[str] = None
    #: U+FFFD / control / format / private-use / unassigned characters among `chars`.
    bad: int = 0
    #: Visible (non-whitespace) characters.
    chars: int = 0
    #: The whole string is representable in CP1252 — a CP1250 byte string misread as CP1252
    #: always is; a string that also holds a correct `ě` or `ř` is not that failure.
    encodable: bool = True
    #: A letter only French uses (`à`, `ê`, `û`, `œ`) — evidence against the Czech reading.
    foreign: bool = False
    #: A letter Czech uses that survives the misread (`á`, `í`, `ú`, `ý`, `š`, `ž`).
    shared: bool = False

    @property
    def is_mojibake(self) -> bool:
        """CP1250 read as CP1252, on positive evidence (see `MOJIBAKE_SHARED_LETTERS`)."""
        systematic = self.suspicious >= GARBAGE_MIN_HITS or self.score < QUALITY_GARBAGE_BELOW
        return systematic and self.encodable and self.shared and not self.foreign

    @property
    def is_undecodable(self) -> bool:
        """Mostly U+FFFD or control characters: the text layer does not decode at all."""
        return bool(self.chars) and self.bad / self.chars > GARBLE_CHAR_SHARE

    @property
    def is_garbage(self) -> bool:
        """Either verdict condemns the line — see `GARBAGE_MIN_HITS` for why the mojibake
        rule needs a count as well as a ratio."""
        return self.is_mojibake or self.is_undecodable


def decode_sanity(text: str) -> DecodeReport:
    """Score 0..1 for "this text decoded correctly", plus the recovered reading.

    The score is the share of letters that are NOT a CP1250→CP1252 misread. It is
    deliberately a ratio, not a count: a caption of three words and a page of three hundred
    must be comparable, and `pages[].quality_score` is documented as a 0–1 axis either way.

    `recovered` is offered only when the whole string round-trips — `encode("cp1252")` then
    `decode("cp1250")`. A string containing a character CP1252 cannot represent was never a
    CP1250 byte sequence read this way, so a partial guess there would be fabrication.
    """
    visible = [c for c in text if not c.isspace()]
    bad = sum(1 for c in visible if c == "�" or unicodedata.category(c) in ("Cc", "Cf", "Co", "Cn"))
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return DecodeReport(score=1.0, suspicious=0, letters=0, bad=bad, chars=len(visible))

    suspicious = sum(1 for c in letters if c in CP1250_MISREADS)
    score = 1.0 - (suspicious / len(letters))

    try:
        encoded = text.encode("cp1252")
        encodable = True
    except UnicodeEncodeError:
        encoded, encodable = b"", False

    recovered: Optional[str] = None
    if suspicious and encodable:
        try:
            candidate = encoded.decode("cp1250")
        except UnicodeDecodeError:
            candidate = None
        if candidate and candidate != text:
            recovered = candidate

    present = set(letters)
    return DecodeReport(
        score=score,
        suspicious=suspicious,
        letters=len(letters),
        recovered=recovered,
        bad=bad,
        chars=len(visible),
        encodable=encodable,
        foreign=bool(present & MOJIBAKE_FOREIGN_LETTERS),
        shared=bool(present & MOJIBAKE_SHARED_LETTERS),
    )


def classify_line(line: DigitalLine, report: DecodeReport) -> Optional[str]:
    """The `categ` this line carries, or None to leave the field absent.

    Only the two spellings `json_to_md.DROP_CATEGORIES` knows are ever produced. `Inverted`
    wins over `Garbage`: mirrored text is a rendering-level fault, and its extracted string
    is not evidence of anything, so reporting a decode verdict on it would be noise.
    """
    if line.inverted:
        return CATEG_INVERTED
    if report.is_garbage:
        return CATEG_GARBAGE
    return None


def assign_group_ids(lines: Sequence[DigitalLine], prefix: str = "p") -> None:
    """Group consecutive lines into paragraphs, in place, from vertical gaps.

    `lines[].group_id` is what `json_to_md` turns back into a blank line, so this is the
    only thing preserving paragraph structure through the JSON. Only lines with geometry and
    no group yet are grouped: DOCX paragraphs and table cells arrive grouped from Layer A,
    and inventing groups from nothing would be worse than the absence the schema allows.

    The threshold is relative (`PARAGRAPH_GAP_RATIO` × a low percentile of the gaps) so it
    holds for a 10 pt body and a 24 pt heading alike; an absolute cut tuned on one fixture
    would not. A change of column, of region (header, body, footer) or of heading level,
    and any upward jump, also start a new group whatever the gap says.

    `prefix` makes the ids unique per document (`p{page_index}-{n}`): a consumer may group
    lines by `group_id` across pages, and "lines sharing a group_id are contiguous".
    """
    positioned = [ln for ln in lines if ln.bbox and ln.group_id is None]
    if not positioned:
        return
    if len(positioned) == 1:
        positioned[0].group_id = f"{prefix}0"
        return

    gaps: List[float] = []
    for previous, current in zip(positioned, positioned[1:], strict=False):
        gaps.append(max(0.0, current.bbox[1] - previous.bbox[3]))

    # The baseline is a LOW PERCENTILE of the positive gaps, not their median: within-
    # paragraph leading is the common small value, and paragraph breaks are the rare large
    # one, so the median is dragged upward by them exactly when there are few lines. On the
    # measured fixture (gaps 2.0 and 34.0) the median is 34.0 and nothing ever splits — the
    # threshold would sit above the very break it exists to find. The 25th percentile also
    # resists a single freak sub-point gap, which a bare `min()` would not.
    ordered = sorted(g for g in gaps if g > 0)
    baseline = ordered[max(0, len(ordered) // 4)] if ordered else 0.0
    threshold = baseline * PARAGRAPH_GAP_RATIO if baseline else None

    group_index = 0
    positioned[0].group_id = f"{prefix}{group_index}"
    for gap, previous, current in zip(gaps, positioned[:-1], positioned[1:], strict=True):
        boundary = (
            (threshold is not None and gap > threshold)
            or current.column != previous.column
            or current.region != previous.region
            or current.heading_level != previous.heading_level
            or current.bbox[1] < previous.bbox[1] - 1.0
        )
        if boundary:
            group_index += 1
        current.group_id = f"{prefix}{group_index}"


def _reason_no_text(page: DigitalPage) -> str:
    drawn = []
    if page.images:
        drawn.append(f"{page.images} image(s)")
    if page.vector_paths:
        drawn.append(f"{page.vector_paths} vector path(s)")
    if drawn:
        return (
            f"no extractable text layer: the page draws {' and '.join(drawn)} but carries no "
            f"text — a scanned page or text drawn as curves"
        )
    return (
        "no extractable text layer: the page draws nothing a parser can see — blank, or its "
        "content is out of the parser's reach; re-acquire it to be sure"
    )


def assess_page(page: DigitalPage) -> None:
    """Fill in `quality_score`, `quality_band`, `needs_ocr` and its reason, in place.

    `needs_ocr_reason` is written whenever `needs_ocr` is, and never alone. The flag means
    different things on the two originator paths — "no text layer" for a scan, "a text
    layer that lies" here — and `json_to_md` emits the reason as a cue the model reads.
    Without it every digital-born page rendered "no extractable text layer", which is false
    for a page that has one.

    A page with no lines has no score: there is nothing the score could be about, and
    writing Clear/1.0 for it is how an image-only page vanished (G1). A PDF page without text
    is flagged (Layer A's `text_layer == "none"`); an empty DOCX page (`blank`, two breaks in
    a row) is not — there is no page image for an OCR engine to read.
    """
    reasons: List[str] = []
    if not page.lines:
        page.quality_score = None
        page.quality_band = None
        if page.text_layer == TEXT_LAYER_NONE:
            reasons.append(_reason_no_text(page))
    else:
        scored = [ln.quality_score for ln in page.lines if ln.quality_score is not None]
        page.quality_score = round(sum(scored) / len(scored), 4) if scored else 1.0
        if page.quality_score >= BAND_CLEAR_AT:
            page.quality_band = "Clear"
        elif page.quality_score >= BAND_NOISY_AT:
            page.quality_band = "Noisy"
        else:
            page.quality_band = "Trash"

        garbled = [ln for ln in page.lines if ln.categ == CATEG_GARBAGE]
        undecodable = [ln for ln in garbled if decode_sanity(ln.text).is_undecodable]
        mojibake = [ln for ln in garbled if ln not in undecodable]
        if mojibake:
            reasons.append(
                f"embedded text layer does not decode: {len(mojibake)} of {len(page.lines)} "
                f"lines carry CP1250 bytes read as CP1252 (mojibake diacritics), page "
                f"decode-sanity {page.quality_score:.2f}. The page has a text layer; it is "
                f"not trustworthy."
            )
        if undecodable:
            reasons.append(
                f"garbled text layer: {len(undecodable)} of {len(page.lines)} lines are mostly "
                f"U+FFFD or control characters (a subset font without /ToUnicode?). The page "
                f"has a text layer; it does not decode."
            )
        if page.text_layer == TEXT_LAYER_OCR:
            reasons.append(
                f"the text layer is a prior OCR run: {page.invisible_text_objects} of "
                f"{page.text_objects} text objects are invisible (render mode 3) over a page "
                f"image. Re-acquire the page through the OCR originator (alto-postprocess, "
                f"source.origin ocr:pdf-text-layer)."
            )

    page.needs_ocr = bool(reasons)
    page.needs_ocr_reason = " ".join(reasons)


def normalize(doc: DigitalDocument) -> DigitalDocument:
    """Run Layer B over an extracted document, in place, and return it.

    Two passes per page. The first judges every line on its own evidence. The second
    applies the page's verdict (G6): once a page is condemned for mojibake, a line with a
    single misread letter is not "one foreign word" any more — it is the same corruption
    at a lower density, and letting it through leaked 2 of `garbled.pdf`'s 3 bad lines.
    Only the shared-letter requirement is relaxed there; a French letter or a character
    CP1252 cannot hold still clears the line.
    """
    for page in doc.pages:
        reports: Dict[int, DecodeReport] = {}
        for line in page.lines:
            report = decode_sanity(line.text)
            reports[id(line)] = report
            quality = report.score
            if report.chars:
                quality = min(quality, 1.0 - report.bad / report.chars)
            line.quality_score = round(quality, 4)
            categ = classify_line(line, report)
            if categ:
                line.categ = categ
        condemned = any(
            line.categ == CATEG_GARBAGE and reports[id(line)].is_mojibake for line in page.lines
        )
        if condemned:
            for line in page.lines:
                report = reports[id(line)]
                if (
                    line.categ is None
                    and report.suspicious >= 1
                    and report.encodable
                    and not report.foreign
                ):
                    line.categ = CATEG_GARBAGE
        assign_group_ids(page.lines, prefix=f"p{page.page_index}-")
        assess_page(page)
    return doc


# ──────────────────────────────────────────────────────────────────────────────
# Layer A — extraction (lazy optional imports)
# ──────────────────────────────────────────────────────────────────────────────

_OLE2_MAGIC = b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1"
_OLE2_ENCRYPTED_STREAM = "EncryptedPackage".encode("utf-16-le")


def sniff(path: str) -> str:
    """`"pdf"` or `"docx"`, decided by content — the extension only names the file.

    A PDF header may sit anywhere in the first KB; a Word package is a ZIP whose package
    relationship or members name WordprocessingML (so `.docm`, `.dotx`, `.dotm` and a
    mis-named `.zip` all qualify, and an `.xlsx` renamed `.docx` does not). OLE2 is either
    a legacy binary `.doc` or an encrypted OOXML package; neither is readable here.
    """
    try:
        with open(path, "rb") as handle:
            head = handle.read(1024)
    except OSError as exc:
        raise DigitalInputError("unsupported", f"unsupported input {path!r}: {exc}") from exc
    if b"%PDF-" in head:
        return "pdf"
    if head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(path) as zf:
                names = set(zf.namelist())
        except zipfile.BadZipFile as exc:
            raise DigitalInputError("corrupt", f"not a readable ZIP package ({exc})") from exc
        if "word/document.xml" in names or (
            "_rels/.rels" in names and any(n.startswith("word/") for n in names)
        ):
            return "docx"
        raise DigitalInputError(
            "unsupported",
            f"unsupported input {path!r}: a ZIP package but not a Word document "
            f"(expected .pdf or .docx)",
        )
    if head.startswith(_OLE2_MAGIC):
        with open(path, "rb") as handle:
            blob = handle.read(64 * 1024 * 1024)
        if _OLE2_ENCRYPTED_STREAM in blob:
            raise DigitalInputError("encrypted", "password-protected Office document")
        raise DigitalInputError(
            "legacy_office_unsupported",
            f"unsupported input {path!r}: a legacy binary Word document (.doc); save it as "
            f".docx first",
        )
    raise DigitalInputError("unsupported", f"unsupported input {path!r}: expected .pdf or .docx")


def extract_pdf(path: str, doc_id: Optional[str] = None) -> DigitalDocument:
    """Layer A for PDF, light engine — `api_util/digital_pdf.py`."""
    from api_util.digital_pdf import extract_pdf as _extract  # noqa: PLC0415

    return _extract(path, doc_id or canonical_doc_id(path), ORIGIN_PDF)


def extract_docx(
    path: str, doc_id: Optional[str] = None, page_breaks: str = "auto"
) -> DigitalDocument:
    """Layer A for DOCX — `api_util/digital_docx.py`."""
    from api_util.digital_docx import extract_docx as _extract  # noqa: PLC0415

    return _extract(path, doc_id or canonical_doc_id(path), ORIGIN_DOCX, page_breaks=page_breaks)


def _refuse_ocr_layer(document: DigitalDocument) -> None:
    """An OCR-layer PDF is not born-digital: refuse it rather than stamp it so (G7)."""
    with_text = [p for p in document.pages if p.text_layer in (TEXT_LAYER_DIGITAL, TEXT_LAYER_OCR)]
    ocr = [p for p in with_text if p.text_layer == TEXT_LAYER_OCR]
    if ocr and len(ocr) >= OCR_LAYER_DOCUMENT_SHARE * len(with_text):
        raise DigitalInputError(
            "ocr_text_layer",
            f"{len(ocr)} of {len(with_text)} text-bearing pages are a prior OCR run's invisible "
            f"text over a page image, so this PDF is not born-digital. Its originator is "
            f"alto-postprocess (`--method text-lines`, source.origin ocr:pdf-text-layer).",
        )


def extract(
    path: str,
    doc_id: Optional[str] = None,
    engine: str = "light",
    docx_page_breaks: str = "auto",
) -> DigitalDocument:
    """Dispatch on the content. Unknown formats fail loudly rather than guessing."""
    if engine not in ENGINES:
        raise ValueError(f"engine must be one of {ENGINES}, got {engine!r}")
    kind = sniff(path)
    if kind == "pdf":
        document = extract_pdf(path, doc_id=doc_id)
        if engine == "docling":
            try:
                from api_util.digital_docling import refine_with_docling  # noqa: PLC0415
            except ImportError as exc:  # pragma: no cover - the module itself has no hard deps
                raise missing_dependency(
                    "docling", "docling", "requirements_digital_docling.txt"
                ) from exc
            document = refine_with_docling(path, document)
        _refuse_ocr_layer(document)
        return document
    if engine == "docling":
        print(
            "[digital-convert] NOTE - --engine docling applies to PDF only: Docling's DOCX "
            "backend reads the same OOXML with no page model. Using the light DOCX reader.",
            file=sys.stderr,
        )
    return extract_docx(path, doc_id=doc_id, page_breaks=docx_page_breaks)


# ──────────────────────────────────────────────────────────────────────────────
# Layer C — serialization into DocumentRecord
# ──────────────────────────────────────────────────────────────────────────────


def _page_rows(doc: DigitalDocument) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for page in doc.pages:
        row: Dict[str, Any] = {"page": page.page, "page_index": page.page_index}
        # Absent, never null: the schema types both as number/enum, and a line-less page
        # has nothing to score.
        if page.quality_score is not None:
            row["quality_score"] = page.quality_score
        if page.quality_band is not None:
            row["quality_band"] = page.quality_band
        # `canvas.unit` is mandatory whenever any bbox exists on the page, and meaningless
        # when none does (DOCX) — so the whole block is conditional on the adapter knowing a
        # page size. A PDF page with no lines (a scan) still has one, and says so.
        if page.width and page.height:
            row["canvas"] = {"width": page.width, "height": page.height, "unit": page.unit}
        if page.needs_ocr:
            row["needs_ocr"] = True
            row["needs_ocr_reason"] = page.needs_ocr_reason
        rows.append(row)
    return rows


def _line_rows(doc: DigitalDocument) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for line in doc.all_lines():
        row: Dict[str, Any] = {"page": line.page, "line": line.line, "text": line.text}
        if line.bbox:
            row["bbox"] = line.bbox
        if line.group_id is not None:
            row["group_id"] = line.group_id
        if line.categ:
            row["categ"] = line.categ
        if line.quality_score is not None:
            row["quality_score"] = line.quality_score
        if line.lang:
            row["lang"] = line.lang
        # Semantic style only — bold/italic/heading_level, plus `region` for page furniture
        # and footnotes (see `api_util/digital_ir.py` for why it lives here). Typeface and
        # point size are deliberately dropped: a reader can act on "this was a heading" and
        # cannot act on "this was Helvetica 12pt".
        style: Dict[str, Any] = {}
        if line.bold:
            style["bold"] = True
        if line.italic:
            style["italic"] = True
        if line.heading_level:
            style["heading_level"] = line.heading_level
        if line.region:
            style["region"] = line.region
        if style:
            row["style"] = style
        rows.append(row)
    return rows


def _table_rows(doc: DigitalDocument) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for page in doc.pages:
        for table in page.tables:
            rows.append(
                {
                    "table_id": table.table_id,
                    "page": table.page,
                    "caption": table.caption,
                    "n_rows": table.n_rows,
                    "n_cols": table.n_cols,
                    "group_id": table.group_id,
                    "cells": table.cells,
                }
            )
    return rows


def _content_text(doc: DigitalDocument) -> str:
    """Reading-order text for search: no `Garbage`, no running headers or footers.

    Furniture repeats on every page and says nothing about the document's content; a
    footnote does, so it stays.
    """
    return "\n".join(
        line.text
        for line in doc.all_lines()
        if line.categ != CATEG_GARBAGE and line.region not in (REGION_HEADER, REGION_FOOTER)
    )


def to_record(
    doc: DigitalDocument,
    baseline: Optional[str] = None,
    run_id: Optional[str] = None,
    paradata_ref: str = "",
    out_dir: str = ".",
    strict: bool = False,
    license_detail: Optional[Dict[str, Any]] = None,
) -> Tuple[DocumentRecord, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Build the record. Returns it with the page and line rows, for Layer D to re-check.

    `DocumentRecord.open()` handles the baseline for us: a missing or absent path is not an
    error (rule 3, standalone-safe), and when one IS given every other tool's blocks are
    deep-copied through untouched (rule 2). Note that `open()` also decides the doc_id — a
    baseline's own id wins over the one derived here, because our id came from a LOCAL
    FILENAME and the record already knows what document it is.

    `set_source()` is called FIRST. The module re-checks blocks written before an origin
    arrives, so this is no longer strictly required — but `source.origin` is what authorises
    this program to write the plane at all, and writing it first keeps the authorisation
    visible in the code rather than dependent on a deferred re-check. It is also
    first-writer-wins, so on a re-run the original acquisition facts are preserved and only
    a genuine contradiction is reported.

    `license_detail` is this run's component licence union (rule 5). Without it the record
    reports the conservative CC BY-NC 4.0 default, which is what every digital-born record
    said until 2026-09-25: nothing here ever contributed the MIT/Apache stack it runs on.
    """
    record = DocumentRecord.open(
        doc.doc_id,
        PROGRAM,
        baseline=baseline,
        run_id=run_id,
        paradata_ref=paradata_ref,
        out_dir=out_dir,
        strict=strict,
    )
    record.set_source(
        sha256=doc.sha256,
        filename=doc.filename,
        media_type=doc.media_type,
        origin=doc.origin,
        page_count=len(doc.pages),
    )
    if license_detail:
        record.add_license_detail(license_detail)

    page_rows = _page_rows(doc)
    line_rows = _line_rows(doc)

    # pages[] and lines[] are FIELD-SPLIT (page-classification contributes category;
    # nlp-enrich contributes morphology), so merge_block — set_block would erase a
    # co-contributor's fields on a re-run over an existing record.
    record.merge_block("pages", page_rows)
    record.merge_block("lines", line_rows)

    # `content` has a single owner per document and `tables` is declared for the two
    # ORIGINATORS only — which are mutually exclusive per record — so neither has a
    # co-contributor to erase and set_block is both correct and quiet for them.
    body = _content_text(doc)
    record.set_block("content", {"text": body or None, "reading_order": doc.reading_order})

    tables = _table_rows(doc)
    if tables:
        record.set_block("tables", tables)

    return record, page_rows, line_rows


# ── provenance: the licence of what actually ran ─────────────────────────────


def _paradata_logger(paradata_dir: str, config: Dict[str, Any]) -> Optional[Any]:
    try:
        from atrium_paradata import ParadataLogger  # noqa: PLC0415
    except ImportError:
        return None
    return ParadataLogger(PROGRAM, config, paradata_dir=paradata_dir, config_dir=REPO_ROOT)


def license_detail_for(
    components: Sequence[str], logger: Optional[Any] = None
) -> Optional[Dict[str, Any]]:
    """The licence union (accretion rule 5) of the `para_config.txt` components used.

    Through `ParadataLogger`, so the licences come from the same `[components]` rows and the
    same `para_licenses` ranking every other stage uses. With no logger of the caller's, a
    throwaway one in a temporary directory computes the block and writes nothing.
    """
    if logger is not None:
        for name in components:
            logger.log_component(name)
        return logger.get_license_block()
    with tempfile.TemporaryDirectory(prefix="digital_convert_") as scratch:
        scratch_logger = _paradata_logger(scratch, {})
        if scratch_logger is None:
            return None
        return license_detail_for(components, scratch_logger)


def prepare(
    input_path: str,
    baseline: Optional[str] = None,
    doc_id: Optional[str] = None,
    run_id: Optional[str] = None,
    paradata_ref: str = "",
    out_dir: str = ".",
    strict: bool = False,
    engine: str = "light",
    docx_page_breaks: str = "auto",
    logger: Optional[Any] = None,
) -> Tuple[DigitalDocument, DocumentRecord, List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Layers A–C for one file: the normalised document, the record and its rows."""
    document = normalize(
        extract(input_path, doc_id=doc_id, engine=engine, docx_page_breaks=docx_page_breaks)
    )
    document.use("jsonschema")  # Layer D's gate runs on every record
    record, page_rows, line_rows = to_record(
        document,
        baseline=baseline,
        run_id=run_id,
        paradata_ref=paradata_ref,
        out_dir=out_dir,
        strict=strict,
        license_detail=license_detail_for(document.components, logger),
    )
    return document, record, page_rows, line_rows


# ──────────────────────────────────────────────────────────────────────────────
# Layer D — validation and export
# ──────────────────────────────────────────────────────────────────────────────


def _gate(
    record: DocumentRecord, page_rows: List[Dict[str, Any]], line_rows: List[Dict[str, Any]]
) -> Dict[str, Any]:
    record.assert_fields_survived("lines", line_rows)
    record.assert_fields_survived("pages", page_rows)
    data = record.to_dict()
    validate_document(data)
    return data


def emit(
    record: DocumentRecord,
    page_rows: List[Dict[str, Any]],
    line_rows: List[Dict[str, Any]],
    out_path: Optional[str] = None,
) -> str:
    """The output gate: round-trip assertion, then schema validation, then write.

    Both checks run BEFORE `finalize()`, so a record that fails either is never written —
    not written-then-flagged. They catch different things and neither subsumes the other:

      * `assert_fields_survived()` catches a **field-ownership drop**, which the schema
        structurally cannot. `lines[]` requires only `page`+`line`, so a row stripped of its
        `text` by a too-narrow grant is a *valid* row. That exact bug shipped once already:
        an earlier draft granted this program only `["group_id"]` on `lines`, `merge_block()`
        honoured it silently, and the resulting text-free records validated clean.
      * `validate_document()` catches everything the schema does describe — and it raises
        rather than passing when `jsonschema` is absent, because a gate that quietly
        no-ops is indistinguishable from a passing one.
    """
    _gate(record, page_rows, line_rows)
    return record.finalize(out_path)


def build_record(
    input_path: str,
    baseline: Optional[str] = None,
    doc_id: Optional[str] = None,
    engine: str = "light",
    docx_page_breaks: str = "auto",
    strict: bool = False,
) -> Dict[str, Any]:
    """The validated record as a dict, written nowhere — for in-memory consumers.

    `api_util/doc_to_visual_md.py` renders PDF/DOCX to Markdown through this, so the LLM's
    diet and the stored record come from one route (G8). Same gates as `emit()`: a record
    that fails the round-trip assertion or the schema is never returned.
    """
    _, record, page_rows, line_rows = prepare(
        input_path,
        baseline=baseline,
        doc_id=doc_id,
        strict=strict,
        engine=engine,
        docx_page_breaks=docx_page_breaks,
    )
    return _gate(record, page_rows, line_rows)


def convert(
    input_path: str,
    out_path: Optional[str] = None,
    baseline: Optional[str] = None,
    doc_id: Optional[str] = None,
    run_id: Optional[str] = None,
    paradata_ref: str = "",
    out_dir: str = ".",
    strict: bool = False,
    engine: str = "light",
    docx_page_breaks: str = "auto",
    paradata_dir: Optional[str] = None,
) -> str:
    """A → B → C → D for one file. Returns the written record's path.

    With `paradata_dir`, the run also writes its paradata record there (the pair the
    accretion model expects of every stage): its `run_id` and file name stamp the blocks
    unless the caller passed its own.
    """
    logger = None
    if paradata_dir:
        logger = _paradata_logger(
            paradata_dir,
            {
                "input": os.path.basename(input_path),
                "engine": engine,
                "docx_page_breaks": docx_page_breaks,
            },
        )
        if logger is not None:
            run_id = run_id or logger.run_id
            paradata_ref = paradata_ref or f"{logger.run_id}_{PROGRAM}.json"
    try:
        _, record, page_rows, line_rows = prepare(
            input_path,
            baseline=baseline,
            doc_id=doc_id,
            run_id=run_id,
            paradata_ref=paradata_ref,
            out_dir=out_dir,
            strict=strict,
            engine=engine,
            docx_page_breaks=docx_page_breaks,
            logger=logger,
        )
        written = emit(record, page_rows, line_rows, out_path=out_path)
    except Exception as exc:
        if logger is not None:
            logger.log_skip(input_path, getattr(exc, "reason", type(exc).__name__))
            logger.finalize(input_total=1, processed_total=0)
        raise
    if logger is not None:
        logger.log_success("document_json")
        logger.log_document_success()
        logger.finalize(input_total=1)
    return written


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Exposed separately so `--help` is testable in-process (repo convention I1)."""
    parser = argparse.ArgumentParser(
        prog="digital_to_json.py",
        description="Convert a digital-born PDF/DOCX into an atrium_document JSON record.",
        epilog="Exit codes: 0 written; 2 dependency missing; 3 not a born-digital PDF/DOCX; "
        "4 corrupt, encrypted or over the ZIP limits.",
    )
    parser.add_argument("input", help="path to the .pdf or .docx to convert")
    # W9: `--document-json-out` is the CANONICAL name, `--out` a retained alias.
    #
    # Every other stage in the ecosystem takes the PAIR --document-json /
    # --document-json-out (alto-postprocess, translator, page-classification,
    # nlp-enrich, llm-enrich). This converter took --document-json in and `--out`
    # out, so the one asymmetric spelling in the ecosystem sat on the newest tool —
    # and a pipeline step written by pattern-matching the others would silently write
    # to the default path instead of the requested one.
    #
    # `--out` still works: it is in digital_born/README.md and in the existing tests.
    # argparse resolves both to `args.out` via the shared dest.
    parser.add_argument(
        "--document-json-out",
        "--out",
        dest="out",
        default=None,
        help="output path for the record (default <out-dir>/<doc_id>.document.json). "
        "`--out` is a retained alias for the canonical --document-json-out.",
    )
    parser.add_argument(
        "--document-json",
        default=None,
        help="previous version of the record to accrete onto (rule 1); missing is not an error",
    )
    parser.add_argument("--doc-id", default=None, help="override the derived doc_id")
    parser.add_argument("--out-dir", default=".", help="directory for the default output path")
    parser.add_argument("--run-id", default=None, help="paradata run id to stamp blocks with")
    parser.add_argument("--paradata-ref", default="", help="paradata record this run wrote")
    parser.add_argument(
        "--paradata-dir",
        default=None,
        help="also write this run's paradata record into DIR (the paradata half of the pair)",
    )
    parser.add_argument(
        "--engine",
        choices=ENGINES,
        default="light",
        help="light (default): pdfplumber / python-docx. docling: Docling layout + table "
        "models for PDF structure — pip install -r requirements_digital_docling.txt",
    )
    parser.add_argument(
        "--docx-page-breaks",
        choices=DOCX_PAGE_BREAK_MODES,
        default="auto",
        help="DOCX pages: auto = explicit breaks + Word's last rendered breaks (default); "
        "explicit = explicit breaks only; none = one page",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="raise instead of warning on an ownership violation",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        written = convert(
            args.input,
            out_path=args.out,
            baseline=args.document_json,
            doc_id=args.doc_id,
            run_id=args.run_id,
            paradata_ref=args.paradata_ref,
            out_dir=args.out_dir,
            strict=args.strict,
            engine=args.engine,
            docx_page_breaks=args.docx_page_breaks,
            paradata_dir=args.paradata_dir,
        )
    except DigitalInputError as exc:
        print(f"[digital-convert] {exc.reason}: {exc}", file=sys.stderr)
        return exc.exit_code
    except RuntimeError as exc:  # a missing optional dependency, reported as advice
        print(f"[digital-convert] {exc}", file=sys.stderr)
        return 2
    print(f"[digital-convert] wrote {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
