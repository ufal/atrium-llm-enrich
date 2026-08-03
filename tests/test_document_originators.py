"""
tests/test_document_originators.py

Issue #18 §1a — block ownership for digital-born documents.

Four blocks (`pages`, `content`, `lines`, `tables`) describe a document's POSITIONAL
PLANE, and there are two ways to acquire one: OCR/ALTO, or direct extraction from a
digital-born PDF/DOCX. They are mutually exclusive per document, so `BLOCK_OWNERS` lists
both possible originators and `source.origin` picks between them per record.

These tests pin three things that were each a live defect before this landed:

  1. `digital-convert` can actually originate the positional blocks (it could not — every
     path either raised or silently ate the payload).
  2. The two originators cannot be mixed on one document, and `merge_block()` is checked
     as well as `set_block()` (it bypassed `_assert_owner()` entirely, so `pages` and
     `lines` were never ownership-checked at all).
  3. `lines[]` rows survive the round trip with their `text` and `bbox` intact. The
     earlier `BLOCK_FIELD_OWNERS` draft granted the converter only `["group_id"]`, which
     `merge_block()` honours SILENTLY — `text` was filtered out with no warning and the
     record still validated, because `lines[]` requires only `page`+`line`.

Style follows tests/test_atrium_document.py (one rule per test, named in the docstring),
but uses the public `logger.run_id` rather than `logger._run_id` — the #13 hardening pass
added it precisely so tests stop reaching into privates.
"""

import json

import pytest

from atrium_document import (
    BLOCK_FIELD_OWNERS,
    BLOCK_OWNERS,
    ORIGIN_ORIGINATORS,
    DocumentRecord,
)
from atrium_paradata import ParadataLogger

DIGITAL = "digital-convert"
ALTO = "alto-postprocess"


@pytest.fixture
def mock_paradata(tmp_path):
    """Generates a mock paradata record and returns the run_id and ref."""
    para_dir = tmp_path / "paradata"
    para_dir.mkdir()
    logger = ParadataLogger(DIGITAL, {}, paradata_dir=str(para_dir))
    logger.log_component("docling")
    paradata_ref = logger.finalize()
    return logger.run_id, paradata_ref


def _open(tmp_path, mock_paradata, program, origin=None, baseline=None, strict=True):
    """A record opened as `program`, with `source.origin` seeded when given."""
    run_id, paradata_ref = mock_paradata
    doc = DocumentRecord(
        doc_id="CTX000000001",
        program=program,
        baseline=baseline,
        run_id=run_id,
        paradata_ref=paradata_ref,
        strict=strict,
    )
    if origin is not None:
        doc.set_source(origin=origin, filename="CTX000000001.pdf")
    return doc


# ── the table itself ─────────────────────────────────────────────────────────


def test_positional_blocks_declare_two_originators():
    """The positional plane has two possible originators; everything else has one."""
    for block in ("pages", "content", "lines", "tables"):
        assert BLOCK_OWNERS[block] == (ALTO, DIGITAL), block
    for block in ("page_categories", "translations", "entities", "enrichment", "forms"):
        assert isinstance(BLOCK_OWNERS[block], str), block


def test_digital_convert_grant_includes_text_and_bbox():
    """The regression that made the first §1 landing silently lossy.

    A grant of only ["group_id"] passes every check and still produces a record with no
    text in it, because merge_block() filters unowned fields without complaining and the
    schema requires only page+line on a lines[] row.
    """
    grant = BLOCK_FIELD_OWNERS["lines"][DIGITAL]
    assert "text" in grant, "the converter must be able to originate line text"
    assert "bbox" in grant, "the PDF adapter's native coordinates have nowhere else to go"
    assert "group_id" in grant


def test_no_program_named_llm_enrich_digital_survives():
    """Renamed to `digital-convert`: these identities are roles, not repo names, and they
    are permanent in provenance.contributors[] once exported."""
    for block, owners in BLOCK_FIELD_OWNERS.items():
        assert "llm-enrich-digital" not in owners, block


def test_every_originator_is_reachable_from_some_origin():
    """A candidate in BLOCK_OWNERS with no ORIGIN_ORIGINATORS prefix could never write."""
    reachable = {originator for _prefix, originator in ORIGIN_ORIGINATORS}
    for block, owners in BLOCK_OWNERS.items():
        if isinstance(owners, tuple):
            assert set(owners) <= reachable, block


# ── set_block: content (not field-split, so the owner check is the whole story) ──


def test_digital_convert_may_originate_content_for_a_digital_born_doc(tmp_path, mock_paradata):
    out = tmp_path / "digital.document.json"
    with _open(tmp_path, mock_paradata, DIGITAL, origin="digital-born-pdf") as doc:
        doc.set_block("content", {"text": "Digital-born body text.", "reading_order": "ltr"})
        doc.finalize(str(out))

    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["content"]["text"] == "Digital-born body text."
    # Rule 4: the read-time contract names the tool that actually wrote it.
    assert record["assembled"]["blocks"]["content"]["program"] == DIGITAL


def test_alto_postprocess_still_owns_content_on_the_ocr_path(tmp_path, mock_paradata):
    """Regression guard: widening authorisation must not change the OCR path at all."""
    out = tmp_path / "alto.document.json"
    with _open(tmp_path, mock_paradata, ALTO, origin="ocr:tesseract-ces") as doc:
        doc.set_block("content", {"text": "OCR'd body text."})
        doc.finalize(str(out))

    record = json.loads(out.read_text(encoding="utf-8"))
    assert record["assembled"]["blocks"]["content"]["program"] == ALTO


def test_origin_mismatch_is_refused(tmp_path, mock_paradata):
    """A digital converter must not claim the positional plane of an OCR'd document."""
    doc = _open(tmp_path, mock_paradata, DIGITAL, origin="ocr:pero")
    with pytest.raises(ValueError, match="originated by 'alto-postprocess'"):
        doc.set_block("content", {"text": "wrong originator"})


def test_origin_mismatch_the_other_way_is_also_refused(tmp_path, mock_paradata):
    doc = _open(tmp_path, mock_paradata, ALTO, origin="digital-born-pdf")
    with pytest.raises(ValueError, match="originated by 'digital-convert'"):
        doc.set_block("content", {"text": "wrong originator"})


def test_a_third_program_is_still_refused_outright(tmp_path, mock_paradata):
    """Widening to two candidates must not widen to anyone."""
    doc = _open(tmp_path, mock_paradata, "translator", origin="digital-born-pdf")
    with pytest.raises(ValueError, match="is owned by"):
        doc.set_block("content", {"text": "not yours"})


def test_unknown_origin_abstains_rather_than_blocking(tmp_path, mock_paradata):
    """Rule 6's spirit: a new origin string may land before this table learns it."""
    with _open(tmp_path, mock_paradata, DIGITAL, origin="some-future-acquisition") as doc:
        doc.set_block("content", {"text": "allowed"})
        assert doc.get_block("content")["text"] == "allowed"


def test_no_source_yet_abstains(tmp_path, mock_paradata):
    """Rule 3: a standalone run with no baseline and no source still emits its own part."""
    with _open(tmp_path, mock_paradata, DIGITAL, origin=None) as doc:
        doc.set_block("content", {"text": "standalone"})
        assert doc.get_block("content")["text"] == "standalone"


# ── merge_block: pages / lines (field-split, and previously unchecked) ────────


def test_digital_convert_lines_round_trip_keeps_text_and_bbox(tmp_path, mock_paradata):
    """The silent-drop regression, end to end.

    Every field handed in must come back. Asserting on the record rather than on the
    grant is the point: the grant is what broke, and a test that reads the grant back
    would have passed while the data was being eaten.
    """
    rows = [
        {"page": "1", "line": 0, "text": "První odstavec, řádek jedna.",
         "bbox": [72.0, 700.0, 300.0, 712.0], "group_id": "p1", "lang": "cs"},
        {"page": "1", "line": 1, "text": "První odstavec, řádek dvě.",
         "bbox": [72.0, 686.0, 290.0, 698.0], "group_id": "p1", "lang": "cs"},
        {"page": "1", "line": 2, "text": "Druhý odstavec.",
         "bbox": [72.0, 660.0, 250.0, 672.0], "group_id": "p2", "lang": "cs"},
    ]
    out = tmp_path / "lines.document.json"
    with _open(tmp_path, mock_paradata, DIGITAL, origin="digital-born-pdf") as doc:
        doc.merge_block("lines", rows)
        doc.finalize(str(out))

    written = json.loads(out.read_text(encoding="utf-8"))["lines"]
    assert len(written) == len(rows)
    for handed_in, came_back in zip(rows, written, strict=False):
        for field, value in handed_in.items():
            assert came_back.get(field) == value, f"{field} was dropped by merge_block"


def test_digital_convert_may_write_page_canvas_and_needs_ocr(tmp_path, mock_paradata):
    """`canvas` carries PyMuPDF/pdfplumber page geometry; `needs_ocr` is the digital->OCR
    handoff for Issue #10's undecodable-text-layer case."""
    out = tmp_path / "pages.document.json"
    with _open(tmp_path, mock_paradata, DIGITAL, origin="digital-born-pdf") as doc:
        doc.merge_block(
            "lines", [{"page": "1", "line": 0, "text": "sondI"}]
        )
        doc.merge_block(
            "pages",
            [{"page": "1", "page_index": 1,
              "canvas": {"width": 612, "height": 792, "unit": "pt"},
              "quality_score": 0.21, "quality_band": "Trash", "needs_ocr": True}],
        )
        doc.finalize(str(out))

    page = json.loads(out.read_text(encoding="utf-8"))["pages"][0]
    assert page["canvas"]["width"] == 612
    assert page["needs_ocr"] is True
    assert page["quality_band"] == "Trash"
    # `ocr` stays unowned by the converter: "was this OCR'd" must remain answerable.
    assert "ocr" not in page


def test_merge_block_enforces_origin_too(tmp_path, mock_paradata):
    """merge_block() never called _assert_owner(), so pages/lines skipped the check
    entirely — the half of §1a that the original write-up missed."""
    doc = _open(tmp_path, mock_paradata, DIGITAL, origin="ABBYY-ALTO")
    with pytest.raises(ValueError, match="originated by 'alto-postprocess'"):
        doc.merge_block("lines", [{"page": "1", "line": 0, "text": "x"}])


def test_nlp_enrich_merging_into_lines_is_unaffected(tmp_path, mock_paradata):
    """A field contributor is not an origination claim: nlp-enrich is not a candidate
    originator, so the origin check must abstain for it on either path."""
    baseline = tmp_path / "CTX000000001.document.json"
    baseline.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "doc_id": "CTX000000001",
                "source": {"origin": "digital-born-pdf"},
                "lines": [{"page": "1", "line": 0, "text": "Původní řádek.", "group_id": "p1"}],
            }
        ),
        encoding="utf-8",
    )
    out = tmp_path / "out.document.json"
    run_id, paradata_ref = mock_paradata
    with DocumentRecord.open(
        doc_id="CTX000000001",
        program="nlp-enrich",
        baseline=str(baseline),
        run_id=run_id,
        paradata_ref=paradata_ref,
        strict=True,
    ) as doc:
        doc.merge_block("lines", [{"page": "1", "line": 0, "lemma": "původní", "upos": "ADJ"}])
        doc.finalize(str(out))

    line = json.loads(out.read_text(encoding="utf-8"))["lines"][0]
    assert line["lemma"] == "původní"
    assert line["text"] == "Původní řádek."   # rule 2: co-contributor's field survives
    assert line["group_id"] == "p1"           # and so does the originator's


# ── the read-time contract ───────────────────────────────────────────────────


def test_read_time_contract_is_the_stamp_not_the_table(tmp_path, mock_paradata):
    """BLOCK_OWNERS authorises writes. Who actually wrote a block in a GIVEN record is
    assembled.blocks[<block>].program — the claim made in docs/document_schema.md, pinned
    here so it is enforced rather than merely asserted."""
    out = tmp_path / "stamped.document.json"
    with _open(tmp_path, mock_paradata, DIGITAL, origin="docx") as doc:
        doc.set_block("content", {"text": "From a DOCX."})
        doc.merge_block("lines", [{"page": "1", "line": 0, "text": "From a DOCX."}])
        doc.finalize(str(out))

    blocks = json.loads(out.read_text(encoding="utf-8"))["assembled"]["blocks"]
    assert blocks["content"]["program"] == DIGITAL
    assert blocks["lines"]["program"] == DIGITAL
    assert BLOCK_OWNERS["lines"] != DIGITAL      # the table alone would have said otherwise


def test_contributors_records_the_originating_run(tmp_path, mock_paradata):
    """Rule 4: the full picture lives in provenance.contributors[], which is what makes
    rejecting the 'no-op alto passthrough' option matter — that option would have put
    alto-postprocess in here for a run that did nothing."""
    run_id, _ = mock_paradata
    out = tmp_path / "prov.document.json"
    with _open(tmp_path, mock_paradata, DIGITAL, origin="digital-born-pdf") as doc:
        doc.set_block("content", {"text": "x"})
        doc.finalize(str(out))

    contributors = json.loads(out.read_text(encoding="utf-8"))["provenance"]["contributors"]
    mine = [c for c in contributors if c["program"] == DIGITAL]
    assert len(mine) == 1
    assert mine[0]["run_id"] == run_id
    assert "content" in mine[0]["blocks"]
    assert not any(c["program"] == ALTO for c in contributors)


# ── non-strict behaviour ─────────────────────────────────────────────────────


def test_non_strict_warns_instead_of_raising(tmp_path, mock_paradata, capsys):
    """`strict=False` stays a warning, matching every other _complain() path."""
    doc = _open(tmp_path, mock_paradata, DIGITAL, origin="ocr:pero", strict=False)
    doc.set_block("content", {"text": "written anyway"})
    err = capsys.readouterr().err
    assert "WARNING" in err
    assert "originated by 'alto-postprocess'" in err
    assert doc.get_block("content")["text"] == "written anyway"
