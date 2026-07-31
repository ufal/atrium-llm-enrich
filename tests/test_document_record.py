"""
tests/test_document_record.py
==============================
Tests for the paired per-document record (`atrium_document.py`) as wired into the
document-level clients through `llm_client_shared.write_document_record()`.

The record is built by **accretion**: each ATRIUM tool takes the previous version of
the JSON, if it is given one, and returns it with only its own block updated. These
tests pin the two properties that make that safe for llm-enrich specifically:

  * llm-enrich writes `enrichment` (and its own `derived_from`/`regenerable` keys) and
    **nothing else** — an upstream tool's blocks survive verbatim;
  * with no baseline the client still produces a valid standalone record.

Plus the projection from this repo's `*_enriched.json` shapes onto `enrichment.items[]`,
which is the part most likely to drift if the enrichment schema changes.
"""

import json
from pathlib import Path

import pytest

from llm_client_shared import enrichment_block, write_document_record

atrium_document = pytest.importorskip("atrium_document")
DocumentRecord = atrium_document.DocumentRecord
FILE_SUFFIX = atrium_document.FILE_SUFFIX
load_document = atrium_document.load_document

DOC = "CTX000000001"

# The two record shapes this repo writes to <doc_id>_enriched.json.
DOC_LEVEL = [
    {
        "file_id": DOC,
        "locator": "gotického kostela",
        "page": "1",
        "enrichment": {
            "extracted_keywords_cs": ["gotický kostel"],
            "extracted_keywords_en": ["gothic church"],
            "teater_category": "kostel",
            "confidence_score": 0.95,
        },
    }
]
LINE_LEVEL = [
    {
        "file_id": DOC,
        "page": 1,
        "line": 14,
        "categ": "Text",
        "quality_score": 0.98,
        "original_text": "Výzkum odhalil základy.",
        "enrichment": {
            "extracted_keywords_cs": ["základy"],
            "extracted_keywords_en": ["foundations"],
            "teater_category": "kostel",
            "confidence_score": 0.9,
        },
    }
]


# ──────────────────────────────────────────────────────────────────────────────
# enrichment_block() — projection of both record shapes
# ──────────────────────────────────────────────────────────────────────────────


def test_enrichment_block_flattens_document_level_records():
    item = enrichment_block(DOC, DOC_LEVEL)["items"][0]
    assert item["locator"] == "gotického kostela"
    assert item["page"] == "1"
    # the nested enrichment dict is flattened up into the item
    assert item["teater_category"] == "kostel"
    assert item["extracted_keywords_en"] == ["gothic church"]
    assert item["confidence_score"] == 0.95


def test_enrichment_block_keeps_line_numbers_for_line_level_records():
    item = enrichment_block(DOC, LINE_LEVEL)["items"][0]
    assert item["line"] == 14
    assert item["page"] == 1
    assert "locator" not in item  # line-level records have none


def test_enrichment_block_adds_page_citation():
    assert enrichment_block(DOC, DOC_LEVEL)["items"][0]["citation"] == f"[Source: {DOC}, Page 1]"


def test_enrichment_block_omits_citation_when_page_unknown():
    records = [{"locator": "x", "page": None, "enrichment": {}}]
    assert "citation" not in enrichment_block(DOC, records)["items"][0]


def test_enrichment_block_tolerates_missing_enrichment_key():
    assert enrichment_block(DOC, [{"locator": "x"}])["items"] == [{"locator": "x"}]


# ──────────────────────────────────────────────────────────────────────────────
# write_document_record() — standalone and accretion behaviour
# ──────────────────────────────────────────────────────────────────────────────


def test_writes_standalone_record_without_baseline(tmp_path):
    path = write_document_record(DOC, DOC_LEVEL, tmp_path, run_id="R1")

    assert path == tmp_path / f"{DOC}{FILE_SUFFIX}"
    record = load_document(str(path))
    assert record["doc_id"] == DOC
    assert record["enrichment"]["items"][0]["teater_category"] == "kostel"
    # rule 3: no baseline → this tool's own part only
    assert record["assembled"]["had_baseline"] is False
    assert "entities" not in record and "pages" not in record


def test_stamps_its_own_block_with_program_and_run_id(tmp_path):
    path = write_document_record(DOC, DOC_LEVEL, tmp_path, run_id="R1", paradata_ref="p.json")

    stamp = load_document(str(path))["assembled"]["blocks"]["enrichment"]
    assert stamp["program"] == "llm-enrich"
    assert stamp["run_id"] == "R1"
    assert stamp["paradata_ref"] == "p.json"


def test_preserves_upstream_blocks_verbatim(tmp_path):
    """The core accretion guarantee: llm-enrich must not disturb anyone else's block."""
    baseline = tmp_path / f"{DOC}{FILE_SUFFIX}"
    with DocumentRecord(DOC, "nlp-enrich", run_id="R0", out_dir=str(tmp_path)) as up:
        up.set_source(sha256="a" * 64, filename=f"{DOC}.alto.xml", origin="ABBYY-ALTO")
        up.set_block("entities", [{"surface": "gotického kostela", "type_onto": "FAC"}])
        up.merge_block("lines", [{"page": "1", "line": 14, "lemma": "výzkum"}])
    before = json.loads(baseline.read_text(encoding="utf-8"))

    write_document_record(DOC, DOC_LEVEL, tmp_path, run_id="R1")
    after = load_document(str(baseline))

    for block in ("source", "entities", "lines"):
        assert after[block] == before[block], f"{block} was modified by llm-enrich"
    assert after["assembled"]["blocks"]["entities"]["program"] == "nlp-enrich"
    assert after["assembled"]["blocks"]["enrichment"]["program"] == "llm-enrich"
    assert after["enrichment"]["items"][0]["locator"] == "gotického kostela"
    assert after["assembled"]["had_baseline"] is True


def test_records_derived_from_and_markdown_recipe(tmp_path):
    path = write_document_record(
        DOC,
        DOC_LEVEL,
        tmp_path,
        run_id="R1",
        enriched_path=tmp_path / f"{DOC}_enriched.json",
        markdown_from=tmp_path / f"{DOC}.pdf",
    )
    record = load_document(str(path))

    assert record["derived_from"]["enriched"].endswith(f"{DOC}_enriched.json")
    recipe = record["regenerable"]["markdown"]
    assert recipe["from"].endswith(f"{DOC}.pdf")
    assert recipe["detail"] == "full"
    # a recipe, never a stored path to the disposable Markdown itself
    assert not json.dumps(record).count(f"{DOC}.md")


def test_no_markdown_recipe_when_input_needed_no_conversion(tmp_path):
    path = write_document_record(DOC, DOC_LEVEL, tmp_path, run_id="R1", markdown_from=None)
    assert "regenerable" not in load_document(str(path))


def test_used_markdown_input_writes_self_referential_json_to_md_recipe(tmp_path):
    """A document-level run (real Markdown fed to the LLM, whether from a pre-converted
    PDF/DOCX or an upstream xml_to_md.py TEITOK pass) gets a recipe pointing at THIS
    SAME document JSON via json_to_md — self-sufficient, no external file required."""
    path = write_document_record(DOC, DOC_LEVEL, tmp_path, run_id="R1", used_markdown_input=True)
    recipe = load_document(str(path))["regenerable"]["markdown"]

    assert recipe["from"] == f"{DOC}{FILE_SUFFIX}"
    assert recipe["converter"] == "json_to_md@1.0"
    assert recipe["detail"] == "full"


def test_used_markdown_input_takes_priority_over_markdown_from(tmp_path):
    path = write_document_record(
        DOC,
        DOC_LEVEL,
        tmp_path,
        run_id="R1",
        markdown_from=tmp_path / f"{DOC}.pdf",
        used_markdown_input=True,
    )
    recipe = load_document(str(path))["regenerable"]["markdown"]
    assert recipe["converter"] == "json_to_md@1.0"


def test_no_markdown_recipe_for_line_level_run(tmp_path):
    """A line-level run never fed Markdown to the LLM, so no recipe should claim one
    can be regenerated, even when a source file happens to be passed in."""
    path = write_document_record(
        DOC, LINE_LEVEL, tmp_path, run_id="R1", used_markdown_input=False, markdown_from=None
    )
    assert "regenerable" not in load_document(str(path))


def test_rerun_replaces_only_its_own_block(tmp_path):
    """Granularity: a second llm-enrich run rewrites `enrichment`, not the neighbours."""
    write_document_record(DOC, DOC_LEVEL, tmp_path, run_id="R1")
    path = tmp_path / f"{DOC}{FILE_SUFFIX}"
    with DocumentRecord.open(DOC, "translator", baseline=str(path), out_dir=str(tmp_path)) as tr:
        tr.set_block("translations", {"source_lang": "cs", "target_lang": "en"})

    write_document_record(DOC, LINE_LEVEL, tmp_path, run_id="R2")
    record = load_document(str(path))

    assert record["translations"]["target_lang"] == "en"  # survived
    assert record["enrichment"]["items"][0]["line"] == 14  # replaced
    assert record["assembled"]["blocks"]["enrichment"]["run_id"] == "R2"


def test_license_detail_accretes_into_provenance(tmp_path):
    path = write_document_record(
        DOC,
        DOC_LEVEL,
        tmp_path,
        run_id="R1",
        license_detail={"components": [{"name": "nametag3", "license": "CC BY-NC-SA 4.0"}]},
    )
    provenance = load_document(str(path))["provenance"]

    assert provenance["license"] == "CC BY-NC-SA 4.0"
    assert provenance["contributors"][0]["program"] == "llm-enrich"


def test_record_validates_against_the_shared_schema(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    schema_path = Path(__file__).resolve().parent.parent / "atrium_document.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    path = write_document_record(DOC, DOC_LEVEL, tmp_path, run_id="R1")
    jsonschema.validate(load_document(str(path)), schema)
