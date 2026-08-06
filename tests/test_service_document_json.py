"""
tests/test_service_document_json.py
====================================
Tests for the document-record half of ``service/api.py`` — the ``document_json`` part in,
the updated record out.

That parameter had **zero** coverage, and the same blindness produced the P0 in
atrium-alto-postprocess's ``/process`` (atrium-project#10, J1): an endpoint that looks
wired, is exercised only through ``/info`` and ``/health`` by the meta-contract test, and
silently writes junk. What is pinned here is the accretion guarantee (an uploaded baseline's
blocks come back untouched) and the Layer D behaviour on the way out (D4).

``_run_extraction`` is driven directly rather than through ``TestClient``, because the
endpoint requires a warmed remote backend and this is deliberately a no-network test: the
engine dict below is exactly what ``_load_engine()`` returns, with a canned ``chat_fn`` in
place of the HTTP call. A ``.csv`` input is used so the real line-quality filter runs
unmodified.
"""

import json

import pytest

pytest.importorskip("fastapi")

from atrium_document import FILE_SUFFIX, load_document  # noqa: E402
from llm_client_shared import build_schema  # noqa: E402
from service.api import _run_extraction  # noqa: E402

DOC = "CTX000000001"

_CSV = (
    "page_num,line_num,text,categ,quality_score\n"
    "1,1,Výzkum odhalil základy gotického kostela.,Text,0.91\n"
)

_LLM_REPLY = json.dumps(
    {
        "extracted_keywords_cs": ["kostel"],
        "extracted_keywords_en": ["church"],
        "teater_category": "kostel",
        "confidence_score": 0.9,
    }
)


@pytest.fixture
def engine():
    """The subset of ``_load_engine()``'s dict that ``_run_extraction`` reads."""
    return {
        "backend": "openrouter",
        "model": "test/model",
        "line_prompt": "system prompt",
        "line_model": build_schema(["kostel"]),
        "line_chat_fn": lambda _messages: _LLM_REPLY,
        "filter_params": {
            "include_non_text": True,
            "min_char_count": 3,
            "min_char_non_text": 8,
            "min_alpha_ratio_non_text": 0.40,
        },
    }


@pytest.fixture
def csv_input(tmp_path):
    path = tmp_path / f"{DOC}.csv"
    path.write_text(_CSV, encoding="utf-8")
    return path


def _baseline(record_dir, **blocks):
    """A caller-uploaded baseline, written the way the endpoint writes it."""
    record_dir.mkdir(parents=True, exist_ok=True)
    path = record_dir / f"{DOC}{FILE_SUFFIX}"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "record_type": "atrium-document",
                "doc_id": DOC,
                **blocks,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return path


def test_uploaded_baseline_comes_back_with_only_enrichment_added(tmp_path, engine, csv_input):
    record_dir = tmp_path / "records"
    _baseline(
        record_dir,
        pages=[{"page": "1", "page_index": 1}],
        entities=[{"page": "1", "line": 1, "surface": "gotického kostela", "type_onto": "FAC"}],
    )

    result = _run_extraction(str(csv_input), csv_input.name, engine, DOC, record_dir)

    record = result["document_json"]
    assert record["doc_id"] == DOC
    assert record["assembled"]["had_baseline"] is True
    assert record["pages"] == [{"page": "1", "page_index": 1}]
    assert record["entities"][0]["surface"] == "gotického kostela"
    assert record["enrichment"]["items"][0]["extracted_keywords_en"] == ["church"]
    assert "document_json_schema_error" not in result


def test_returned_record_is_schema_checked_and_the_response_says_when_it_is_not(
    tmp_path, engine, csv_input
):
    """Layer D on the way out (D4).

    ``write_document_record()`` refuses to emit a record whose invalidity is ours, so what
    reaches here is a record it deliberately let through because the CALLER's baseline did
    not validate. Returning that in silence is the failure mode D4 is about; 500-ing on
    somebody else's data would contradict rule 6. So the record is returned, and the
    response carries the schema error in a field an automated caller can test.
    """
    record_dir = tmp_path / "records"
    _baseline(record_dir, pages="not-an-array")

    result = _run_extraction(str(csv_input), csv_input.name, engine, DOC, record_dir)

    assert "pages" in result["document_json_schema_error"]
    assert result["document_json"]["enrichment"]["items"]  # still returned (rule 6)
    assert result["document_json"]["pages"] == "not-an-array"  # and untouched


def test_layer_d_refusal_is_a_500_not_a_502(tmp_path, engine, csv_input, monkeypatch):
    """A record llm-enrich cannot emit is a defect on THIS side.

    ``_extract_from_path`` maps every ``RuntimeError`` to ``502 LLM backend error`` — correct
    for a ``chat_fn`` that exhausted its retries, wrong for the Layer D refusal, which would
    then blame the provider and invite a retry that fails identically.
    """
    from fastapi import HTTPException

    import llm_client_shared

    monkeypatch.setattr(
        llm_client_shared,
        "schema_gate",
        lambda record, what: None if what.startswith("baseline") else "deliberate failure",
    )
    record_dir = tmp_path / "records"

    with pytest.raises(HTTPException) as excinfo:
        _run_extraction(str(csv_input), csv_input.name, engine, DOC, record_dir)

    assert excinfo.value.status_code == 500
    assert "schema" in excinfo.value.detail
    assert not (record_dir / f"{DOC}{FILE_SUFFIX}").exists()


def test_no_baseline_still_returns_this_tools_own_part(tmp_path, engine, csv_input):
    """Rule 3: a standalone run is not an error, it just has one block."""
    record_dir = tmp_path / "records"

    result = _run_extraction(str(csv_input), csv_input.name, engine, DOC, record_dir)

    assert result["document_json"]["assembled"]["had_baseline"] is False
    assert load_document(str(record_dir / f"{DOC}{FILE_SUFFIX}"))["enrichment"]["items"]


def test_no_record_is_written_when_the_run_produced_nothing(tmp_path, engine):
    """No results -> no record, same as the batch entry points."""
    empty = tmp_path / f"{DOC}.csv"
    empty.write_text("page_num,line_num,text,categ,quality_score\n", encoding="utf-8")
    record_dir = tmp_path / "records"

    result = _run_extraction(str(empty), empty.name, engine, DOC, record_dir)

    assert result["results"] == []
    assert "document_json" not in result
    assert not (record_dir / f"{DOC}{FILE_SUFFIX}").exists()
