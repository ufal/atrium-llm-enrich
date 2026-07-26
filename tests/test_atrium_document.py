import pytest
from atrium_document import AtriumDocument, SourceInfo, ProvenanceInfo, PageInfo, ContentInfo, LineInfo, EntityInfo, \
    EnrichmentInfo, EnrichmentItem


def test_atrium_document_full_instantiation_and_serialization():
    # Test instantiating all models in atrium_document.py to cover unused branches and achieve required test coverage
    source = SourceInfo(
        doc_id="CTX999999999",
        filename="CTX999999999.alto.xml",
        media_type="application/alto+xml",
        sha256="abc123hash",
        page_count=2,
        language=["cs"],
        origin="ABBYY-ALTO"
    )

    provenance = ProvenanceInfo(
        run_id="260726-120000",
        pipeline="pc->alto->translate->nlp->llm",
        paradata_ref="paradata/260726-120000_pipeline-run.json",
        license="CC BY-NC-SA 4.0"
    )

    page = PageInfo(
        page="1",
        page_index=1,
        quality_score=0.95,
        quality_band="Clear",
        canvas={"width": 612, "height": 792, "unit": "pt"},
        needs_ocr=False
    )

    content = ContentInfo(
        text="Test document full reading order plain text.",
        reading_order="ltr-columns"
    )

    line = LineInfo(
        page="1",
        line=1,
        categ="Text",
        quality_score=0.95,
        bbox=[0, 0, 100, 20],
        teitok_ref="CTX999999999.s1",
        text="Test line content."
    )

    entity = EntityInfo(
        surface="Test Entity",
        lemma="test entity",
        type_onto="ORG",
        type_cnec="ic",
        type_teitok="ORG",
        page="1",
        line=1,
        char_span=[0, 11],
        bbox=[0, 0, 50, 20],
        teitok_ref="CTX999999999.name1",
        translation_en="Test Entity EN"
    )

    enrichment_item = EnrichmentItem(
        locator="Test Entity",
        page="1",
        extracted_keywords_cs=["test"],
        extracted_keywords_en=["test"],
        teater_category="test",
        confidence_score=0.99,
        citation="[Source: CTX999999999, Page 1]"
    )

    enrichment = EnrichmentInfo(
        items=[enrichment_item],
        summary="Test summary.",
        topics=["testing"],
        page_categories={"1": "Text"}
    )

    doc = AtriumDocument(
        schema_version="1.0",
        doc_id="CTX999999999",
        source=source,
        provenance=provenance,
        pages=[page],
        content=content,
        lines=[line],
        entities=[entity],
        enrichment=enrichment
    )

    # Verify serialization and validation behaviors
    data = doc.model_dump()
    assert data["doc_id"] == "CTX999999999"
    assert data["source"]["filename"] == "CTX999999999.alto.xml"
    assert len(data["pages"]) == 1
    assert data["content"]["text"] == "Test document full reading order plain text."

    # Test round-trip reconstruction from state dict
    json_str = doc.model_dump_json()
    reconstructed = AtriumDocument.model_validate_json(json_str)
    assert reconstructed.doc_id == doc.doc_id
    assert reconstructed.source.sha256 == doc.source.sha256