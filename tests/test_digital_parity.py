"""The #10 §9 measurement, as a test: every case the JSON route failed, end to end.

`agent_dev_logs/plans/10.plan.md` §9 ran the same inputs through markitdown, flexiconv, the
legacy direct converters and the JSON route (`digital_to_json` → `json_to_md`) and found the
JSON route behind on eight points (G1–G8). This module is that table's JSON-route column,
re-run on every commit against the generated fixtures: the Markdown the clients' auto-
convert now produces (`doc_to_visual_md.convert_to_visual_md`) must contain what the row
requires and nothing the row forbids.

Two-column ordering on a REAL complex page stays out of scope here (a Docling/MinerU
bake-off on the hub #22 harness, #10 §10 Phase 6); `two_column.pdf` pins the simple case.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from api_util import doc_to_visual_md

GENERATOR = Path(__file__).resolve().parent / "fixtures" / "digital" / "make_fixtures.py"


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    spec = importlib.util.spec_from_file_location("make_fixtures", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path_factory.mktemp("digital")
    for name, payload in module.build_all().items():
        (out / name).write_bytes(payload)
    return out


#: (§9 row, fixture, must appear in order, must not appear)
CASES = [
    ("page boundaries (PDF)", "minimal.pdf", ["## Page 1", "PAGE_BREAK: pg_2", "## Page 2"], []),
    (
        "page boundaries + heading + table (DOCX)",
        "minimal.docx",
        [
            "### Zpráva o sondě",
            "| Vrstva | Mocnost |",
            "| Ornice | 30 cm |",
            "## Page 2",
            "Třetí odstavec, už na druhé straně.",
        ],
        [],
    ),
    (
        "G1 text-less pages flagged, and told apart",
        "image_only.pdf",
        [
            "## Page 1",
            "## Page 2",
            "NEEDS_OCR: pg_2 (no extractable text layer: the page draws 1 image(s)",
            "## Page 3",
            "NEEDS_OCR: pg_3 (no extractable text layer: the page draws nothing",
        ],
        [],
    ),
    (
        "G6 garbled text layer: page flagged, no bad line leaks",
        "garbled.pdf",
        ["NEEDS_OCR: pg_1 (embedded text layer does not decode: 3 of 3 lines"],
        ["sondì", "høeby", "mìla"],
    ),
    (
        "G2 words separated, columns in order, furniture, roman labels",
        "two_column.pdf",
        [
            "## Page i",
            "HEADER_START",
            "### Hradiste u Horni Mezi: zachranny vyzkum",
            "Sonda II odkryla val.",
            "Konec leveho sloupce.",
            "Pravy sloupec zacina zde.",
            "FOOTER_START",
            "## Page ii",
        ],
        ["SondaIIodkryla"],
    ),
    (
        "PDF ruled table (legacy pdf_to_md parity)",
        "table.pdf",
        [
            "Tabulka vrstev je nize.",
            "| Vrstva | Mocnost | Nalezy |",
            "| Ornice | 30 cm | keramika |",
            "Text pod tabulkou.",
        ],
        [],
    ),
    (
        "G4 package without the main-part Override",
        "bare.docx",
        ["### Zpráva o sondě", "## Page 2"],
        [],
    ),
    (
        "G5 tracked changes, furniture, footnote, pages from three kinds of break",
        "rich.docx",
        [
            "HEADER_START",
            "### Hradiště u Horní Mezi",
            "#### Průběh výzkumu",
            "**Sonda II**",
            "Nalezena bronzová spona, viz katalog.",
            "[^1]: Katalog nálezů je uložen v archivu.",
            "## Page 2",
            "| Nálezy |  |",
            "FOOTER_START",
            "## Page 3",
            "## Page 4",
        ],
        ["železný nůž", "PAGE "],
    ),
]


@pytest.mark.parametrize("row, name, required, forbidden", CASES, ids=[c[0] for c in CASES])
def test_json_route_closes_the_section_9_row(fixtures, row, name, required, forbidden):
    pytest.importorskip("pdfplumber")
    pytest.importorskip("docx")
    pytest.importorskip("jsonschema")
    md = doc_to_visual_md.convert_to_visual_md(fixtures / name)
    positions = []
    for needle in required:
        assert needle in md, f"{row}: {needle!r} missing from\n{md}"
        positions.append(md.index(needle))
    assert positions == sorted(positions), f"{row}: out of order in\n{md}"
    for needle in forbidden:
        assert needle not in md, f"{row}: {needle!r} must not appear in\n{md}"


def test_ocr_layer_pdf_is_refused_on_the_json_route(fixtures):
    """G7: the auto-convert must not render an OCR layer as born-digital text either."""
    pytest.importorskip("pdfplumber")
    from api_util.digital_ir import DigitalInputError

    with pytest.raises(DigitalInputError, match="not born-digital"):
        doc_to_visual_md.convert_to_visual_md(fixtures / "ocr_layer.pdf")


def test_a_docx_born_record_is_enriched_by_the_llm_stage(fixtures, remote_client_env, stub_llm):
    """#18's pipeline-compatibility criterion, for the format the hub smoke does not run:
    DOCX -> digital_to_json -> openrouter_client (`--input` the record, the
    `--document-json`/`--document-json-out` pair, as the smoke's stage D2 does), with the model
    stubbed. The record is ingested without schema or structural errors, `enrichment` lands,
    and the digital-convert blocks — styles, regions, tables — come through untouched."""
    pytest.importorskip("docx")
    pytest.importorskip("jsonschema")
    import openrouter_client
    from api_util import digital_to_json
    from atrium_document import load_document, validate_document

    env = remote_client_env
    record_in = env.root / "rich.document.json"
    digital_to_json.convert(str(fixtures / "rich.docx"), out_path=str(record_in))
    before = load_document(str(record_in))

    stub_llm(openrouter_client)
    record_out = env.root / "rich.llm.document.json"
    openrouter_client.main(
        [
            "--config",
            str(env.config),
            "--input",
            str(record_in),
            "--output-dir",
            str(env.output_dir),
            "--model",
            "test/model",
            "--api-key",
            "test-key",
            "--document-json",
            str(record_in),
            "--document-json-out",
            str(record_out),
        ]
    )

    after = load_document(str(record_out))
    validate_document(after)
    assert "enrichment" in after
    for block in ("source", "pages", "lines", "content", "tables"):
        assert after[block] == before[block], f"{block} changed in the llm stage"
    assert after["assembled"]["blocks"]["lines"]["program"] == "digital-convert"
