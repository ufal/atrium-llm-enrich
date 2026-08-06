"""
tests/conftest.py
=================
Shared pytest fixtures and sys.path wiring for atrium-nlp-enrich unit tests.

sys.path is patched here (once, at collection time) so that every test module
can import from both the repo root (``keywords.py``, ``atrium_paradata.py``)
and the ``api_util/`` subdirectory (``call_udpipe``, ``call_nametag``,
``summarize_nt_udp``).
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def pytest_configure(config):
    # `slow` is declared in pytest.ini; this stays as the belt-and-braces registration for
    # anyone invoking pytest with a different ini (e.g. `-c` in a container), and is
    # harmless when both are present. With --strict-markers now on, one of the two has to
    # be authoritative — pytest.ini is.
    config.addinivalue_line("markers", "slow: marks tests as slow integration smoke tests")


# ── path wiring ───────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "api_util"))

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ── CoNLL-U fixtures (shared with atrium-nlp-enrich; teitok_alto is identical) ──


@pytest.fixture
def sample_conllu(tmp_path):
    """Three-sentence CoNLL-U file written to a temp path."""
    content = (FIXTURES_DIR / "sample.conllu").read_text(encoding="utf-8")
    dest = tmp_path / "sample.conllu"
    dest.write_text(content, encoding="utf-8")
    return str(dest)


@pytest.fixture
def two_page_conllu(tmp_path):
    """CoNLL-U whose sent_id counter resets to 1 mid-file (two-page document)."""
    content = (FIXTURES_DIR / "two_page.conllu").read_text(encoding="utf-8")
    dest = tmp_path / "two_page.conllu"
    dest.write_text(content, encoding="utf-8")
    return str(dest)


@pytest.fixture
def page_break_conllu(tmp_path):
    """Merged CoNLL-U using ``# page_break = true`` comments instead of sent_id resets."""
    content = (FIXTURES_DIR / "page_break.conllu").read_text(encoding="utf-8")
    dest = tmp_path / "page_break.conllu"
    dest.write_text(content, encoding="utf-8")
    return str(dest)


@pytest.fixture
def empty_conllu(tmp_path):
    """CoNLL-U file with only a comment header — no token lines."""
    dest = tmp_path / "empty.conllu"
    dest.write_text("# newdoc\n", encoding="utf-8")
    return str(dest)


# ── remote-client end-to-end scaffolding (atrium-project#10, D1) ──────────────
#
# openrouter_client.main() and ollama_client.main() were both untestable end-to-end: every
# path they read (config, vocabulary, paradata, output) is repo-relative, and the first
# thing they do is build an HTTP session. The fixtures below redirect all of it into
# tmp_path and stub the one network call, so the D1 regression — a `.teitok.xml` input
# whose doc_id forked and discarded every upstream block — can be pinned by running the
# real `main()` rather than a re-implementation of it.

#: The doc_id the whole pipeline uses for the fixture document. Deliberately a name whose
#: `Path.stem` is WRONG (`CTX000000001.teitok`), because that is the bug.
E2E_DOC_ID = "CTX000000001"

_TEITOK_FIXTURE = """<?xml version="1.0" encoding="UTF-8"?>
<teiCorpus>
    <text>
        <pb n="1"/>
        <s text="Výzkum odhalil základy gotického kostela."/>
        <lb/>
        <s text="Sonda I byla založena roku 1956."/>
    </text>
</teiCorpus>
"""


@pytest.fixture
def remote_client_env(tmp_path):
    """A self-contained working tree for a one-document openrouter/ollama ``main()`` run."""
    vocab = tmp_path / "vocab.json"
    vocab.write_text(
        json.dumps({"Site Types": {"kostel": {"en": "church"}}}, ensure_ascii=False),
        encoding="utf-8",
    )

    config = tmp_path / "llm_config.txt"
    config.write_text(
        "\n".join(
            [
                f"VOCAB_PATH={vocab}",
                f"PARADATA_DIR={tmp_path / 'paradata'}",
                f"OUTPUT_DIR={tmp_path / 'out'}",
                "INCLUDE_NON_TEXT=true",
                "",
            ]
        ),
        encoding="utf-8",
    )

    teitok = tmp_path / f"{E2E_DOC_ID}.teitok.xml"
    teitok.write_text(_TEITOK_FIXTURE, encoding="utf-8")

    record_dir = tmp_path / "doc_json"
    record_dir.mkdir()

    return SimpleNamespace(
        root=tmp_path,
        doc_id=E2E_DOC_ID,
        config=config,
        teitok=teitok,
        output_dir=tmp_path / "out",
        record_dir=record_dir,
        # Where an upstream tool writes it, and the only place llm-enrich may write it.
        baseline=record_dir / f"{E2E_DOC_ID}.document.json",
        # Where the pre-fix `Path.stem` derivation looked instead — nothing may appear here.
        forked_record=record_dir / f"{E2E_DOC_ID}.teitok.document.json",
    )


@pytest.fixture
def seeded_baseline(remote_client_env):
    """Pre-seed ``--document-json-dir`` with the record the upstream stages leave behind.

    Written by the real tools (alto-postprocess originates pages/lines under an
    `ABBYY-ALTO` origin, nlp-enrich adds entities) rather than hand-rolled, so the blocks
    llm-enrich must preserve carry genuine ownership stamps. Asserted schema-valid here on
    purpose: an invalid baseline would demote the Layer D gate to a warning (D4) and quietly
    change what the tests using this fixture are measuring.
    """
    from atrium_document import DocumentRecord, load_document, validate_document

    env = remote_client_env
    with DocumentRecord(
        env.doc_id, "alto-postprocess", run_id="R-ALTO", out_dir=str(env.record_dir)
    ) as alto:
        alto.set_source(
            sha256="a" * 64,
            filename=f"{env.doc_id}.alto.xml",
            media_type="application/alto+xml",
            origin="ABBYY-ALTO",
            page_count=1,
        )
        alto.merge_block("pages", [{"page": "1", "page_index": 1, "quality_score": 0.91}])
        alto.merge_block(
            "lines",
            [
                {
                    "page": "1",
                    "line": 1,
                    "text": "Výzkum odhalil základy gotického kostela.",
                    "categ": "Text",
                    "quality_score": 0.91,
                }
            ],
        )

    with DocumentRecord.open(
        env.doc_id, "nlp-enrich", baseline=str(env.baseline), out_dir=str(env.record_dir)
    ) as nlp:
        nlp.merge_block(
            "entities",
            [
                {
                    "page": "1",
                    "line": 1,
                    "char_span": [24, 41],
                    "surface": "gotického kostela",
                    "type_onto": "FAC",
                }
            ],
        )

    validate_document(load_document(str(env.baseline)))
    return env.baseline


@pytest.fixture
def stub_llm(monkeypatch):
    """Make a client module's inference offline. Call it with the module under test.

    Two things are neutralised, and the second one is not cosmetic:

    * ``make_chat_fn`` — replaced with a canned, schema-valid reply, so no HTTP happens and
      no API key or Ollama daemon is needed.
    * ``should_process_line`` — a pass-through. TEITOK rows carry no quality signal, so
      ``read_input_rows()`` synthesises ``quality_score=0.0`` for every one of them and the
      filter maps anything below 0.40 to ``categ="Trash"`` and skips it. A ``.teitok.xml``
      input therefore enriches ZERO lines today and never reaches
      ``write_document_record()`` at all — a separate defect from D1, found while writing
      these tests and reported rather than fixed here, because what counts as processable is
      a cost/quality decision (it is identical in ``llm_utils.py``'s GPU path). Everything
      downstream of the filter — row reading, the record write, the accretion — is the real
      code under test.
    """

    def _install(client_module):
        def fake_make_chat_fn(*_args, **_kwargs):
            def chat_fn(_messages):
                return json.dumps(
                    {
                        "extracted_keywords_cs": ["kostel"],
                        "extracted_keywords_en": ["church"],
                        "teater_category": "kostel",
                        "confidence_score": 0.9,
                    }
                )

            return chat_fn

        import llm_client_shared

        monkeypatch.setattr(client_module, "make_chat_fn", fake_make_chat_fn)
        monkeypatch.setattr(llm_client_shared, "should_process_line", lambda *a, **k: (True, ""))

    return _install
