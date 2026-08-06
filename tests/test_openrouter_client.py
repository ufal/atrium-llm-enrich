"""
tests/test_openrouter_client.py
================================
Tests for openrouter_client.py, focused on the 2026-07-03 review-pass fix
for finding #1: --attach-as-file was dead code. _build_attachment_content()
existed and was documented but run_document_level() had no way to receive
it, so the document body was always inlined as plain text and the
file-attachment path did nothing regardless of the flag. These tests
exercise _build_attachment_content() directly, then reproduce the exact
per-document closure _make_doc_builder() builds inside main() to confirm
its output is what actually reaches the chat_fn once wired through
llm_client_shared.run_document_level()'s user_content_builder parameter.
"""

import base64
import json

import openrouter_client
from llm_client_shared import run_document_level
from openrouter_client import _build_attachment_content, build_arg_parser

# ── _build_attachment_content ───────────────────────────────────────────────


def test_build_attachment_content_inlines_by_default():
    content = _build_attachment_content("Vyzkum odhalil zaklady.", "sample.md", as_file=False)
    assert content == "DOCUMENT:\nVyzkum odhalil zaklady."


def test_build_attachment_content_as_file_encodes_base64_file_part():
    content = _build_attachment_content("Vyzkum odhalil zaklady.", "sample.md", as_file=True)
    assert isinstance(content, list)
    text_part, file_part = content
    assert text_part == {"type": "text", "text": "DOCUMENT (attached below):"}
    assert file_part["type"] == "file"
    assert file_part["file"]["filename"] == "sample.md"

    data_url = file_part["file"]["file_data"]
    assert data_url.startswith("data:text/markdown;base64,")
    b64_payload = data_url.split(",", 1)[1]
    assert base64.b64decode(b64_payload).decode("utf-8") == "Vyzkum odhalil zaklady."


# ── run_document_level(user_content_builder=...) — finding #1 regression ───


def _fake_empty_items_chat_fn(captured):
    def chat_fn(messages):
        captured.append(messages)
        return json.dumps({"items": []})

    return chat_fn


class _DummyDocModel:
    """Minimal stand-in for build_document_schema()'s DocumentEnrichment —
    only .items is ever read by run_document_level()."""

    items = []

    @classmethod
    def model_validate_json(cls, data):
        return cls()


def test_attach_as_file_reaches_run_document_level_when_wired(tmp_path):
    doc_path = tmp_path / "sample.md"
    doc_path.write_text("Vyzkum odhalil zaklady.", encoding="utf-8")

    captured = []

    def doc_builder(doc_text):
        # exactly the closure _make_doc_builder() returns inside main()
        return _build_attachment_content(doc_text, doc_path.name, True)

    run_document_level(
        doc_path,
        _fake_empty_items_chat_fn(captured),
        "system prompt",
        _DummyDocModel,
        user_content_builder=doc_builder,
    )

    sent_content = captured[0][1]["content"]
    assert isinstance(sent_content, list)  # the file-attachment content part, not inlined text
    assert sent_content[1]["file"]["filename"] == "sample.md"


def test_attach_as_file_flag_defaults_to_false():
    args = build_arg_parser().parse_args([])
    assert args.attach_as_file is False

    args = build_arg_parser().parse_args(["--attach-as-file"])
    assert args.attach_as_file is True


# ── main() end-to-end over a .teitok.xml input — D1 regression ───────────────
#
# atrium-project#10, D1 (P0). `doc_id = f.stem` strips only the LAST extension, so a
# `CTX000000001.teitok.xml` input — one this client's own input filter accepts and its
# docstring advertises — produced the doc_id `CTX000000001.teitok`.
# write_document_record() then looked for a baseline called
# `CTX000000001.teitok.document.json`, which no upstream tool ever writes, so
# DocumentRecord.open() fell back to rule 3 and DISCARDED every upstream block — pages,
# lines, entities, translations — emitting an orphan record under an id nothing else in the
# pipeline uses. Nothing caught it: this file referenced neither document_json, nor doc_id,
# nor .teitok.xml, and the E2E smoke feeds a single-dot `.csv` whose `.stem` happens to be
# right. The test below runs the real main() against a pre-seeded baseline, which is the
# only shape that reproduces it.


def _run_main(env, extra=()):
    openrouter_client.main(
        [
            "--config",
            str(env.config),
            "--input",
            str(env.teitok),
            "--output-dir",
            str(env.output_dir),
            "--model",
            "test/model",
            "--api-key",
            "test-key",
            *extra,
        ]
    )


def test_teitok_run_preserves_every_upstream_block(remote_client_env, seeded_baseline, stub_llm):
    """The whole point of the accretion contract: llm-enrich adds `enrichment` and touches
    nothing else. Before the fix this assertion failed on all three blocks at once."""
    from atrium_document import load_document

    env = remote_client_env
    stub_llm(openrouter_client)

    _run_main(env, ["--document-json-dir", str(env.record_dir)])

    assert env.baseline.exists(), (
        f"no record at the canonical doc_id; dir holds "
        f"{sorted(p.name for p in env.record_dir.iterdir())}"
    )
    assert not env.forked_record.exists(), "record written under the forked `.teitok` doc_id"

    record = load_document(str(env.baseline))
    assert record["doc_id"] == env.doc_id
    assert record["assembled"]["had_baseline"] is True
    for block in ("pages", "lines", "entities"):
        assert block in record, f"upstream {block!r} block was discarded"
    assert record["lines"][0]["text"].startswith("Výzkum odhalil")
    assert record["entities"][0]["surface"] == "gotického kostela"
    assert record["source"]["origin"] == "ABBYY-ALTO"
    # …and llm-enrich's own contribution really is there, so the test is not passing by
    # having skipped the write altogether.
    assert record["enrichment"]["items"][0]["extracted_keywords_en"] == ["church"]
    assert record["assembled"]["blocks"]["enrichment"]["program"] == "llm-enrich"


def test_teitok_run_names_its_enriched_output_by_the_canonical_doc_id(
    remote_client_env, seeded_baseline, stub_llm
):
    """The same derivation names `<doc_id>_enriched.json`, so the fix is visible in the
    CLI's own output too — `CTX000000001.teitok_enriched.json` was the old name."""
    env = remote_client_env
    stub_llm(openrouter_client)

    _run_main(env, ["--document-json-dir", str(env.record_dir)])

    assert (env.output_dir / f"{env.doc_id}_enriched.json").exists()


def test_document_json_single_file_pair_round_trips_the_baseline(
    remote_client_env, seeded_baseline, stub_llm
):
    """The --document-json/--document-json-out convenience pair (issue #13) copies the
    baseline into a scratch dir under the derived doc_id, so it forks on exactly the same
    name and had to be fixed with the loop."""
    from atrium_document import load_document

    env = remote_client_env
    stub_llm(openrouter_client)
    out_path = env.root / "5_llm.json"

    _run_main(
        env,
        ["--document-json", str(seeded_baseline), "--document-json-out", str(out_path)],
    )

    assert out_path.exists()
    record = load_document(str(out_path))
    assert record["doc_id"] == env.doc_id
    for block in ("pages", "lines", "entities"):
        assert block in record, f"upstream {block!r} block was discarded"
