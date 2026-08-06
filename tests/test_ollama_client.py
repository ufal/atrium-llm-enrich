"""
tests/test_ollama_client.py
============================
Tests for ollama_client.py, focused on the 2026-07-03 review-pass fix for
finding #2: ensure_model_pulled()'s bare-name tag matching over-matched —
it treated e.g. 'qwen2.5:14b' as satisfying a request for 'qwen2.5:7b',
skipping the pull and then running inference against a tag Ollama never
had (and could crash outright on a model entry missing its 'name' key).
The fix requires an EXACT tag match, with a bare (untagged) request
additionally satisfied only by '<name>:latest'.
"""

import json
from unittest.mock import MagicMock

import pytest
import requests

import ollama_client
import openrouter_client
from ollama_client import build_arg_parser, ensure_model_pulled

# ── ensure_model_pulled ──────────────────────────────────────────────────────


def _session_with_tags(names):
    """A MagicMock stand-in for requests.Session whose GET /api/tags returns
    the given model names."""
    session = MagicMock()
    tags_resp = MagicMock()
    tags_resp.raise_for_status.return_value = None
    tags_resp.json.return_value = {"models": [{"name": n} for n in names]}
    session.get.return_value = tags_resp
    return session


def test_ensure_model_pulled_exact_tag_present_skips_pull():
    session = _session_with_tags(["qwen2.5:7b"])
    ensure_model_pulled("http://x", "qwen2.5:7b", session, timeout=10)
    session.post.assert_not_called()


def test_ensure_model_pulled_bare_name_satisfied_by_latest():
    session = _session_with_tags(["qwen2.5:latest"])
    ensure_model_pulled("http://x", "qwen2.5", session, timeout=10)
    session.post.assert_not_called()


def test_ensure_model_pulled_different_tag_of_same_family_still_pulls():
    # THE bug this fixes: qwen2.5:14b being installed must NOT satisfy a
    # request for qwen2.5:7b — the old bare-name-family match skipped the
    # pull here and then ran inference against a tag Ollama didn't have.
    session = _session_with_tags(["qwen2.5:14b"])
    session.post.return_value.__enter__.return_value = MagicMock(
        iter_lines=lambda: [], raise_for_status=lambda: None
    )
    ensure_model_pulled("http://x", "qwen2.5:7b", session, timeout=10)
    session.post.assert_called_once()
    assert session.post.call_args.kwargs["json"] == {"name": "qwen2.5:7b"}


def test_ensure_model_pulled_missing_model_streams_pull_progress(capsys):
    session = _session_with_tags([])
    pull_resp = MagicMock()
    pull_resp.raise_for_status.return_value = None
    pull_resp.iter_lines.return_value = [
        json.dumps({"status": "pulling manifest"}).encode(),
        json.dumps({"status": "success"}).encode(),
    ]
    session.post.return_value.__enter__.return_value = pull_resp

    ensure_model_pulled("http://x", "qwen2.5:7b", session, timeout=10)

    out = capsys.readouterr().out
    assert "pulling manifest" in out
    assert "success" in out
    assert "ready" in out


def test_ensure_model_pulled_ignores_tags_entries_without_a_name():
    # A registry row with no 'name' key (e.g. a manifest-only entry) must not
    # crash the set comprehension, and must not itself count as a match.
    session = MagicMock()
    tags_resp = MagicMock()
    tags_resp.raise_for_status.return_value = None
    tags_resp.json.return_value = {"models": [{"digest": "sha256:abc"}, {"name": "qwen2.5:7b"}]}
    session.get.return_value = tags_resp

    ensure_model_pulled("http://x", "qwen2.5:7b", session, timeout=10)
    session.post.assert_not_called()


def test_ensure_model_pulled_unreachable_host_raises_runtime_error():
    session = MagicMock()
    session.get.side_effect = requests.exceptions.ConnectionError("refused")

    with pytest.raises(RuntimeError, match="Could not reach Ollama"):
        ensure_model_pulled("http://x", "qwen2.5:7b", session, timeout=10)


# ── document-json flag parity with openrouter_client — J5 ────────────────────


def _document_json_flags(parser):
    """The document-record options a parser exposes, read through its public surface."""
    return {name for name in vars(parser.parse_args([])) if name.startswith("document_json")}


def test_document_json_flags_match_the_openrouter_client():
    """atrium-project#10, J5: this parser had only --document-json-dir while
    openrouter_client had the --document-json/--document-json-out single-file pair too — an
    incomplete port of the issue #13 feature, not a decision (the two clients mirror each
    other line-for-line otherwise). Comparing the two parsers, rather than asserting a
    hard-coded list, is what makes the NEXT one-sided addition fail here."""
    assert _document_json_flags(build_arg_parser()) == _document_json_flags(
        openrouter_client.build_arg_parser()
    ) == {"document_json_dir", "document_json", "document_json_out"}


def test_document_json_flags_default_to_none_and_parse_as_paths():
    from pathlib import Path

    assert build_arg_parser().parse_args([]).document_json is None
    assert build_arg_parser().parse_args([]).document_json_out is None

    args = build_arg_parser().parse_args(
        ["--document-json", "in.json", "--document-json-out", "out.json"]
    )
    assert args.document_json == Path("in.json")
    assert args.document_json_out == Path("out.json")


# ── main() end-to-end over a .teitok.xml input — D1 regression ───────────────
#
# The same P0 as openrouter_client's (atrium-project#10, D1): `doc_id = f.stem` on an
# accepted `CTX000000001.teitok.xml` input yielded `CTX000000001.teitok`, so
# write_document_record() missed the upstream baseline entirely and emitted an orphan with
# every upstream block discarded. Kept parallel to tests/test_openrouter_client.py, because
# the two clients are maintained as line-for-line mirrors and a fix applied to only one of
# them is exactly how this defect arose.


def _run_main(env, extra=()):
    ollama_client.main(
        [
            "--config",
            str(env.config),
            "--input",
            str(env.teitok),
            "--output-dir",
            str(env.output_dir),
            "--model",
            "qwen2.5:7b",
            "--skip-pull-check",
            *extra,
        ]
    )


def test_teitok_run_preserves_every_upstream_block(remote_client_env, seeded_baseline, stub_llm):
    """llm-enrich adds `enrichment` and touches nothing else. Before the fix this failed on
    pages, lines and entities at once."""
    from atrium_document import load_document

    env = remote_client_env
    stub_llm(ollama_client)

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
    assert record["enrichment"]["items"][0]["extracted_keywords_en"] == ["church"]
    assert record["assembled"]["blocks"]["enrichment"]["program"] == "llm-enrich"


def test_document_json_single_file_pair_round_trips_the_baseline(
    remote_client_env, seeded_baseline, stub_llm
):
    """The ported pair (J5) end-to-end: baseline in, updated record at the exact path
    requested — the shape the hub's e2e-pipeline-smoke chains stage to stage."""
    from atrium_document import load_document

    env = remote_client_env
    stub_llm(ollama_client)
    out_path = env.root / "5_llm.json"

    _run_main(
        env,
        ["--document-json", str(seeded_baseline), "--document-json-out", str(out_path)],
    )

    assert out_path.exists()
    record = load_document(str(out_path))
    assert record["doc_id"] == env.doc_id
    assert record["enrichment"]["items"][0]["teater_category"] == "kostel"
    for block in ("pages", "lines", "entities"):
        assert block in record, f"upstream {block!r} block was discarded"


def test_document_json_out_without_a_record_says_so_and_writes_nothing(
    remote_client_env, stub_llm, capsys
):
    """Degrade-gracefully, but audibly (the J4 concern, ported with the flags): when no
    record was produced the promised file must not silently appear empty."""
    env = remote_client_env
    stub_llm(ollama_client)
    # No results -> no record -> nothing to copy out.
    empty_teitok = env.root / "empty.teitok.xml"
    empty_teitok.write_text("<teiCorpus><text/></teiCorpus>", encoding="utf-8")
    out_path = env.root / "5_llm.json"

    ollama_client.main(
        [
            "--config",
            str(env.config),
            "--input",
            str(empty_teitok),
            "--output-dir",
            str(env.output_dir),
            "--model",
            "qwen2.5:7b",
            "--skip-pull-check",
            "--document-json-out",
            str(out_path),
        ]
    )

    assert not out_path.exists()
    assert "was NOT written" in capsys.readouterr().err
