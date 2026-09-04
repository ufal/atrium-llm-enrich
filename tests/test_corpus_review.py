"""
tests/test_corpus_review.py — unit and end-to-end tests for corpus_review.py.

The end-to-end tests run against the real committed synthetic corpus
(data_samples/{UDP,DOC_LINE_CATEG}/CTX00000000{1,2,3}) precisely BECAUSE it is
synthetic and tiny: this pins the honest 4-hit ceiling the module docstring
describes, so nobody mistakes a later change to the matcher for more "evidence"
than three placeholder documents can actually provide.
"""

from pathlib import Path

import pytest

import corpus_review as cr
import vocab_build as vb
import vocab_review as vr
from vocab_manager import VocabularyManager

REPO_ROOT = Path(__file__).resolve().parent.parent
VOCAB_DIR = REPO_ROOT / "data_samples" / "vocab"
CONFIG = REPO_ROOT / "data_samples" / "taxonomy_config.json"
UDP_DIR = REPO_ROOT / "data_samples" / "UDP"
LINES_DIR = REPO_ROOT / "data_samples" / "DOC_LINE_CATEG"


def _shipped_manager():
    return VocabularyManager(config_path=str(CONFIG))


def _require_flat():
    if not (VOCAB_DIR / "amcr_flat.json").exists() or not (VOCAB_DIR / "teater_flat.json").exists():
        pytest.skip("flat artifacts not present in this checkout")


def _require_corpus():
    if not UDP_DIR.exists() or not LINES_DIR.exists():
        pytest.skip("synthetic corpus (data_samples/UDP, DOC_LINE_CATEG) not present")


# ── read_conllu_tokens ────────────────────────────────────────────────────────────


def test_read_conllu_tokens_skips_multiword_and_empty_nodes(tmp_path):
    """UD convention: an id containing '-' regroups already-listed single tokens
    (e.g. Czech "na+dně" contractions never occur, but the convention is universal),
    and an id containing '.' is an empty node. Both must be skipped or a word count
    would double."""
    conllu = (
        "# text = test\n"
        "1-2\tdo\t_\t_\t_\t_\t_\t_\t_\t_\n"
        "1\tdo\tdo\tADP\t_\t_\t_\t_\t_\t_\n"
        "2\tokna\tokno\tNOUN\t_\t_\t_\t_\t_\t_\n"
        "2.1\telided\telidet\tVERB\t_\t_\t_\t_\t_\t_\n"
    )
    p = tmp_path / "x.conllu"
    p.write_text(conllu, encoding="utf-8")
    tokens = cr.read_conllu_tokens(p)
    assert [t.form for t in tokens] == ["do", "okna"]
    assert [t.lemma for t in tokens] == ["do", "okno"]


def test_read_conllu_tokens_skips_comments_and_blank_lines(tmp_path):
    p = tmp_path / "x.conllu"
    p.write_text("# generator = x\n\n1\tx\ty\tNOUN\t_\t_\t_\t_\t_\t_\n", encoding="utf-8")
    tokens = cr.read_conllu_tokens(p)
    assert len(tokens) == 1


# ── _form_lemma_map / _line_lemmas ────────────────────────────────────────────────


def test_form_lemma_map_is_casefolded_and_picks_most_common():
    tokens = [
        cr.ConlluToken(form="Keramika", lemma="keramika", upos="NOUN"),
        cr.ConlluToken(form="keramiky", lemma="keramika", upos="NOUN"),
        cr.ConlluToken(form="X", lemma="a", upos="NOUN"),
        cr.ConlluToken(form="X", lemma="a", upos="NOUN"),
        cr.ConlluToken(form="X", lemma="b", upos="NOUN"),
    ]
    m = cr._form_lemma_map(tokens)
    assert m["keramika"] == "keramika"
    assert m["keramiky"] == "keramika"
    assert m["x"] == "a"  # 2 votes for "a" vs 1 for "b"


def test_form_lemma_map_skips_underscore_lemmas():
    tokens = [cr.ConlluToken(form="x", lemma="_", upos="X")]
    assert cr._form_lemma_map(tokens) == {}


def test_line_lemmas_matches_inflected_forms_via_the_document_map():
    """The concrete case the module exists for: 'středověku' (genitive) never equals
    the vocabulary's dictionary-form label 'středověk' as a substring."""
    form_lemma = {"středověku": "středověk", "raného": "raný"}
    lemmas = cr._line_lemmas("z raného středověku.", form_lemma)
    assert "středověk" in lemmas
    assert "raný" in lemmas


def test_line_lemmas_falls_back_to_surface_word_when_unseen():
    lemmas = cr._line_lemmas("neznámé slovo", {})
    assert lemmas == ["neznámé", "slovo"]


def test_line_lemmas_ignores_digits_and_punctuation():
    lemmas = cr._line_lemmas("A123/2024!", {})
    assert lemmas == ["a"]


# ── _single_word_terms ────────────────────────────────────────────────────────────


def test_single_word_terms_excludes_multi_word_labels():
    nested = {
        "Feature": {
            "kostel": {"en": "church", "source": "amcr", "source_id": "1"},
            "zlomek keramiky": {"en": "ceramic sherd", "source": "amcr", "source_id": "2"},
        }
    }
    idx = cr._single_word_terms(nested)
    assert "kostel" in idx
    assert "zlomek keramiky" not in idx
    assert not any(len(k.split()) > 1 for k in idx)


def test_single_word_terms_uses_bare_cs_for_a_qualified_entry():
    """A bracketed homonym split (B3) is single-word once its bare label is used --
    'zámek (sídlo elity)' itself is not, but its bare_cs 'zámek' is."""
    nested = {
        "Activity Area": {
            "zámek (sídlo elity)": {
                "en": "châteaux",
                "source": "teater",
                "source_id": "1439",
                "bare_cs": "zámek",
            }
        }
    }
    idx = cr._single_word_terms(nested)
    assert "zámek" in idx  # norm_label casefolds but does not strip diacritics
    entry = idx["zámek"][0]
    assert entry["cs"] == "zámek (sídlo elity)"  # the enum key stays as offered


# ── end-to-end against the real committed synthetic corpus ──────────────────────


def test_corpus_term_evidence_finds_the_known_synthetic_hits():
    """The four terms the three synthetic demo documents are known to contain must
    always be found -- but as a SUBSET, never an exact set.

    This test previously asserted equality against exactly those four, which pinned it
    to a corpus of exactly three placeholder documents. That was never a safe
    assumption for a test reading live `data_samples/` content: the whole purpose of
    this module is to be run once the real reports land, at which point dozens of
    genuine hits (zámek, bronz, klášter, jehlice, ...) appear and an equality
    assertion fails on success. Everything asserted here now holds at any corpus
    size; the per-term doc_count floors are the demo documents' own contribution,
    which more documents can only increase.
    """
    _require_flat()
    _require_corpus()
    manager = _shipped_manager()
    nested = cr.shipped_nested(VOCAB_DIR, manager)
    rows, stats = cr.corpus_term_evidence_rows(nested, UDP_DIR, LINES_DIR)

    hits = {r["cs"]: r for r in rows if r["occurrences"] > 0}

    demo_docs = {"CTX000000001", "CTX000000002", "CTX000000003"}
    present = {p.stem for p in UDP_DIR.glob("*.conllu")}
    if demo_docs <= present:
        assert {"hradiště", "sonda", "keramika", "středověk"} <= set(hits)
        assert hits["hradiště"]["doc_count"] >= 2
        assert hits["sonda"]["doc_count"] >= 2
        assert hits["keramika"]["doc_count"] >= 1
        assert hits["středověk"]["doc_count"] >= 1

    # Structural invariants, true of any corpus.
    assert stats["documents"] == len(list(UDP_DIR.glob("*.conllu")))
    assert stats["terms_with_hits"] == len(hits)
    for row in hits.values():
        assert 1 <= row["doc_count"] <= stats["documents"]
        assert row["occurrences"] >= row["doc_count"]


def test_corpus_term_evidence_example_lines_cite_the_real_line():
    _require_flat()
    _require_corpus()
    manager = _shipped_manager()
    nested = cr.shipped_nested(VOCAB_DIR, manager)
    rows, _stats = cr.corpus_term_evidence_rows(nested, UDP_DIR, LINES_DIR)
    keramika = next(r for r in rows if r["cs"] == "keramika")
    assert "Nalezená keramika" in keramika["example_lines"]


def test_corpus_branch_evidence_covers_the_same_rules_as_exclusion_impact():
    """The corpus roll-up must never disagree with the other O3/O4 sheets about
    which rules exist -- all built from the same _excluded_buckets()."""
    _require_flat()
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    exclusion_rules = {r["rule"] for r in vr.exclusion_impact_rows(per_source, manager)}
    branch_rules = {r["rule"] for r in cr.corpus_branch_evidence_rows(per_source, manager, UDP_DIR)}
    assert exclusion_rules == branch_rules


def test_corpus_branch_evidence_finds_stredovek_under_vyskovy_bod_typ():
    """vyskovy_bod_typ (an AMCR 'elevation reference point' list, excluded under M4)
    happens to also carry a record labelled 'středověk' -- the same label the OFFERED
    Chronology term uses. This is a real, if surprising, cross-list label collision
    the corpus can surface: 'středověk' occurring in text says nothing about which of
    the two records a model meant, only that the LABEL is attested in real usage."""
    _require_flat()
    _require_corpus()
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    nested = cr.shipped_nested(VOCAB_DIR, manager)
    rows = cr.corpus_branch_evidence_rows(per_source, manager, UDP_DIR, nested=nested)
    row = next(r for r in rows if r["rule"] == "heslar:vyskovy_bod_typ")
    # Floors, not exact counts: the claim under test is that the collision is
    # DETECTED, and 'středověk' occurs more often the more real reports are present.
    assert row["occurrences"] >= 1
    assert row["doc_count"] >= 1
    # 'středověk' is ALSO an offered Chronology term, so it is shared, not unique --
    # this excluded list earns no evidence from it.
    assert "středověk" in row["sample_shared_terms"]
    assert row["shared_occurrences"] >= 1


# ── gold_workbook_rows ────────────────────────────────────────────────────────────


def test_gold_workbook_covers_every_committed_line():
    """One workbook row per DOC_LINE_CATEG data row -- no line silently dropped, none
    duplicated. The expected count is computed from the files actually on disk rather
    than hard-coded: this used to assert 16 (the three demo documents' rows), which
    broke the moment the real reports were present locally and the true answer became
    2172. The invariant is the 1:1 mapping, not any particular corpus size."""
    _require_flat()
    _require_corpus()
    manager = _shipped_manager()
    nested = cr.shipped_nested(VOCAB_DIR, manager)
    rows = cr.gold_workbook_rows(nested, LINES_DIR, UDP_DIR)

    import csv

    expected = 0
    for path in sorted(LINES_DIR.glob("*.csv")):
        with open(path, "r", encoding="utf-8", newline="") as fh:
            expected += sum(1 for _ in csv.DictReader(fh))
    assert len(rows) == expected
    assert expected > 0, "no DOC_LINE_CATEG rows found — corpus fixture is empty"


def test_gold_workbook_gold_columns_start_blank():
    _require_flat()
    _require_corpus()
    manager = _shipped_manager()
    nested = cr.shipped_nested(VOCAB_DIR, manager)
    rows = cr.gold_workbook_rows(nested, LINES_DIR, UDP_DIR)
    for row in rows:
        assert row["gold_category"] == ""
        assert row["gold_keywords_cs"] == ""
        assert row["model_category"] == ""
        assert row["model_keywords_cs"] == ""


def test_gold_workbook_candidate_terms_on_the_known_line():
    _require_flat()
    _require_corpus()
    manager = _shipped_manager()
    nested = cr.shipped_nested(VOCAB_DIR, manager)
    rows = cr.gold_workbook_rows(nested, LINES_DIR, UDP_DIR)
    row = next(
        r for r in rows if r["file_id"] == "CTX000000001" and r["page"] == "2" and r["line"] == "1"
    )
    candidates = row["candidate_terms"].split("; ")
    assert "keramika" in candidates
    assert "středověk" in candidates


def test_gold_workbook_is_sorted_by_file_then_page_then_line():
    _require_flat()
    _require_corpus()
    manager = _shipped_manager()
    nested = cr.shipped_nested(VOCAB_DIR, manager)
    rows = cr.gold_workbook_rows(nested, LINES_DIR, UDP_DIR)
    keys = [(r["file_id"], int(r["page"]), int(r["line"])) for r in rows]
    assert keys == sorted(keys)


# ── CLI smoke test ────────────────────────────────────────────────────────────────


def test_main_all_writes_three_csvs(tmp_path):
    """Writes into --vocab-dir/--udp-dir/--lines-dir copies, never the real
    data_samples/ tree, during a test."""
    _require_flat()
    _require_corpus()
    import shutil

    vocab_dest = tmp_path / "vocab"
    vocab_dest.mkdir()
    for name in ("amcr_flat.json", "teater_flat.json"):
        shutil.copy(VOCAB_DIR / name, vocab_dest / name)

    udp_dest = tmp_path / "UDP"
    shutil.copytree(UDP_DIR, udp_dest)
    lines_dest = tmp_path / "DOC_LINE_CATEG"
    shutil.copytree(LINES_DIR, lines_dest)

    rc = cr.main(
        [
            "--all",
            "--vocab-dir",
            str(vocab_dest),
            "--config",
            str(CONFIG),
            "--udp-dir",
            str(udp_dest),
            "--lines-dir",
            str(lines_dest),
        ]
    )
    assert rc == 0
    assert (vocab_dest / "corpus_term_evidence.csv").exists()
    assert (vocab_dest / "corpus_branch_evidence.csv").exists()
    assert (vocab_dest / "gold_workbook.csv").exists()


def test_main_with_no_flags_is_a_usage_error():
    assert cr.main([]) == 2


def test_main_handles_missing_corpus_directories_gracefully(tmp_path, capsys):
    """The generator must degrade to zero occurrences, not crash, when the real
    corpus is absent -- which is the state of this repository today."""
    _require_flat()
    import shutil

    vocab_dest = tmp_path / "vocab"
    vocab_dest.mkdir()
    for name in ("amcr_flat.json", "teater_flat.json"):
        shutil.copy(VOCAB_DIR / name, vocab_dest / name)

    rc = cr.main(
        [
            "--term-evidence",
            "--vocab-dir",
            str(vocab_dest),
            "--config",
            str(CONFIG),
            "--udp-dir",
            str(tmp_path / "no-such-udp-dir"),
            "--lines-dir",
            str(tmp_path / "no-such-lines-dir"),
        ]
    )
    assert rc == 0
    import csv

    rows = list(csv.DictReader(open(vocab_dest / "corpus_term_evidence.csv", encoding="utf-8")))
    assert all(int(r["occurrences"]) == 0 for r in rows)


# ── the unique/shared split (the heslar:zeme "malta" trap) ──────────────────────


def _malta_corpus(tmp_path):
    """The three real corpus lines where 'malta' means MORTAR, with a CoNLL-U that
    lemmatises maltu/malty/maltou -> malta the way UDPipe does."""
    udp, lines = tmp_path / "UDP", tmp_path / "LINES"
    udp.mkdir()
    lines.mkdir()
    real = [
        (
            "CTX198402945",
            "1",
            "26",
            "z lomového zdivá nasucho a jen místy na maltu, ve spodní části",
        ),
        ("CTX199603106", "3", "3", "charakteru s příměsí malty a úlomků opuky a cihel"),
        (
            "CTX199603106",
            "3",
            "6",
            "nevelkých rozměrů, spojovaných vápennou maltou z hrubého říčního",
        ),
    ]
    import csv as _csv

    by_doc = {}
    for f, pg, ln, txt in real:
        by_doc.setdefault(f, []).append((pg, ln, txt))
    for doc, rows in by_doc.items():
        with open(lines / f"{doc}.csv", "w", encoding="utf-8", newline="") as fh:
            w = _csv.DictWriter(fh, fieldnames=["file", "page_num", "line_num", "text", "categ"])
            w.writeheader()
            for pg, ln, txt in rows:
                w.writerow(
                    {"file": doc, "page_num": pg, "line_num": ln, "text": txt, "categ": "Clear"}
                )
        with open(udp / f"{doc}.conllu", "w", encoding="utf-8") as fh:
            for _pg, _ln, txt in rows:
                fh.write(f"# text = {txt}\n")
                for i, word in enumerate(txt.replace(",", " ").split(), start=1):
                    bare = word.strip(",.").lower()
                    lemma = "malta" if bare in ("maltu", "malty", "maltou") else bare
                    fh.write(f"{i}\t{word}\t{lemma}\tNOUN\t_\t_\t_\t_\t_\t_\n")
                fh.write("\n")
    return udp, lines


def test_branch_evidence_does_not_credit_zeme_for_the_mortar_homograph(tmp_path):
    """The single most consequential row in this sheet. `malta` is MORTAR in these
    lines (an offered Material term: AMCR HES-000910/HES-000992, TEATER 2499), but
    norm_label's casefold makes it identical to the country `Malta` (HES-001366) in
    the excluded `zeme` list. Credited to zeme, it reads as evidence that country
    names appear in archaeological reports -- on exactly the branch O3/O4 Q1 and the
    geographic guardrail turn on. It must land in shared, never unique."""
    _require_flat()
    udp, lines = _malta_corpus(tmp_path)
    manager = _shipped_manager()
    nested = cr.shipped_nested(VOCAB_DIR, manager)
    per_source = vb._load_flat(VOCAB_DIR)

    rows = cr.corpus_branch_evidence_rows(per_source, manager, udp, nested=nested)
    zeme = next(r for r in rows if r["rule"] == "heslar:zeme")

    assert zeme["occurrences"] == 3  # the raw signal is real...
    assert zeme["unique_occurrences"] == 0  # ...but none of it is zeme's own
    assert zeme["unique_hit_term_count"] == 0
    assert zeme["shared_occurrences"] == 3
    assert "malta" in zeme["sample_shared_terms"]
    assert zeme["sample_unique_terms"] == ""


def test_branch_evidence_sorts_by_unique_not_total_occurrences(tmp_path):
    """A branch whose every hit is explained by an already-offered label must not
    outrank one with genuine unique evidence, however large its raw total."""
    _require_flat()
    udp, lines = _malta_corpus(tmp_path)
    manager = _shipped_manager()
    nested = cr.shipped_nested(VOCAB_DIR, manager)
    per_source = vb._load_flat(VOCAB_DIR)
    rows = cr.corpus_branch_evidence_rows(per_source, manager, udp, nested=nested)

    uniq = [r["unique_occurrences"] for r in rows]
    assert uniq == sorted(uniq, reverse=True)


def test_branch_evidence_without_nested_reports_everything_as_unique(tmp_path):
    """Documented fallback: with no offered vocabulary to compare against, nothing
    CAN be classified as shared, so callers wanting the split must pass nested."""
    _require_flat()
    udp, _lines = _malta_corpus(tmp_path)
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    rows = cr.corpus_branch_evidence_rows(per_source, manager, udp)
    zeme = next(r for r in rows if r["rule"] == "heslar:zeme")
    assert zeme["shared_hit_term_count"] == 0
    assert zeme["unique_occurrences"] == zeme["occurrences"] == 3


def test_branch_evidence_unique_plus_shared_reconciles_with_the_total(tmp_path):
    _require_flat()
    udp, _lines = _malta_corpus(tmp_path)
    manager = _shipped_manager()
    nested = cr.shipped_nested(VOCAB_DIR, manager)
    per_source = vb._load_flat(VOCAB_DIR)
    for row in cr.corpus_branch_evidence_rows(per_source, manager, udp, nested=nested):
        assert row["unique_hit_term_count"] + row["shared_hit_term_count"] == row["hit_term_count"]
        assert row["unique_occurrences"] + row["shared_occurrences"] == row["occurrences"]


# ── the overwrite guard ─────────────────────────────────────────────────────────


def _corpus_doc_count():
    """However many documents this checkout actually has — 3 on a clean clone, 19 with
    the real reports restored. Every guard assertion below is derived from this rather
    than hardcoded: the guard's behaviour depends on the RELATIVE size of the recorded
    corpus vs the one on disk, never on an absolute number."""
    return len(list(UDP_DIR.glob("*.conllu")))


def _vocab_copy(tmp_path):
    import shutil

    dest = tmp_path / "vocab"
    dest.mkdir()
    for name in ("amcr_flat.json", "teater_flat.json"):
        shutil.copy(VOCAB_DIR / name, dest / name)
    return dest


def test_refuses_to_overwrite_sheets_built_from_a_larger_corpus(tmp_path):
    """The real reports are untracked (issue #19 attachment), so a clean checkout has
    only the three demo documents. Running the generator there used to silently
    replace real evidence with placeholder numbers."""
    _require_flat()
    _require_corpus()
    import json

    dest = _vocab_copy(tmp_path)
    bigger = _corpus_doc_count() + 5
    (dest / "corpus_review.meta.json").write_text(
        json.dumps({"documents": bigger, "total_tokens": 999999, "distinct_lemmas": 99999}) + "\n",
        encoding="utf-8",
    )
    (dest / "corpus_term_evidence.csv").write_text("REAL EVIDENCE\n", encoding="utf-8")

    rc = cr.main(
        [
            "--all",
            "--vocab-dir",
            str(dest),
            "--config",
            str(CONFIG),
            "--udp-dir",
            str(UDP_DIR),
            "--lines-dir",
            str(LINES_DIR),
        ]
    )
    assert rc == 1
    assert (dest / "corpus_term_evidence.csv").read_text(encoding="utf-8") == "REAL EVIDENCE\n"


def test_force_overrides_the_guard(tmp_path):
    _require_flat()
    _require_corpus()
    import json

    dest = _vocab_copy(tmp_path)
    (dest / "corpus_review.meta.json").write_text(
        json.dumps({"documents": _corpus_doc_count() + 5}) + "\n", encoding="utf-8"
    )
    (dest / "corpus_term_evidence.csv").write_text("REAL EVIDENCE\n", encoding="utf-8")

    rc = cr.main(
        [
            "--all",
            "--force",
            "--vocab-dir",
            str(dest),
            "--config",
            str(CONFIG),
            "--udp-dir",
            str(UDP_DIR),
            "--lines-dir",
            str(LINES_DIR),
        ]
    )
    assert rc == 0
    assert (dest / "corpus_term_evidence.csv").read_text(encoding="utf-8").startswith("cs,en,")
    meta = json.loads((dest / "corpus_review.meta.json").read_text(encoding="utf-8"))
    assert meta["documents"] == _corpus_doc_count()


def test_a_same_size_or_larger_corpus_is_never_refused(tmp_path):
    _require_flat()
    _require_corpus()
    import json

    dest = _vocab_copy(tmp_path)
    (dest / "corpus_review.meta.json").write_text(
        json.dumps({"documents": _corpus_doc_count()}) + "\n", encoding="utf-8"
    )
    rc = cr.main(
        [
            "--all",
            "--vocab-dir",
            str(dest),
            "--config",
            str(CONFIG),
            "--udp-dir",
            str(UDP_DIR),
            "--lines-dir",
            str(LINES_DIR),
        ]
    )
    assert rc == 0


def test_corpus_meta_records_provenance_and_merges_sheet_list(tmp_path):
    """Running one report must not drop the provenance of sheets an earlier run
    wrote from the same corpus."""
    _require_flat()
    _require_corpus()
    import json

    dest = _vocab_copy(tmp_path)
    common = [
        "--vocab-dir",
        str(dest),
        "--config",
        str(CONFIG),
        "--udp-dir",
        str(UDP_DIR),
        "--lines-dir",
        str(LINES_DIR),
    ]
    assert cr.main(["--gold-workbook", *common]) == 0
    assert cr.main(["--branch-evidence", *common]) == 0

    meta = json.loads((dest / "corpus_review.meta.json").read_text(encoding="utf-8"))
    assert meta["documents"] == _corpus_doc_count()
    assert set(meta["sheets"]) == {"gold_workbook.csv", "corpus_branch_evidence.csv"}


def test_corrupt_meta_sidecar_does_not_block_a_rebuild(tmp_path):
    _require_flat()
    _require_corpus()
    dest = _vocab_copy(tmp_path)
    (dest / "corpus_review.meta.json").write_text("{not json", encoding="utf-8")
    rc = cr.main(
        [
            "--term-evidence",
            "--vocab-dir",
            str(dest),
            "--config",
            str(CONFIG),
            "--udp-dir",
            str(UDP_DIR),
            "--lines-dir",
            str(LINES_DIR),
        ]
    )
    assert rc == 0
