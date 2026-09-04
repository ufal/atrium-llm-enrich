"""
tests/test_vocab_build.py — end-to-end regressions for the vocab_build.py pipeline.

These run against the real committed flat harvests (data_samples/vocab/*_flat.json) and
the shipped taxonomy_config.json/taxonomy_overrides.json, offline and in well under a
second (`--from-flat`). That makes them a much stronger check than a synthetic fixture
for the specific bugs they guard: they fail the moment the real data stops producing
the outcome a domain reviewer already signed off on.
"""

from pathlib import Path

import pytest

import vocab_build as vb
import vocab_sources as vs
from vocab_manager import VocabularyManager

REPO_ROOT = Path(__file__).resolve().parent.parent
VOCAB_DIR = REPO_ROOT / "data_samples" / "vocab"
CONFIG = REPO_ROOT / "data_samples" / "taxonomy_config.json"
OVERRIDES = REPO_ROOT / "data_samples" / "taxonomy_overrides.json"


def _shipped_manager():
    return VocabularyManager(config_path=str(CONFIG))


def _built_union():
    """Rebuild the union nesting from the committed flat files, in memory only."""
    if not (VOCAB_DIR / "amcr_flat.json").exists() or not (VOCAB_DIR / "teater_flat.json").exists():
        pytest.skip("flat artifacts not present in this checkout")

    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    filtered = {
        name: vb._filter_excluded(records, manager) for name, (records, _m) in per_source.items()
    }
    qualifiers = manager.qualifier_overrides()
    records = filtered.get("amcr", []) + filtered.get("teater", [])
    nested, audit, collisions = vb._nest(manager, records, qualifiers=qualifiers)
    return nested, audit


# ── B5: exclusion applied before dedup ──────────────────────────────────────────


@pytest.mark.parametrize(
    "cs,expected_theme",
    [
        ("úřední písemnost", "Artefact"),  # TEATER 602 (branch 288, excluded) no longer wins
        ("olej", "Artefact"),  # TEATER 605 (branch 288, excluded) no longer wins
        ("vodní pramen", "Feature"),  # TEATER 1022 (branch 288, excluded) no longer wins
        ("papír", "Material"),  # AMCR HES-000230 (dokument_material, excluded by A2)
    ],
)
def test_b5_regression_terms_survive_exclusion_before_dedup(cs, expected_theme):
    """Finding 3 (comment 5395681950): a term whose *winning* record sat in an excluded
    list used to be dropped entirely, even though a kept list also offered it. All four
    of these were named as concrete casualties in the issue thread."""
    nested, _audit = _built_union()
    assert cs in nested[expected_theme], f"{cs!r} missing from {expected_theme!r}"


def test_papir_would_be_lost_without_the_b5_fix():
    """Guards the fix itself, not just the outcome: prove the naive (post-dedup)
    exclusion order really does lose 'papír', so this test would have caught the bug
    finding 3 described before B5 landed."""
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    unfiltered_union = per_source["amcr"][0] + per_source["teater"][0]
    pairs = vs.to_term_pairs(unfiltered_union)  # dedup with NO exclusion pre-filter
    assert pairs["papír"]["source"] == "amcr"
    assert pairs["papír"]["source_id"] == "HES-000230"  # the list A2 excludes
    assert manager.is_excluded(pairs["papír"]) is True


# ── A4 / B4: per-term facet overrides ────────────────────────────────────────────


@pytest.mark.parametrize(
    "cs,expected_theme,winner_id",
    [
        ("kostel", "Feature", "HES-000021"),
        ("kaple", "Feature", "HES-000020"),
        ("mlýn", "Feature", "HES-000028"),
        ("cesta", "Feature", "HES-000048"),
        ("zahrada", "Activity Area", "HES-000526"),
        ("přírodní útvar", "Finds Context", "HES-000470"),
    ],
)
def test_a4_placement_overrides_take_effect(cs, expected_theme, winner_id):
    nested, audit = _built_union()
    assert nested[expected_theme][cs]["source_id"] == winner_id
    row = next(r for r in audit if r["cs"] == cs)
    assert row["placed_by"] == f"override:amcr:{winner_id}"


@pytest.mark.parametrize(
    "cs,expected_theme,expected_sub",
    [
        ("kostel", "Feature", "druh objektu"),
        ("kaple", "Feature", "druh objektu"),
        ("mlýn", "Feature", "druh objektu"),
        ("cesta", "Feature", "druh objektu"),
        ("zahrada", "Activity Area", "areál aktivity"),
        ("přírodní útvar", "Finds Context", "nálezové okolnosti"),
    ],
)
def test_placement_overrides_also_carry_their_sub_header(cs, expected_theme, expected_sub):
    """The six A4 records were moved by `facet` but kept the sub-header of the list they
    were moved *out* of — `kostel` rendered as "Feature / areál aktivity". Each override
    now names the sub-header too, and the target must already be a header that facet
    uses, so the prompt does not grow a group with one term in it."""
    nested, audit = _built_union()
    assert nested[expected_theme][cs]["sub"] == expected_sub
    assert next(r for r in audit if r["cs"] == cs)["sub"] == expected_sub

    siblings = sum(1 for e in nested[expected_theme].values() if e.get("sub") == expected_sub)
    assert siblings > 1, f"{expected_sub!r} would be a one-term sub-header in {expected_theme!r}"


def test_muzeum_and_pamatkova_pece_subbranches_move_to_location_and_admin():
    """A3/A4: the whole 32+21-term TEATER sub-branches (140/220 and 140/253) move via
    teater_branch_map, not a per-term override — every descendant should follow, not
    just the group's own header term."""
    nested, _audit = _built_union()
    for cs in (
        "muzejní sbírky",
        "archeologická sbírka",
        "archeoskanzen",
        "objekty (památková péče)",
    ):
        assert cs in nested["Location & Admin"], cs


# ── B2/B3: id tracking and qualifier splits on the real data ────────────────────


def test_zamek_offers_both_senses_each_with_its_own_id():
    """Done-criteria (post-review plan §9): 'zámek' offers both senses to the model,
    each carrying its own id."""
    nested, _audit = _built_union()
    lock = nested["Artefact"]["zámek"]
    assert lock["source"] == "amcr" and lock["source_id"] == "HES-000817"
    assert {d["id"] for d in lock["discarded_ids"]} == {"2358"}

    chateau = nested["Activity Area"]["zámek (sídlo elity)"]
    assert chateau["source"] == "teater" and chateau["source_id"] == "1439"
    assert chateau["bare_cs"] == "zámek"


def test_kostel_carries_all_three_ids():
    """Done-criteria: 'kostel' resolves to both AMCR records plus the TEATER one, with
    all three ids attached."""
    nested, _audit = _built_union()
    entry = nested["Feature"]["kostel"]
    all_ids = {entry["source_id"]} | {d["id"] for d in entry["discarded_ids"]}
    assert all_ids == {"HES-000021", "HES-000465", "1333"}


# ── invariants that must survive the rebuild ─────────────────────────────────────


def test_documentation_facet_is_gone_and_udalost_typ_moved():
    m = _shipped_manager()
    assert "Documentation" not in m.themes()
    assert m.settings["heslar_map"]["udalost_typ"] == "Location & Admin"
    nested, _audit = _built_union()
    assert "Documentation" not in nested


def test_no_unplaced_terms_after_the_rebuild():
    nested, _audit = _built_union()
    assert nested.get("Other", {}) == {}


def test_shipped_overrides_file_validates():
    _shipped_manager().validate_settings()  # must not raise


# ── drift guard: the committed artifacts must match a fresh build ───────────────


def test_committed_artifacts_match_a_fresh_build():
    """Config and artifact must move together, or a decision exists in code without
    being in force at runtime — exactly what happened between commit a5e3c8a (config
    edits only) and the follow-up that actually rebuilt data_samples/vocab/. --check
    never writes (see _emit in vocab_build.py), so this only reads the real tree; it
    does not need a tmp_path the way a build test would."""
    if not (VOCAB_DIR / "amcr_flat.json").exists() or not (VOCAB_DIR / "teater_flat.json").exists():
        pytest.skip("flat artifacts not present in this checkout")

    exit_code = vb.main(["--from-flat", "--check"])
    assert exit_code == 0, (
        "data_samples/vocab/*.json and *.csv are stale relative to "
        "taxonomy_config.json/taxonomy_overrides.json — run "
        "`python3 vocab_build.py --from-flat` and commit the result."
    )


# ── the geographic guardrail gate (issue #6, O4 / C1) ───────────────────────────


def test_the_build_refuses_a_vocabulary_that_contradicts_the_prompt():
    """A reinstated geographic branch and a prompt that still forbids selecting one is
    the failure the O3/O4 package warns about — the scores would measure the
    contradiction rather than the model. Caught at build time, before any artifact is
    written, rather than showing up as an unexplained score."""
    manager = _shipped_manager()
    assert vb._check_geo_guardrail(manager) is None  # the shipped pair agrees

    manager.taxonomy["_settings"]["teater_branch_map"]["2560"] = "Location & Admin"
    with pytest.raises(SystemExit, match="teater:2560 is offered to the model"):
        vb._check_geo_guardrail(manager)


def test_the_gate_runs_before_anything_is_written(tmp_path, monkeypatch):
    """`--check` never writes, but a real build does — so the gate has to sit ahead of
    the first _emit, or a contradictory vocabulary reaches disk before anyone is told."""
    if not (VOCAB_DIR / "amcr_flat.json").exists():
        pytest.skip("flat artifacts not present in this checkout")

    import json

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["_settings"]["teater_branch_map"]["2560"] = "Location & Admin"
    del config["_settings"]["_exclusions"]["teater:2560"]
    broken = tmp_path / "taxonomy_config.json"
    broken.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")

    emitted = []
    monkeypatch.setattr(vb, "_emit", lambda path, *a, **k: emitted.append(path))
    with pytest.raises(SystemExit, match="guardrail"):
        vb.main(["--from-flat", "--config", str(broken), "--overrides", str(OVERRIDES)])
    assert not [p for p in emitted if "nested" in p.name]


# ── audit/nested split: sub and same_as_count close the hand-join gap ───────────────


def test_audit_row_carries_the_relabelled_sub():
    """The audit used to have `theme`/`placed_by` but not `sub`, while the nested JSON
    had `sub` but not the placement reason — a reviewer had to join the two files by
    hand to see both. `sub` in the audit must be the *relabelled* sub-header the prompt
    actually renders (via `heslar_labels`), not the raw heslar/scheme name, so it reads
    identically to `nested[theme][cs]["sub"]`."""
    nested, audit = _built_union()
    row = next(r for r in audit if r["cs"] == "kostel")
    assert row["sub"] == nested["Feature"]["kostel"]["sub"]
    assert row["sub"], "sub must be non-empty for a term with a heslar/scheme"


def test_audit_same_as_count_matches_the_attached_links():
    """`same_as_count` cannot be known inside build_nested() — composite links are only
    resolved by attach_same_as() after nesting (they can span two facets or two
    sources). vb._with_same_as_counts() joins the two back together on (theme, cs),
    which is exact because audit and nested are 1:1 by construction."""
    from vocab_manager import attach_same_as

    nested, audit = _built_union()
    same_as_links = attach_same_as(nested)
    assert same_as_links > 0

    enriched = vb._with_same_as_counts(audit, nested)
    assert len(enriched) == len(audit)

    for row in enriched:
        entry = nested[row["theme"]][row["cs"]]
        assert row["same_as_count"] == len(entry.get("same_as") or [])

    # A known composite pair: "most/brod" links to standalone "most".
    most_brod = next(r for r in enriched if r["cs"] == "most/brod")
    assert most_brod["same_as_count"] >= 1


def test_audit_text_defaults_same_as_count_when_absent():
    """A caller that builds an audit list straight from build_nested() and never calls
    _with_same_as_counts() (e.g. a test, or a future report) must still get a valid CSV
    — same_as_count defaults to 0 rather than raising KeyError."""
    row = {
        "cs": "x",
        "en": "y",
        "source": "amcr",
        "source_id": "1",
        "scheme": "s",
        "sub": "",
        "theme": "Feature",
        "placed_by": "heslar:s",
    }
    text = vb._audit_text([row])
    assert text.splitlines()[0] == ",".join(vb._AUDIT_COLUMNS)
    assert text.splitlines()[1].endswith(",0")


def test_shipped_audit_csv_has_sub_and_same_as_count():
    """Integration check against the real committed artifact, not just the in-memory
    rebuild — this is the file a reviewer actually opens."""
    import csv

    if not (VOCAB_DIR / "union_placement_audit.csv").exists():
        pytest.skip("union_placement_audit.csv not present in this checkout")
    with open(VOCAB_DIR / "union_placement_audit.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert set(vb._AUDIT_COLUMNS) <= set(rows[0].keys())
    assert sum(1 for r in rows if int(r["same_as_count"]) > 0) > 0
