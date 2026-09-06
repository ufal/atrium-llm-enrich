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


# ── M11/M12: the reinstatement, and the placements it must not undo ─────────────


@pytest.mark.parametrize(
    "cs,facet,ids",
    [
        # HES-000230 is dokument_material, excluded under M4 — B5's point is that papír
        # survives anyway, via the kept objekt_specifikace record.
        ("papír", "Material", {"HES-000917", "HES-000999", "2529"}),
        ("olej", "Artefact", {"605", "2424"}),
        ("vodní pramen", "Feature", {"1022", "1685"}),
        ("úřední písemnost", "Artefact", {"602", "2250"}),
    ],
)
def test_the_b5_terms_keep_their_reviewed_facet_after_reinstatement(cs, facet, ids):
    """The four terms the exclusion-before-dedup fix rescued (comment 5395681950).

    Reinstating TEATER 288 (M11 Q2) brings back records whose ids sort BELOW the content
    branches — 605 < 2424, 1022 < 1685, 602 < 2250 — so each would capture its label and
    drag the term into the reinstated facet. Three overrides pin them back. This is a
    mechanism, not three accidents: the next harvest can produce more, and a silent
    facet change is exactly what a reviewed placement must not suffer."""
    nested, _audit = _built_union()
    assert cs in nested[facet], f"{cs!r} left {facet!r}"
    entry = nested[facet][cs]
    reachable = {entry["source_id"]} | {d["id"] for d in (entry.get("discarded_ids") or [])}
    assert ids <= reachable, f"{cs!r} lost ids: {ids - reachable}"


def test_reinstatement_added_the_two_context_facets_and_moved_nothing_else():
    """Q1 and Q2 land in their own facets so either can be retired on its own (M11's
    "evaluate later"), and the eight archaeological facets keep every term they had."""
    nested, _audit = _built_union()
    assert len(nested["Cultural & Geographic Context"]) == 972
    assert len(nested["Related Disciplines & Society"]) == 1666
    assert sum(len(t) for t in nested.values()) == 4718
    assert nested.get("Other", {}) == {}


def test_the_reinstated_facets_sit_last_in_the_prompt():
    """Facet order is load-bearing — build_system_prompt truncates a *suffix*. Terms
    reinstated on probation must be the first dropped when context is tight, not
    compete with Chronology for the front."""
    m = _shipped_manager()
    order = m._theme_order()
    tail = [f for f in order if f != "Other"][-2:]
    assert set(tail) == {"Cultural & Geographic Context", "Related Disciplines & Society"}


# ── M13: @david-spacil's collision verdicts (issue #6, comment 5541507280) ──────


@pytest.mark.parametrize(
    "qualified,bare,facet,winner_id",
    [
        ("komunikace (aktivita)", "komunikace", "Activity Area", "HES-000006"),
        ("kost (předmět)", "kost", "Artefact", "HES-000753"),
        ("pastvina/louka (nálezové okolnosti)", "pastvina/louka", "Finds Context", "HES-000244"),
        ("rostlinné makrozbytky (materiál)", "rostlinné makrozbytky", "Material", "HES-001009"),
        ("vejce (materiál)", "vejce", "Material", "HES-000956"),
        ("žároviště (spálená vrstva)", "žároviště", "Feature", "HES-000463"),
    ],
)
def test_each_m13_split_offers_both_senses(qualified, bare, facet, winner_id):
    """All seven verdicts (six here plus `zámek`) put the qualifier on the record that
    LEAVES the group, so the bare label stays offered by whatever remains. Both senses
    must therefore be selectable — which is the whole point of the split."""
    nested, _audit = _built_union()
    assert nested[facet][qualified]["source_id"] == winner_id
    assert nested[facet][qualified]["bare_cs"] == bare
    assert any(bare in terms for terms in nested.values()), f"{bare!r} lost its bare entry"


def test_the_seven_splits_are_the_whole_verdict_set():
    nested, _audit = _built_union()
    split = {cs for terms in nested.values() for cs, e in terms.items() if e.get("bare_cs")}
    assert split == {
        "komunikace (aktivita)",
        "kost (předmět)",
        "pastvina/louka (nálezové okolnosti)",
        "rostlinné makrozbytky (materiál)",
        "vejce (materiál)",
        "zámek (sídlo elity)",
        "žároviště (spálená vrstva)",
    }


def test_no_record_that_survives_exclusion_is_lost_by_dedup_or_splitting():
    """M7's standing invariant, stated so it survives a config change: every record that
    reaches dedup must be findable in the built vocabulary — as an entry's own id, or on
    its survivor's `discarded_ids`. A split moves a record into its own entry; it never
    discards one. Asserting a fixed count instead would have to be rewritten every time
    a branch is reinstated, which is exactly when the invariant matters most."""
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    kept = {
        (r.source, r.source_id)
        for name, (records, _m) in per_source.items()
        for r in vb._filter_excluded(records, manager)
        if r.cs and r.en and not (r.source == "teater" and not r.broader)
    }

    nested, _audit = _built_union()
    reachable = {
        (e["source"], e["source_id"]) for terms in nested.values() for e in terms.values()
    } | {
        (d["source"], d["id"])
        for terms in nested.values()
        for e in terms.values()
        for d in (e.get("discarded_ids") or [])
    }
    assert kept - reachable == set(), f"{len(kept - reachable)} records vanished"


# ── the geographic guardrail gate (issue #6, O4 / C1) ───────────────────────────


def test_the_build_refuses_a_vocabulary_that_contradicts_the_prompt():
    """A reinstated geographic branch and a prompt that still forbids selecting one is
    the failure the O3/O4 package warns about — the scores would measure the
    contradiction rather than the model. Caught at build time, before any artifact is
    written, rather than showing up as an unexplained score."""
    manager = _shipped_manager()
    assert vb._check_geo_guardrail(manager) is None  # the shipped pair agrees

    # Since M11 the branches ARE reinstated and the wording IS relaxed, so the
    # contradiction has to be constructed rather than assumed: put the strict clause
    # back in force while 2560 stays selectable — the exact state the decision package
    # named as the one outcome to avoid.
    settings = manager.taxonomy["_settings"]
    settings["geo_guardrail"] = {
        **settings["geo_guardrail"],
        "active": True,
        "covers": ["teater:2560"],
    }
    assert settings["teater_branch_map"]["2560"] != "__exclude__"
    with pytest.raises(SystemExit, match="teater:2560 is offered to the model"):
        vb._check_geo_guardrail(manager)


def test_the_gate_runs_before_anything_is_written(tmp_path, monkeypatch):
    """`--check` never writes, but a real build does — so the gate has to sit ahead of
    the first _emit, or a contradictory vocabulary reaches disk before anyone is told."""
    if not (VOCAB_DIR / "amcr_flat.json").exists():
        pytest.skip("flat artifacts not present in this checkout")

    import json

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    config["_settings"]["geo_guardrail"] = {
        **config["_settings"]["geo_guardrail"],
        "active": True,
        "covers": ["teater:2560"],
    }
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
