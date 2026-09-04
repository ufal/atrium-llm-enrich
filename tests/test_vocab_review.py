"""
tests/test_vocab_review.py — unit and end-to-end tests for vocab_review.py.

Like test_vocab_build.py, the end-to-end tests run against the real committed flat
harvests: a synthetic fixture could not have caught the geographic_or_ethnic /
"settled vs. still-open" categorisation bug this file's own history includes (see
test_exclusion_impact_settled_and_open_totals_reconcile) -- that bug was invisible on
made-up data and only showed up once the real term counts stopped adding up to 2930.
"""

import csv
from pathlib import Path

import pytest

import vocab_build as vb
import vocab_review as vr
import vocab_sources as vs
from vocab_manager import VocabularyManager

REPO_ROOT = Path(__file__).resolve().parent.parent
VOCAB_DIR = REPO_ROOT / "data_samples" / "vocab"
CONFIG = REPO_ROOT / "data_samples" / "taxonomy_config.json"


def _shipped_manager():
    return VocabularyManager(config_path=str(CONFIG))


def _require_flat():
    if not (VOCAB_DIR / "amcr_flat.json").exists() or not (VOCAB_DIR / "teater_flat.json").exists():
        pytest.skip("flat artifacts not present in this checkout")


# ── gloss_similarity / group_dissimilarity (pure functions) ─────────────────────


@pytest.mark.parametrize(
    "a,b,expected_low",
    [
        ("amphora", "amphorae", False),  # near-identical, plural variance
        ("armour and weapons", "weapons and armour", False),  # same tokens, reordered
        ("lock", "châteaux", True),  # genuinely different
        ("colourant", "dye", True),  # genuinely different
    ],
)
def test_gloss_similarity_separates_variance_from_real_difference(a, b, expected_low):
    score = vr.gloss_similarity(a, b)
    assert 0.0 <= score <= 1.0
    assert (score < 0.5) == expected_low


def test_gloss_similarity_is_symmetric_and_identical_is_one():
    assert vr.gloss_similarity("hearth", "hearth") == 1.0
    assert vr.gloss_similarity("a", "b") == vr.gloss_similarity("b", "a")


def test_group_dissimilarity_is_the_worst_pair_not_the_average():
    a = vs.VocabRecord(cs="x", en="fireplace", source="amcr", source_id="1")
    b = vs.VocabRecord(cs="x", en="hearth", source="teater", source_id="2", broader=("1",))
    c = vs.VocabRecord(cs="x", en="fire place", source="teater", source_id="3", broader=("1",))
    # Three distinct glosses, three pairwise scores — the group score must be the
    # lowest of the three (the worst pair), not their average or the first computed.
    pairwise = [
        vr.gloss_similarity("fireplace", "hearth"),
        vr.gloss_similarity("fireplace", "fire place"),
        vr.gloss_similarity("hearth", "fire place"),
    ]
    assert vr.group_dissimilarity([a, b, c]) == min(pairwise)
    assert min(pairwise) < sum(pairwise) / len(pairwise)  # genuinely below the average


def test_gloss_similarity_matches_regardless_of_which_side_a_set_iterates_first():
    """SequenceMatcher.ratio() alone is not symmetric (autojunk + matching-block
    search depend on argument order); group_dissimilarity iterates a *set* of
    glosses, whose iteration order Python does not guarantee, so gloss_similarity
    itself must not depend on which argument comes first."""
    assert vr.gloss_similarity("hearth", "fire place") == vr.gloss_similarity(
        "fire place", "hearth"
    )


def test_group_dissimilarity_single_gloss_group_is_not_flagged():
    a = vs.VocabRecord(cs="x", en="church", source="amcr", source_id="1")
    b = vs.VocabRecord(cs="x", en="church", source="teater", source_id="2", broader=("1",))
    assert vr.group_dissimilarity([a, b]) == 1.0


# ── collision_review_rows (M8 / Track 2) ─────────────────────────────────────────


def test_collision_review_zamek_ranks_near_the_top_of_real_collisions():
    """Validates the ranking heuristic against the one collision already confirmed
    as a genuine homonym (M8, motyc 5439363875) — it must not be buried among the
    ~470 real collision groups, most of which are plain translation variance."""
    _require_flat()
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    filtered = [
        r for name, (recs, _m) in per_source.items() for r in vb._filter_excluded(recs, manager)
    ]
    rows = vr.collision_review_rows(filtered, manager)

    ordered_groups = list(dict.fromkeys(r["cs"] for r in rows))
    assert "zámek" in ordered_groups
    rank = ordered_groups.index("zámek") + 1
    assert rank <= 20, f"zámek ranked {rank} of {len(ordered_groups)} — heuristic regressed"


def test_collision_review_excludes_single_gloss_groups():
    """A group where every record shares one EN gloss is plain dedup (B1/B2 already
    carry every id) and must not appear — it needs no human review."""
    _require_flat()
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    filtered = [
        r for name, (recs, _m) in per_source.items() for r in vb._filter_excluded(recs, manager)
    ]
    rows = vr.collision_review_rows(filtered, manager)
    assert "kostel" not in {r["cs"] for r in rows}  # both AMCR records say "church"


def test_collision_review_rows_are_ranked_ascending():
    _require_flat()
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    filtered = [
        r for name, (recs, _m) in per_source.items() for r in vb._filter_excluded(recs, manager)
    ]
    rows = vr.collision_review_rows(filtered, manager)
    scores = [r["dissimilarity"] for r in rows]
    assert scores == sorted(scores)


# ── composite_pair_rows (O1/F / Track 3) ─────────────────────────────────────────


def _built_union():
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    filtered = {name: vb._filter_excluded(recs, manager) for name, (recs, _m) in per_source.items()}
    qualifiers = manager.qualifier_overrides()
    records = filtered.get("amcr", []) + filtered.get("teater", [])
    nested, _audit, _collisions = vb._nest(manager, records, qualifiers=qualifiers)
    return nested


@pytest.mark.parametrize(
    "composite,component",
    [
        ("most/brod", "most"),
        ("muzeum/skanzen", "muzeum"),
        ("budova/stavba", "budova"),
        ("zámek/zámeček/vila", "vila"),
    ],
)
def test_composite_pairs_finds_the_flagship_examples(composite, component):
    """The four pairs named throughout the issue thread (finding 2, comment
    5427831261) must all be detected on the real, currently-adopted vocabulary."""
    _require_flat()
    nested = _built_union()
    rows = vr.composite_pair_rows(nested)
    hits = [r for r in rows if r["composite_cs"] == composite and r["component_cs"] == component]
    assert hits, f"{composite!r} <-> {component!r} not found in composite_pair_rows"


def test_composite_pairs_never_matches_a_label_against_itself():
    _require_flat()
    nested = _built_union()
    rows = vr.composite_pair_rows(nested)
    for r in rows:
        assert r["composite_cs"] != r["component_cs"]


def test_composite_sheet_reports_the_reviewers_own_verdicts():
    """A reviewer rules on a pair by editing taxonomy_overrides.json; the sheet they
    review must then show that ruling, or they cannot tell a pair they have already
    handled from one they have not."""
    _require_flat()
    manager = _shipped_manager()
    nested = _built_union()

    plain = vr.composite_pair_rows(nested, manager)
    assert {r["link_status"] for r in plain} == {"auto"}
    most_brod = next(r for r in plain if r["composite_cs"] == "most/brod")

    manager.overrides[(most_brod["composite_source"], most_brod["composite_id"])] = {
        "same_as_suppress": [
            {"source": most_brod["component_source"], "id": most_brod["component_id"]}
        ],
        "reason": "fabricated for this test",
    }
    after = vr.composite_pair_rows(nested, manager)
    assert len(after) == len(plain)
    row = next(
        r
        for r in after
        if r["composite_cs"] == "most/brod" and r["component_cs"] == most_brod["component_cs"]
    )
    assert row["link_status"] == "suppressed"


def test_composite_sheet_lists_a_hand_declared_pair_no_label_implies():
    """A `same_as` link the detector cannot see has no row of its own from detection,
    so it would be invisible on the sheet — the one place a reviewer looks."""
    _require_flat()
    manager = _shipped_manager()
    nested = _built_union()
    before = len(vr.composite_pair_rows(nested, manager))

    kostel = nested["Feature"]["kostel"]
    kaple = nested["Feature"]["kaple"]
    manager.overrides[(kostel["source"], kostel["source_id"])] = {
        "same_as": [{"source": kaple["source"], "id": kaple["source_id"]}],
        "reason": "fabricated for this test",
    }
    rows = vr.composite_pair_rows(nested, manager)
    assert len(rows) == before + 1
    added = next(r for r in rows if r["link_status"] == "manual")
    assert {added["composite_cs"], added["component_cs"]} == {"kostel", "kaple"}


# ── specificity_pair_rows (D1 / O2) ──────────────────────────────────────────────


def test_specificity_pairs_names_the_nearest_offered_ancestor():
    """The row must name the smallest step the model got wrong, not the branch root —
    `paleogén` sits under `třetihory`, four rungs below `geologická doba`, and a
    reviewer ruling on partial credit needs the near miss, not the top of the tree."""
    _require_flat()
    per_source = vb._load_flat(VOCAB_DIR)
    rows = {r["cs"]: r for r in vr.specificity_pair_rows(_built_union(), per_source)}

    paleogen = rows["paleogén"]
    assert paleogen["nearest_ancestor_cs"] == "třetihory"
    assert paleogen["outermost_offered_ancestor_cs"] == "geologická doba"
    assert paleogen["rungs_above"] == 3


def test_specificity_pairs_only_lists_terms_whose_ancestor_is_also_offered():
    """A term whose ancestors were all excluded is not a scoring ambiguity — nothing
    else was selectable, so the model had no less-specific answer to give."""
    _require_flat()
    per_source = vb._load_flat(VOCAB_DIR)
    nested = _built_union()
    offered = {cs for terms in nested.values() for cs in terms}
    for row in vr.specificity_pair_rows(nested, per_source):
        assert row["nearest_ancestor_cs"] in offered
        assert row["outermost_offered_ancestor_cs"] in offered
        assert row["cs"] != row["nearest_ancestor_cs"]


def test_every_specificity_pair_sits_inside_one_facet():
    """Load-bearing for D1: if these pairs crossed facets, scoring the facet instead of
    the term would separate a near miss from a category error. They do not — every one
    is intra-facet, so facet-level scoring cannot tell the two apart and the rule has
    to be made at term level."""
    _require_flat()
    per_source = vb._load_flat(VOCAB_DIR)
    rows = vr.specificity_pair_rows(_built_union(), per_source)
    assert rows, "the shipped vocabulary must produce specificity pairs"
    assert all(r["same_facet"] for r in rows)


def test_specificity_verdict_column_is_left_for_the_reviewer():
    """Same convention as collision_review.csv's `verdict` and the reinstatement
    sheet's `would_be_facet`: the tool measures, @motyc and @david-spacil rule (M10)."""
    _require_flat()
    per_source = vb._load_flat(VOCAB_DIR)
    rows = vr.specificity_pair_rows(_built_union(), per_source)
    assert {r["verdict"] for r in rows} == {""}


def test_specificity_pairs_is_empty_without_a_hierarchy():
    """AMCR has no broader chain, so an AMCR-only build has no ladder at all — the
    report must be empty rather than inventing one from label shape."""
    nested = {"Feature": {"most": {"cs": "most", "source": "amcr", "source_id": "A1"}}}
    assert vr.specificity_pair_rows(nested, {"amcr": ([], None)}) == []


# ── exclusion_impact_rows (O3/O4 / Track 4) ──────────────────────────────────────


def test_exclusion_impact_settled_and_open_totals_reconcile():
    """The three status buckets must partition every excluded term with no overlap
    and no gap — this is the check that would have caught the geographic_or_ethnic
    vs. settled-M4 conflation this tool's own history had (240 already-settled AMCR
    technical terms were briefly folded into an 'open' total)."""
    _require_flat()
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    rows = vr.exclusion_impact_rows(per_source, manager)

    by_status = {}
    for r in rows:
        by_status[r["status"]] = by_status.get(r["status"], 0) + r["term_count"]
    assert set(by_status) == {
        "settled (M4)",
        "open: geographic/ethnic (O3/O4 Q1)",
        "open: other (O4 Q2)",
    }
    assert sum(by_status.values()) == sum(r["term_count"] for r in rows)


@pytest.mark.parametrize(
    "rule",
    ["heslar:zeme", "heslar:jazyk", "teater:2560", "teater:2900", "teater:3076"],
)
def test_exclusion_impact_flags_exactly_the_guardrail_conflicting_lists(rule):
    """O3/O4: only the lists the guardrail's own wording names (country, language,
    geographic region) or TEATER's own ethnic/historical/dynastic labels are Q1 —
    everything else excluded is Q2 or already settled, not a guardrail question."""
    _require_flat()
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    rows = vr.exclusion_impact_rows(per_source, manager)
    row = next(r for r in rows if r["rule"] == rule)
    assert row["status"] == "open: geographic/ethnic (O3/O4 Q1)"


def test_exclusion_impact_technical_amcr_lists_are_settled_not_open():
    """dokument_material etc. were ruled on under M4 — they are not part of O4 and
    must not appear alongside the still-open questions."""
    _require_flat()
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    rows = vr.exclusion_impact_rows(per_source, manager)
    row = next(r for r in rows if r["rule"] == "heslar:dokument_material")
    assert row["status"] == "settled (M4)"


# ── CLI smoke test ────────────────────────────────────────────────────────────────


def test_main_all_writes_three_csvs(tmp_path, monkeypatch):
    """The CLI writes into --vocab-dir, never the real data_samples/vocab/ during a
    test — this test passes a tmp_path copy so it never touches the committed tree."""
    _require_flat()
    import shutil

    dest = tmp_path / "vocab"
    dest.mkdir()
    for name in ("amcr_flat.json", "teater_flat.json"):
        shutil.copy(VOCAB_DIR / name, dest / name)

    rc = vr.main(["--all", "--vocab-dir", str(dest), "--config", str(CONFIG)])
    assert rc == 0
    assert (dest / "collision_review.csv").exists()
    assert (dest / "composite_pairs.csv").exists()
    assert (dest / "exclusion_impact.csv").exists()


def test_main_with_no_flags_is_a_usage_error(capsys):
    assert vr.main([]) == 2


# ── collision_review_rows enrichment: source metadata + aat_verdict ─────────────


def test_collision_review_carries_source_metadata_columns():
    """The flat harvests hold note_cs/alt_cs/alt_en/de/uri/exact_match on every
    VocabRecord, and none of it reached a review sheet before this — a homonym
    verdict was being issued from a bare EN gloss while a disambiguating scope note
    sat in the committed flat file. zámek's TEATER 2358 record is the concrete case:
    its note_cs is what actually explains why it means "lock", not "châteaux"."""
    _require_flat()
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    filtered = [
        r for name, (recs, _m) in per_source.items() for r in vb._filter_excluded(recs, manager)
    ]
    label_index = vr._label_index(per_source)
    rows = vr.collision_review_rows(filtered, manager, label_index=label_index)

    for col in ("note_cs", "alt_cs", "alt_en", "de", "uri", "broader_labels", "exact_match"):
        assert col in rows[0]

    lock = next(
        r for r in rows if r["cs"] == "zámek" and r["source"] == "teater" and r["id"] == "2358"
    )
    assert "uzavírání dveří" in lock["note_cs"]
    assert lock["de"] == "Schloss (s)"


def test_collision_review_aat_verdict_distribution_reconciles():
    """Verified against the shipped union vocabulary's 127 differing-gloss groups.
    Only conflicting/agreeing are strong signal; a regression in _aat_verdict's
    threshold would silently change which ~20 groups a reviewer can bulk-handle."""
    _require_flat()
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    filtered = [
        r for name, (recs, _m) in per_source.items() for r in vb._filter_excluded(recs, manager)
    ]
    rows = vr.collision_review_rows(filtered, manager)

    import collections

    by_group = collections.OrderedDict()
    for r in rows:
        by_group.setdefault(r["cs"], r["aat_verdict"])
    counts = collections.Counter(by_group.values())
    assert counts == {"agreeing": 17, "none": 46, "one_sided": 61, "conflicting": 3}


def test_aat_verdict_conflicting_when_getty_alignments_disagree():
    a = vs.VocabRecord(
        cs="x", en="p", source="amcr", source_id="1", exact_match=("http://getty/A",)
    )
    b = vs.VocabRecord(
        cs="x", en="q", source="teater", source_id="2", exact_match=("http://getty/B",)
    )
    assert vr._aat_verdict([a, b]) == "conflicting"


def test_aat_verdict_agreeing_when_getty_alignments_share_a_uri():
    a = vs.VocabRecord(
        cs="x",
        en="p",
        source="amcr",
        source_id="1",
        exact_match=("http://getty/A", "http://getty/shared"),
    )
    b = vs.VocabRecord(
        cs="x", en="q", source="teater", source_id="2", exact_match=("http://getty/shared",)
    )
    assert vr._aat_verdict([a, b]) == "agreeing"


def test_aat_verdict_one_sided_and_none():
    a = vs.VocabRecord(
        cs="x", en="p", source="amcr", source_id="1", exact_match=("http://getty/A",)
    )
    b = vs.VocabRecord(cs="x", en="q", source="teater", source_id="2")
    assert vr._aat_verdict([a, b]) == "one_sided"
    assert vr._aat_verdict([b]) == "none"


def test_render_broader_uses_labels_and_falls_back_to_ids():
    r = vs.VocabRecord(cs="x", en="y", source="teater", source_id="9", broader=("1267", "999"))
    index = {("teater", "1267"): "6) Areál aktivity / Activity area"}
    assert vr._render_broader(r, index) == "6) Areál aktivity / Activity area > 999"
    assert vr._render_broader(r, {}) == "1267 > 999"


def test_render_broader_empty_when_no_ancestors():
    r = vs.VocabRecord(cs="x", en="y", source="amcr", source_id="1")
    assert vr._render_broader(r, {}) == ""


# ── exclusion_impact_rows: override-driven exclusions are not silently dropped ──


def test_exclusion_impact_includes_override_exclusions():
    """Defect fix: the filter used to keep only rule.startswith(('heslar:', 'teater:')),
    so a per-term exclusion via taxonomy_overrides.json would be missing from the sheet
    entirely — latent today (no shipped override excludes anything), so this test
    fabricates one rather than relying on real data."""
    manager = _shipped_manager()
    manager.overrides[("amcr", "FAKE-999")] = {
        "facet": "__exclude__",
        "reason": "fabricated for the override-exclusion regression test",
    }
    fake = vs.VocabRecord(
        cs="fake term", en="fake gloss", source="amcr", source_id="FAKE-999", scheme="areal"
    )
    per_source = {"amcr": ([fake], {})}

    rows = vr.exclusion_impact_rows(per_source, manager)
    row = next(r for r in rows if r["rule"] == "override:amcr:FAKE-999")
    assert row["term_count"] == 1
    assert row["status"] == "settled (per-term override)"
    assert row["stated_reason"] == "fabricated for the override-exclusion regression test"


def test_exclusion_status_comes_from_the_config_not_from_this_module():
    """The Q1/Q2 split used to be two hard-coded sets in vocab_review.py, so moving one
    branch between "already ruled on" and "still open" needed a developer. It is now a
    `status` on the `_exclusions` entry: editing the config alone must change both the
    status column and the guardrail_conflict flag."""
    _require_flat()
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)

    before = {r["rule"]: r["status"] for r in vr.exclusion_impact_rows(per_source, manager)}
    assert before["teater:3094"] == "open: other (O4 Q2)"

    manager.settings["_exclusions"]["teater:3094"]["status"] = "settled"
    after = {r["rule"]: r["status"] for r in vr.exclusion_impact_rows(per_source, manager)}
    assert after["teater:3094"] == "settled (M4)"
    assert {k: v for k, v in after.items() if k != "teater:3094"} == {
        k: v for k, v in before.items() if k != "teater:3094"
    }


def test_guardrail_conflict_follows_the_declared_status():
    """The one status the tool acts on rather than prints: `guardrail_conflict` marks
    the rows whose reinstatement also needs the prompt's geographic wording relaxed."""
    _require_flat()
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)

    rows = {r["rule"]: r for r in vr.reinstatement_preview_rows(per_source, manager)}
    assert rows["teater:2560"]["guardrail_conflict"] is True  # etnika — Q1
    assert rows["teater:3094"]["guardrail_conflict"] is False  # povolání — Q2

    # Both halves, because validate_settings now refuses to let them drift apart:
    # `covers` is what says the wording reaches this branch at all.
    manager.settings["_exclusions"]["teater:3094"]["status"] = "open_geo_ethnic"
    manager.settings["geo_guardrail"]["covers"].append("teater:3094")
    flipped = {r["rule"]: r for r in vr.reinstatement_preview_rows(per_source, manager)}
    assert flipped["teater:3094"]["guardrail_conflict"] is True


def test_subbranch_impact_inherits_its_roots_status_when_unkeyed():
    """A sub-branch nobody keyed separately is covered by its root's ruling, so the
    guardrail flag must fall back to the root rather than silently reading False."""
    _require_flat()
    manager = _shipped_manager()
    notes = manager.exclusion_notes()
    assert "teater:2560" in notes  # keyed in its own right
    assert not any(k.startswith("teater:3095") for k in notes)  # a child of 3094, not keyed

    assert vr._status_of(notes, "teater:2560", "teater:2557") == "open_geo_ethnic"
    assert vr._status_of(notes, "teater:3095", "teater:3094") == "open_other"
    assert vr._status_of(notes, "teater:9999") == "settled"  # unknown → the default


def test_exclusion_impact_override_status_is_distinct_from_m4():
    """An override-driven exclusion must never be silently folded into 'settled (M4)'
    — M4 names a specific, already-ruled-on class of AMCR technical lists, and an
    override could exclude anything."""
    manager = _shipped_manager()
    manager.overrides[("teater", "FAKE-1")] = {"facet": "__exclude__", "reason": ""}
    fake = vs.VocabRecord(
        cs="fake teater term",
        en="fake gloss",
        source="teater",
        source_id="FAKE-1",
        broader=("1050",),
    )
    per_source = {"teater": ([fake], {})}
    rows = vr.exclusion_impact_rows(per_source, manager)
    row = next(r for r in rows if r["rule"] == "override:teater:FAKE-1")
    assert row["status"] != "settled (M4)"


# ── teater_subbranch_impact_rows (O3/O4, finer grain than exclusion_impact) ─────


def test_subbranch_impact_decomposes_only_excluded_depth1_roots():
    """Exactly the 5 real excluded depth-1 roots (1, 288, 2557, 3094, 3549) decompose
    into 23 sub-branch rows in total — matching the '23 deferred rows' the domain
    review itself named. A currently-kept root (e.g. 1050 Chronology) must never
    appear, and neither must a leaf inside an already-finest-grain sub-branch (2560
    etnika's own ~390 individual ethnic-group names must not explode into rows)."""
    _require_flat()
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    rows = vr.teater_subbranch_impact_rows(per_source, manager)

    assert len(rows) == 23
    assert set(r["root"] for r in rows) == {"1", "288", "2557", "3094", "3549"}
    # 2560 (etnika) is a sub-branch of 2557 and is included once; its own leaf
    # children (individual ethnic-group names) must not be decomposed further.
    assert "2560" not in {r["root"] for r in rows}


def test_subbranch_impact_children_plus_root_reconciles_with_exclusion_impact():
    """Sum of a root's children's term_count, plus the root's own single record
    (which exclusion_impact_rows counts too, since it has its own cs/en), must equal
    exclusion_impact.csv's coarser per-root figure exactly -- the two sheets describe
    the same terms at different granularity and must never silently disagree.

    2557 is the deliberate exception: unlike the other four roots, ALL five of its
    children (2558/2560/2900/3076/3091) are THEMSELVES individually-mapped
    teater_branch_map keys, so assign_theme's most-specific-first walk routes every
    descendant to a child rule and never falls back to "teater:2557" -- that rule
    then covers only 2557's own single record. Reconciling the sum-of-children with
    that root would be reconciling against the wrong thing.
    """
    _require_flat()
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    sub_rows = vr.teater_subbranch_impact_rows(per_source, manager)
    root_rows = vr.exclusion_impact_rows(per_source, manager)
    root_totals = {r["rule"]: r["term_count"] for r in root_rows}

    by_root = {}
    for r in sub_rows:
        by_root[r["root"]] = by_root.get(r["root"], 0) + r["term_count"]

    for root, children_total in by_root.items():
        if root == "2557":
            assert root_totals["teater:2557"] == 1
            continue
        assert children_total + 1 == root_totals[f"teater:{root}"], root


def test_subbranch_impact_guardrail_conflict_matches_status_geo_ethnic():
    """Only 2560/2900/3076 carry the guardrail conflict at sub-branch grain -- 2558
    (battles) and 3091 (wars) sit in the same parent branch but are not geographic,
    ethnic, or dynastic, so they must read False even though their parent (2557) is
    the guardrail-conflicting family."""
    _require_flat()
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    rows = vr.teater_subbranch_impact_rows(per_source, manager)
    by_id = {r["subbranch"]: r["guardrail_conflict"] for r in rows}
    assert by_id["2560"] is True
    assert by_id["2900"] is True
    assert by_id["3076"] is True
    assert by_id["2558"] is False
    assert by_id["3091"] is False


def test_teater_children_index_builds_expected_adjacency():
    a = vs.VocabRecord(cs="root", en="root", source="teater", source_id="1")
    b = vs.VocabRecord(cs="child", en="child", source="teater", source_id="2", broader=("1",))
    c = vs.VocabRecord(
        cs="grandchild", en="grandchild", source="teater", source_id="3", broader=("1", "2")
    )
    index = vr._teater_children_index([a, b, c])
    assert index["1"] == ["2"]
    assert index["2"] == ["3"]


def test_teater_subtree_includes_the_node_itself():
    a = vs.VocabRecord(cs="root", en="root", source="teater", source_id="1")
    b = vs.VocabRecord(cs="child", en="child", source="teater", source_id="2", broader=("1",))
    c = vs.VocabRecord(cs="other", en="other", source="teater", source_id="9")
    subtree = vr._teater_subtree([a, b, c], "1")
    assert {r.source_id for r in subtree} == {"1", "2"}


# ── reinstatement_preview_rows (O3/O4, the go/no-go numbers) ────────────────────


def test_reinstatement_preview_never_mutates_the_config_or_the_manager():
    """The preview must be pure: it answers 'what if' without writing anything to
    taxonomy_config.json or changing the manager's own settings, so a reviewer can
    run it freely without any risk of it silently becoming the real build."""
    _require_flat()
    import hashlib

    before = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)

    vr.reinstatement_preview_rows(per_source, manager)

    after = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    assert before == after
    assert manager.settings["teater_branch_map"]["2560"] == "__exclude__"


def test_reinstatement_preview_covers_exactly_the_same_rules_as_exclusion_impact():
    """The two O3/O4 sheets must describe the same set of excluded rules -- one at a
    glance (exclusion_impact.csv), one with the go/no-go numbers
    (reinstatement_preview.csv) -- or a reviewer could read one and miss a rule the
    other one covers."""
    _require_flat()
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    exclusion_rules = {r["rule"] for r in vr.exclusion_impact_rows(per_source, manager)}
    preview_rules = {r["rule"] for r in vr.reinstatement_preview_rows(per_source, manager)}
    assert exclusion_rules == preview_rules


def test_reinstatement_preview_raw_count_matches_exclusion_impact_term_count():
    """The two sheets must never disagree about how big a rule's raw pool is -- they
    are both built from the same _excluded_buckets(), so a mismatch would mean the
    shared bucketing itself is inconsistent between calls."""
    _require_flat()
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    exclusion_by_rule = {
        r["rule"]: r["term_count"] for r in vr.exclusion_impact_rows(per_source, manager)
    }
    for row in vr.reinstatement_preview_rows(per_source, manager):
        assert row["raw_count"] == exclusion_by_rule[row["rule"]], row["rule"]


def test_reinstatement_preview_usable_and_collide_never_exceed_raw():
    """usable_count + collides_with_offered can fall short of raw_count (a branch can
    have its own internal duplicate labels, absorbed before either bucket), but can
    never exceed it -- every eligible record contributes to at most one bucket."""
    _require_flat()
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    for row in vr.reinstatement_preview_rows(per_source, manager):
        assert row["usable_count"] + row["collides_with_offered"] <= row["raw_count"]
        assert row["usable_count"] >= 0
        assert row["collides_with_offered"] >= 0


def test_reinstatement_preview_etnika_adds_the_full_391():
    """Cross-check against the exact figure verified during investigation: 2560
    (etnika) collides with nothing already offered, so reinstating it alone would add
    all 391 terms net-new -- matching the in-memory reinstatement probe this sheet's
    design was validated against."""
    _require_flat()
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    row = next(
        r for r in vr.reinstatement_preview_rows(per_source, manager) if r["rule"] == "teater:2560"
    )
    assert row["usable_count"] == 391
    assert row["collides_with_offered"] == 0
    assert row["guardrail_conflict"] is True


def test_reinstatement_preview_would_be_facet_is_left_blank_for_the_reviewer():
    """P2: the tool never picks a facet on a reviewer's behalf."""
    _require_flat()
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    rows = vr.reinstatement_preview_rows(per_source, manager)
    assert all(row["would_be_facet"] == "" for row in rows)


def test_approx_prompt_chars_matches_the_rendered_shape():
    assert vr._approx_prompt_chars("kostel", "church") == len("kostel (church)") + 2
    assert vr._approx_prompt_chars("x", None) == len("x ()") + 2


# ── drift guard: committed review sheets must match a fresh generation ──────────


def test_committed_review_sheets_match_a_fresh_generation():
    """The same failure mode as commit a5e3c8a, one level up: a taxonomy_config.json
    or taxonomy_overrides.json edit changes what these sheets should say, and nothing
    re-runs vocab_review.py, so a reviewer opens a stale CSV and rules on it. The
    vocabulary artifacts have had a drift gate since the artifact/config split bit us;
    the REVIEW sheets a domain expert actually reads had none.

    Only the corpus-INDEPENDENT sheets are covered. corpus_term_evidence.csv,
    corpus_branch_evidence.csv and gold_workbook.csv are deliberately excluded: they
    are built from data_samples/UDP + DOC_LINE_CATEG, and the real reports are not
    tracked in git (issue #19 attachment), so a fresh run here would legitimately
    differ from sheets generated against the full corpus. corpus_review.py's own
    --force guard is what protects those.
    """
    _require_flat()
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    filtered = {name: vb._filter_excluded(recs, manager) for name, (recs, _m) in per_source.items()}
    records = filtered.get("amcr", []) + filtered.get("teater", [])

    qualifiers = manager.qualifier_overrides()
    nested, _audit, _collisions = vb._nest(manager, records, qualifiers=qualifiers)

    expected = {
        "collision_review.csv": (
            vr.collision_review_rows(records, manager, label_index=vr._label_index(per_source)),
            vr.COLLISION_COLUMNS,
        ),
        "composite_pairs.csv": (vr.composite_pair_rows(nested, manager), vr.COMPOSITE_COLUMNS),
        "exclusion_impact.csv": (
            vr.exclusion_impact_rows(per_source, manager),
            vr.EXCLUSION_COLUMNS,
        ),
        "teater_subbranch_impact.csv": (
            vr.teater_subbranch_impact_rows(per_source, manager),
            vr.SUBBRANCH_COLUMNS,
        ),
        "reinstatement_preview.csv": (
            vr.reinstatement_preview_rows(per_source, manager),
            vr.REINSTATEMENT_COLUMNS,
        ),
        "specificity_pairs.csv": (
            vr.specificity_pair_rows(nested, per_source),
            vr.SPECIFICITY_COLUMNS,
        ),
    }

    import io

    stale = []
    for name, (rows, columns) in expected.items():
        path = VOCAB_DIR / name
        if not path.exists():
            stale.append(f"{name} (missing)")
            continue
        buf = io.StringIO(newline="")
        writer = csv.DictWriter(buf, fieldnames=list(columns), lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
        if buf.getvalue() != path.read_text(encoding="utf-8"):
            stale.append(name)

    assert not stale, (
        "committed review sheets are stale relative to the current taxonomy config: "
        + ", ".join(stale)
        + " — run `python3 vocab_review.py --all` and commit the result."
    )
