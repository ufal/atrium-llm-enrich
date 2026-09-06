"""
tests/test_vocab_review.py — unit and end-to-end tests for vocab_review.py.

Like test_vocab_build.py, the end-to-end tests run against the real committed flat
harvests: a synthetic fixture could not have caught the geographic_or_ethnic /
"settled vs. still-open" categorisation bug this file's own history includes (see
test_exclusion_impact_settled_and_open_totals_reconcile) -- that bug was invisible on
made-up data and only showed up once the real term counts stopped adding up to 2930.
"""

import csv
import json
from pathlib import Path

import pytest

import prompt_template
import vocab_build as vb
import vocab_review as vr
import vocab_sources as vs
from vocab_manager import VocabularyManager

REPO_ROOT = Path(__file__).resolve().parent.parent
VOCAB_DIR = REPO_ROOT / "data_samples" / "vocab"
CONFIG = REPO_ROOT / "data_samples" / "taxonomy_config.json"


def _shipped_manager():
    return VocabularyManager(config_path=str(CONFIG))


# The O3/O4 branches, as they were configured before M11 reinstated them. Several tests
# below exercise the exclusion/reinstatement reporting itself, which needs something
# excluded to report. Rebuilding that state here rather than reading it out of the
# shipped config keeps those tests measuring the TOOL rather than the current ruling —
# which is what they were always meant to do, and what stops the next ruling breaking
# them again.
_Q1_RULES = {
    "heslar:zeme": "249 country names.",
    "heslar:jazyk": "9 language names.",
    "teater:2560": "391 ethnonyms.",
    "teater:2900": "317 historical regions.",
    "teater:3076": "15 ruling dynasties.",
}
_Q2_RULES = {
    "teater:1": "Theory and approaches.",
    "teater:288": "Cross-border disciplines.",
    "teater:2557": "Auxiliary-historical branch root.",
    "teater:2558": "battles.",
    "teater:3091": "wars.",
    "teater:3094": "professions.",
    "teater:3549": "society.",
}


def _manager_with_q1q2_excluded():
    """The shipped manager with the twelve O3/O4 rules put back to `__exclude__`."""
    manager = _shipped_manager()
    settings = manager.settings
    for rule, reason in {**_Q1_RULES, **_Q2_RULES}.items():
        kind, ident = rule.split(":", 1)
        settings["heslar_map" if kind == "heslar" else "teater_branch_map"][ident] = "__exclude__"
        settings["_exclusions"][rule] = {
            "status": "open_geo_ethnic" if rule in _Q1_RULES else "open_other",
            "reason": reason,
        }
    settings["geo_guardrail"] = {
        **settings["geo_guardrail"],
        "active": True,
        "covers": sorted(_Q1_RULES),
    }
    manager._invalidate_cache()
    return manager


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


def _shipped_collision_rows():
    _require_flat()
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    filtered = [
        r for _name, (recs, _m) in per_source.items() for r in vb._filter_excluded(recs, manager)
    ]
    return vr.collision_review_rows(filtered, manager), manager


def test_every_collision_group_names_exactly_one_bare_label_holder():
    """One member of each group owns the plain Czech word. Zero would mean the label is
    offered by nobody; two would mean the sheet disagrees with what the build produces."""
    rows, _ = _shipped_collision_rows()
    groups: dict = {}
    for row in rows:
        groups.setdefault(vs.norm_label(row["cs"]), []).append(row)
    for label, members in groups.items():
        holders = [m for m in members if m["holds_bare_label"] == "yes"]
        assert len(holders) == 1, f"{label}: {len(holders)} bare-label holders"


def test_the_bare_label_holder_is_the_record_the_build_actually_gives_it_to():
    """The column is only worth anything if it agrees with `to_term_pairs`. A holder must
    own an unqualified entry in the built vocabulary — never a bracketed one."""
    rows, manager = _shipped_collision_rows()
    if not (VOCAB_DIR / "union_nested.json").exists():
        pytest.skip("union_nested.json not built")
    nested = json.loads((VOCAB_DIR / "union_nested.json").read_text(encoding="utf-8"))

    owner: dict = {}
    for facet, terms in nested.items():
        if facet.startswith("_"):
            continue
        for cs, entry in terms.items():
            if entry.get("source") and entry.get("source_id"):
                owner[(entry["source"], entry["source_id"])] = (cs, entry.get("bare_cs"))

    for row in rows:
        if row["holds_bare_label"] != "yes":
            continue
        entry = owner.get((row["source"], row["id"]))
        assert entry is not None, f"{row['source']}:{row['id']} holds a label it does not own"
        cs, bare_cs = entry
        assert bare_cs is None, f"{row['source']}:{row['id']} carries a qualifier: {cs!r}"


def test_a_qualified_record_never_holds_the_bare_label():
    """M13's convention, as a property: the qualifier goes on the record that *leaves*
    the group. The seven splits in force must all read blank in this column."""
    rows, manager = _shipped_collision_rows()
    qualifiers = manager.qualifier_overrides()
    assert qualifiers, "the shipped overrides declare no qualifier — test is vacuous"
    for row in rows:
        if (row["source"], row["id"]) in qualifiers:
            assert row["holds_bare_label"] == "", (
                f"{row['source']}:{row['id']} is qualified and still marked as holding "
                "the bare label"
            )


def test_the_malta_trap_is_visible_in_the_sheet():
    """The concrete case the runbook warns about: `malta` is held by a mortar record, so
    the country amcr:HES-001366 is a discarded id rather than something the model can
    pick. A reviewer splitting this group has to qualify the country — qualifying the
    holder instead just hands the plain word to the next mortar record."""
    rows, _ = _shipped_collision_rows()
    group = [r for r in rows if vs.norm_label(r["cs"]) == vs.norm_label("malta")]
    assert len(group) == 4, f"expected the 4-record malta group, got {len(group)}"

    holder = next(r for r in group if r["holds_bare_label"] == "yes")
    assert holder["en"] == "mortar", "the bare word malta is a mortar record, not the country"
    country = next(r for r in group if r["id"] == "HES-001366")
    assert country["holds_bare_label"] == ""


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


def test_specificity_pairs_are_intra_facet_except_where_an_override_moved_a_term():
    """Load-bearing for D1: a pair that sits inside one facet cannot be separated by
    scoring the facet instead of the term, so the partial-credit rule has to be made at
    term level. Before the M11 reinstatement every pair was intra-facet; now three are
    not, and all three are terms a `facet` override pins away from where their TEATER
    ancestor landed (`olej`, `vodní pramen`, `úřední písemnost` — the B5 regression
    fixes). The conclusion is unchanged and stronger: 3 of 3225 is not an escape
    hatch."""
    _require_flat()
    per_source = vb._load_flat(VOCAB_DIR)
    rows = vr.specificity_pair_rows(_built_union(), per_source)
    assert rows, "the shipped vocabulary must produce specificity pairs"
    cross = [r for r in rows if r["same_facet"] is not True]
    assert {r["cs"] for r in cross} == {"olej", "vodní pramen", "úřední písemnost"}
    assert len(cross) / len(rows) < 0.01


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


# ── context_budget_rows: where truncation actually cuts ──────────────────────────


def test_the_budget_ladder_still_matches_the_model_registry():
    """The ladder is hard-coded because llm_utils imports torch. That is only safe if a
    test notices when the registry gains a window the sheet does not model."""
    import re

    src = (REPO_ROOT / "llm_utils.py").read_text(encoding="utf-8")
    windows = {int(m) for m in re.findall(r'"context_window":\s*(\d+)', src)}
    assert windows == set(vr.CONTEXT_WINDOWS), (
        f"registry has {sorted(windows - set(vr.CONTEXT_WINDOWS))} that the budget sheet "
        f"does not model, or models {sorted(set(vr.CONTEXT_WINDOWS) - windows)} that no "
        "model has"
    )
    assert re.search(r"^MAX_NEW_TOKENS = (\d+)", src, re.M).group(1) == "2048"
    assert vr.CONTEXT_RESERVED == 2048 + 512


def test_a_large_window_keeps_the_whole_vocabulary():
    _require_flat()
    manager = _shipped_manager()
    rows = vr.context_budget_rows(_built_union(), manager)
    big = [r for r in rows if r["context_window"] == 131072]
    assert big and all(r["status"] == "full" for r in big)


def test_a_tight_window_cuts_the_probationary_facet_first():
    """The layout M11 asked for, made checkable: the 2 638 reinstated terms sit last in
    the priority order, so a 32k model loses them BEFORE it loses any archaeological
    facet. That is what makes "evaluate later if problems arise" affordable — the terms
    on probation are also the terms a tight budget spends last."""
    _require_flat()
    manager = _shipped_manager()
    rows = {
        r["facet"]: r
        for r in vr.context_budget_rows(_built_union(), manager)
        if r["context_window"] == 32768
    }
    assert rows["Related Disciplines & Society"]["status"] != "full"
    for facet in ("Chronology", "Activity Area", "Feature", "Artefact", "Material", "Methods"):
        assert rows[facet]["status"] == "full", f"{facet} was cut before the probation facet"


def test_the_smallest_window_cannot_hold_the_vocabulary_at_all():
    """8 192 is a real registry entry (aya-expanse-8b, bielik-11b, the GGUF build) and
    three of the archived, unsuccessful runs used exactly those models. Under 10 % of
    the vocabulary reaches them — worth knowing before a result is read as a model
    quality signal."""
    _require_flat()
    manager = _shipped_manager()
    rows = [
        r for r in vr.context_budget_rows(_built_union(), manager) if r["context_window"] == 8192
    ]
    kept = sum(r["surviving"] for r in rows)
    total = sum(r["terms"] for r in rows)
    assert kept / total < 0.10


def test_turning_off_a_prompt_block_buys_room_for_terms():
    """The overhead is charged from the CONFIGURED prompt, so a reviewer trimming a
    block in llm_config.txt can see what it buys instead of guessing."""
    _require_flat()
    manager = _shipped_manager()
    nested = _built_union()
    with_examples = vr.context_budget_rows(nested, manager, windows=(8192,), prompt_config={})
    without = vr.context_budget_rows(
        nested, manager, windows=(8192,), prompt_config={"PROMPT_EXAMPLES": "false"}
    )
    assert sum(r["surviving"] for r in without) > sum(r["surviving"] for r in with_examples)


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
    # Since M11 reinstated Q1 and Q2, every remaining exclusion is settled under M4 —
    # the two open buckets are empty because the questions were answered, not because
    # the tool stopped reporting them. Asserted as a subset so the partition check keeps
    # working if a branch is ever excluded again on evidence (which is what P3 invites).
    assert set(by_status) <= {
        "settled (M4)",
        "open: geographic/ethnic (O3/O4 Q1)",
        "open: other (O4 Q2)",
        "settled (per-term override)",
    }
    assert "settled (M4)" in by_status
    assert sum(by_status.values()) == sum(r["term_count"] for r in rows)


@pytest.mark.parametrize(
    "rule",
    ["heslar:zeme", "heslar:jazyk", "teater:2560", "teater:2900", "teater:3076"],
)
def test_the_q1_lists_are_no_longer_excluded_at_all(rule):
    """O3/O4 Q1 was the set of lists the guardrail's own wording named. M11 reinstated
    every one of them, so none should appear in the exclusion sheet — a Q1 rule still
    listed here would mean the ruling was only half applied."""
    _require_flat()
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    rows = vr.exclusion_impact_rows(per_source, manager)
    assert rule not in {r["rule"] for r in rows}


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
    """Verified against the shipped union vocabulary's 136 differing-gloss groups —
    127 before M11, plus the 9 the reinstatement added.

    `aat_verdict` is the signal that actually predicted @david-spacil's verdicts: all 3
    `conflicting` groups in the 127 he reviewed turned out to be splits, against 2–6 %
    for every other class (the dissimilarity ranking he was given did far worse — 1 of
    7 splits inside its top 30). A regression in `_aat_verdict`'s threshold would
    silently change which groups the next reviewer can bulk-handle, so the distribution
    is pinned."""
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
    assert counts == {"agreeing": 19, "none": 52, "one_sided": 62, "conflicting": 3}
    assert sum(counts.values()) == 136


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
    manager = _manager_with_q1q2_excluded()
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
    manager = _manager_with_q1q2_excluded()
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
    manager = _manager_with_q1q2_excluded()
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
    manager = _manager_with_q1q2_excluded()
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

    A per-term `facet` override is the second, smaller exception, and it is not a
    disagreement between the sheets: an override is checked BEFORE either map, so a
    record it rescues is no longer excluded by its branch and drops out of the
    exclusion count -- while the sub-branch walk, which follows TEATER's tree rather
    than the placement rules, still sees it. The three M11 regression pins
    (teater:605, 1022, 602, all under 288) are exactly that case.
    """
    _require_flat()
    manager = _manager_with_q1q2_excluded()
    per_source = vb._load_flat(VOCAB_DIR)
    sub_rows = vr.teater_subbranch_impact_rows(per_source, manager)
    root_rows = vr.exclusion_impact_rows(per_source, manager)
    root_totals = {r["rule"]: r["term_count"] for r in root_rows}

    by_root = {}
    for r in sub_rows:
        by_root[r["root"]] = by_root.get(r["root"], 0) + r["term_count"]

    rescued = {
        source_id
        for (source, source_id), override in manager.overrides.items()
        if source == "teater" and override.get("facet") not in (None, "__exclude__")
    }
    teater_records = {r.source_id: r for r in per_source["teater"][0] if r.source_id}

    for root, children_total in by_root.items():
        if root == "2557":
            assert root_totals["teater:2557"] == 1
            continue
        pinned = sum(
            1
            for sid in rescued
            if (rec := teater_records.get(sid)) is not None and root in rec.broader
        )
        assert children_total + 1 - pinned == root_totals[f"teater:{root}"], root


def test_subbranch_impact_guardrail_conflict_matches_status_geo_ethnic():
    """Only 2560/2900/3076 carry the guardrail conflict at sub-branch grain -- 2558
    (battles) and 3091 (wars) sit in the same parent branch but are not geographic,
    ethnic, or dynastic, so they must read False even though their parent (2557) is
    the guardrail-conflicting family."""
    _require_flat()
    manager = _manager_with_q1q2_excluded()
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
    manager = _manager_with_q1q2_excluded()
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
    manager = _manager_with_q1q2_excluded()
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

    # The two prompt-dependent sheets, generated the way the CLI generates them: from
    # the committed llm_config.txt, not from the module defaults. That is the whole
    # point of covering them here — context_budget.csv had no drift gate at all, and it
    # is the sheet that answers "does the vocabulary fit the model we are running?".
    # Reading the same config the CLI reads means flipping PROMPT_VOCAB_GROUPING and
    # regenerating stays green, while flipping it and NOT regenerating fires.
    prompt_config = {}
    if vr.PROMPT_CONFIG.exists():
        prompt_config = prompt_template.load_run_config(vr.PROMPT_CONFIG)
    expected["context_budget.csv"] = (
        vr.context_budget_rows(nested, manager, prompt_config=prompt_config),
        vr.BUDGET_COLUMNS,
    )
    census_nested, census_audit = vr._nest_as_built(manager, records)
    expected["facet_census.csv"] = (
        vr.facet_census_rows(census_nested, census_audit, manager, prompt_config=prompt_config),
        vr.CENSUS_COLUMNS,
    )

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


# ── the grouping's cost against the context window ───────────────────────────────


def _budget_totals(prompt_config):
    nested = json.loads((VOCAB_DIR / "union_nested.json").read_text(encoding="utf-8"))
    manager = VocabularyManager(
        config_path=str(REPO_ROOT / "data_samples" / "taxonomy_config.json")
    )
    rows = vr.context_budget_rows(nested, manager, prompt_config=prompt_config)
    totals: dict = {}
    for row in rows:
        totals[row["context_window"]] = totals.get(row["context_window"], 0) + row["surviving"]
    return totals


def test_group_headers_are_charged_against_the_window():
    """~120 header lines are not free, and they are the whole of what the grouping flag
    moves. Leaving them uncharged reported the shipped two-level layout at the flat
    layout's term count — 445 at 8 192 where it actually reaches 419."""
    if not (VOCAB_DIR / "union_nested.json").exists():
        pytest.skip("union_nested.json not built")

    grouped = _budget_totals({"PROMPT_VOCAB_GROUPING": "facet_sub"})
    facet = _budget_totals({"PROMPT_VOCAB_GROUPING": "facet"})
    flat = _budget_totals({"PROMPT_VOCAB_GROUPING": "flat"})

    assert grouped[8192] < facet[8192] < flat[8192]
    assert grouped[32768] < facet[32768] < flat[32768]
    # at a window that fits the whole vocabulary the layout costs nothing
    assert grouped[128000] == facet[128000] == flat[128000]


def test_the_budget_sheet_records_which_layout_it_was_computed_for():
    """Without it the term counts are ambiguous — they differ by 126 terms at 32k
    depending on a flag the sheet would not have named."""
    if not (VOCAB_DIR / "context_budget.csv").exists():
        pytest.skip("context_budget.csv not generated")
    with open(VOCAB_DIR / "context_budget.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "empty budget sheet"
    assert {r["grouping"] for r in rows} == {"facet_sub"}, "the sheet must state one layout"


# ── facet_census_rows (A1-facets: what is in each facet, and what it costs) ──────


def _census():
    nested = json.loads((VOCAB_DIR / "union_nested.json").read_text(encoding="utf-8"))
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    filtered = {name: vb._filter_excluded(r, manager) for name, (r, _m) in per_source.items()}
    records = filtered.get("amcr", []) + filtered.get("teater", [])
    _n, audit = vr._nest_as_built(manager, records)
    return vr.facet_census_rows(nested, audit, manager), nested


def test_census_has_one_row_per_rendered_facet_in_render_order():
    """The sheet exists so ten facets fit on one screen — one row each, and the order
    is the order the prompt renders them, which is what makes cumulative_tokens mean
    anything."""
    _require_flat()
    rows, nested = _census()
    rendered = [f for f in nested if not f.startswith("_") and nested[f]]
    assert len(rows) == len(rendered)
    assert [r["render_position"] for r in rows] == list(range(1, len(rows) + 1))
    priorities = [r["priority"] for r in rows]
    assert priorities == sorted(priorities, reverse=True), "render order must follow priority"


def test_census_term_counts_reconcile_with_the_vocabulary():
    """A census that does not add up to the vocabulary is worse than no census."""
    _require_flat()
    rows, nested = _census()
    for row in rows:
        assert row["terms"] == len(nested[row["facet"]])
    assert sum(r["terms"] for r in rows) == sum(
        len(t) for f, t in nested.items() if not f.startswith("_")
    )
    assert abs(sum(r["share_of_vocab"] for r in rows) - 1.0) < 0.01


def test_census_cumulative_tokens_are_a_running_total():
    """cumulative_tokens is the column that explains truncation: a facet is cut when
    everything *before* it has already spent the budget."""
    _require_flat()
    rows, _nested = _census()
    running = 0
    for row in rows:
        running += row["prompt_tokens"]
        assert row["cumulative_tokens"] == running


def test_census_kept_whole_from_agrees_with_the_budget_sheet():
    """The two sheets are one computation seen from two sides; if they can disagree,
    one of them is lying to a reviewer."""
    _require_flat()
    rows, nested = _census()
    manager = _shipped_manager()
    budget = vr.context_budget_rows(nested, manager)
    for row in rows:
        claimed = row["kept_whole_from"]
        if claimed.startswith(">"):
            assert not any(b["facet"] == row["facet"] and b["status"] == "full" for b in budget)
            continue
        match = [
            b for b in budget if b["facet"] == row["facet"] and str(b["context_window"]) == claimed
        ]
        assert match and match[0]["status"] == "full"
        smaller = [
            b for b in budget if b["facet"] == row["facet"] and b["context_window"] < int(claimed)
        ]
        assert all(b["status"] != "full" for b in smaller), "not the *smallest* window"


def test_census_rule_attribution_matches_the_placement_audit():
    """top_rules is the column that makes a re-layout costable — it names the map
    values a reviewer would flip. It has to come from the same audit rows the
    placement CSV is written from, not from a second guess at the same question."""
    _require_flat()
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    filtered = {name: vb._filter_excluded(r, manager) for name, (r, _m) in per_source.items()}
    records = filtered.get("amcr", []) + filtered.get("teater", [])
    nested, audit, _c = vb._nest(manager, records, qualifiers=manager.qualifier_overrides())
    rows = vr.facet_census_rows(nested, audit, manager)

    for row in rows:
        in_audit = {
            str(a.get("placed_by") or "").rsplit(":", 2)[0]
            if str(a.get("placed_by") or "").startswith("override:")
            else str(a.get("placed_by") or "")
            for a in audit
            if a.get("theme") == row["facet"]
        }
        assert row["feeding_rules"] == len(in_audit)
        for chunk in filter(None, row["top_rules"].split("; ")):
            assert chunk.rsplit(" (", 1)[0] in in_audit


def test_census_charges_group_headers_like_the_budget_sheet_does():
    """Both sheets bill the same ~120 header lines, so a `flat` census is strictly
    cheaper than a `facet_sub` one. If only one of them charged headers, the census
    would under-report the vocabulary's real prompt cost."""
    _require_flat()
    nested = json.loads((VOCAB_DIR / "union_nested.json").read_text(encoding="utf-8"))
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    filtered = {name: vb._filter_excluded(r, manager) for name, (r, _m) in per_source.items()}
    records = filtered.get("amcr", []) + filtered.get("teater", [])
    _n, audit, _c = vb._nest(manager, records, qualifiers=manager.qualifier_overrides())

    grouped = vr.facet_census_rows(
        nested, audit, manager, prompt_config={"PROMPT_VOCAB_GROUPING": "facet_sub"}
    )
    flat = vr.facet_census_rows(
        nested, audit, manager, prompt_config={"PROMPT_VOCAB_GROUPING": "flat"}
    )
    assert flat[-1]["cumulative_tokens"] < grouped[-1]["cumulative_tokens"]


def test_census_is_read_only():
    """Every sheet in this module is evidence, never an edit."""
    _require_flat()
    config = REPO_ROOT / "data_samples" / "taxonomy_config.json"
    before = config.read_bytes()
    _census()
    assert config.read_bytes() == before


def test_census_counts_same_as_against_the_vocabulary_as_shipped():
    """Regression: the census first shipped reporting with_same_as = 0 for every facet
    while union_nested.json held 169 linked terms. `attach_same_as` runs in
    vocab_build.main, not in `_nest`, so a report built off a bare `_nest` counts none —
    and this is the A1-facets evidence sheet, telling a reviewer no facet contains a
    composite-linked term."""
    _require_flat()
    rows, nested = _census()
    shipped = sum(
        1
        for f, terms in nested.items()
        if not f.startswith("_")
        for e in terms.values()
        if e.get("same_as")
    )
    assert shipped > 0, "the committed artifact carries same_as; the fixture is wrong"
    assert sum(r["with_same_as"] for r in rows) == shipped


def test_nest_as_built_matches_the_committed_artifact_on_same_as():
    """The helper's whole justification: nesting the way the build nests. If it drifts
    from vocab_build.main, every sheet that reads a linked field is quietly wrong."""
    _require_flat()
    manager = _shipped_manager()
    per_source = vb._load_flat(VOCAB_DIR)
    filtered = {name: vb._filter_excluded(r, manager) for name, (r, _m) in per_source.items()}
    records = filtered.get("amcr", []) + filtered.get("teater", [])
    nested, _audit = vr._nest_as_built(manager, records)
    committed = json.loads((VOCAB_DIR / "union_nested.json").read_text(encoding="utf-8"))

    fresh = {
        (f, cs)
        for f, terms in nested.items()
        if not f.startswith("_")
        for cs, e in terms.items()
        if e.get("same_as")
    }
    shipped = {
        (f, cs)
        for f, terms in committed.items()
        if not f.startswith("_")
        for cs, e in terms.items()
        if e.get("same_as")
    }
    assert fresh == shipped
