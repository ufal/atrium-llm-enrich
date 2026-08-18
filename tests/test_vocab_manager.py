"""
tests/test_vocab_manager.py – Unit tests for vocab_manager.py's
VocabularyManager: config/taxonomy loading, keyword theme assignment, injectable
LLM classification, and vocab persistence. Network paths (AMCR OAI-PMH sync) are
never exercised — every test provides an on-disk vocab file or a mock predictor.
"""

import json

from vocab_manager import VocabularyManager

TAXONOMY = {
    "Site Types": {"priority": 10, "keywords": {"cs": ["hrad", "mohyla"]}},
    "Find Types": {"priority": 8, "keywords": {"cs": ["keramika", "nůž"]}},
}


def _mgr(tmp_path, taxonomy=TAXONOMY, llm_predictor=None):
    cfg = tmp_path / "taxonomy.json"
    cfg.write_text(json.dumps(taxonomy), encoding="utf-8")
    vocab = tmp_path / "vocab.json"
    return VocabularyManager(str(vocab), str(cfg), llm_predictor=llm_predictor)


# ── config loading ──────────────────────────────────────────────────────────
def test_load_config_from_file(tmp_path):
    assert "Site Types" in _mgr(tmp_path).taxonomy


def test_load_config_missing_uses_builtin_default(tmp_path):
    m = VocabularyManager(str(tmp_path / "v.json"), str(tmp_path / "nope.json"))
    assert "Site Types" in m.taxonomy  # built-in default taxonomy


# ── _assign_theme ───────────────────────────────────────────────────────────
def test_assign_theme_matches_keyword(tmp_path):
    m = _mgr(tmp_path)
    assert m._assign_theme({"cs": "starý hrad"}) == "Site Types"
    assert m._assign_theme({"cs": "zdobená keramika"}) == "Find Types"


def test_assign_theme_no_match_is_other(tmp_path):
    assert _mgr(tmp_path)._assign_theme({"cs": "nesmysl"}) == "Other"


def test_assign_theme_higher_priority_wins(tmp_path):
    m = _mgr(tmp_path)
    # contains both a Find (pri 8) and Site (pri 10) keyword → higher priority wins
    assert m._assign_theme({"cs": "hrad s keramikou"}) == "Site Types"


# ── classify_with_llm (injectable predictor) ────────────────────────────────
def test_classify_with_llm_none_without_predictor(tmp_path):
    assert _mgr(tmp_path).classify_with_llm({"cs": "x"}) is None


def test_classify_with_llm_returns_matched_category(tmp_path):
    m = _mgr(tmp_path, llm_predictor=lambda prompt: "Find Types")
    assert m.classify_with_llm({"cs": "x", "en": "y"}) == "Find Types"


def test_classify_with_llm_unmatched_returns_none(tmp_path):
    m = _mgr(tmp_path, llm_predictor=lambda prompt: "Nonexistent")
    assert m.classify_with_llm({"cs": "x"}) is None


def test_classify_with_llm_swallows_predictor_error(tmp_path):
    def boom(prompt):
        raise RuntimeError("llm down")

    assert _mgr(tmp_path, llm_predictor=boom).classify_with_llm({"cs": "x"}) is None


# ── persistence + stats ─────────────────────────────────────────────────────
def test_save_and_load_round_trip(tmp_path):
    m = _mgr(tmp_path)
    m.vocab_data = {"Site Types": {"hrad": {"en": "castle"}}}
    m.save()

    loaded = _mgr(tmp_path).load()
    assert loaded == {"Site Types": {"hrad": {"en": "castle"}}}


def test_vocab_statistics_counts_terms_per_theme(tmp_path):
    m = _mgr(tmp_path)
    m.vocab_data = {"Site Types": {"a": 1, "b": 2}, "Find Types": {"c": 3}}
    assert m.vocab_statistics() == {"Site Types": 2, "Find Types": 1}


def test_get_prompt_string_is_cached(tmp_path):
    m = _mgr(tmp_path)
    m.vocab_data = {"X": {"a": 1}}
    first = m.get_prompt_string()
    assert first is m.get_prompt_string()  # served from cache
    m._invalidate_cache()
    assert m.get_prompt_string() == first  # rebuilt to equal content


# ── build_nested: the pure nesting stage ────────────────────────────────────
NESTED_TAXONOMY = {
    "_settings": {
        "tie_break": ["Site Types", "Find Types"],
        "heslar_map": {"obdobi": "Chronology", "zeme": "__exclude__"},
        "teater_branch_map": {"1050": "Chronology", "3549": "__exclude__"},
    },
    "Site Types": {"priority": 10, "keywords": {"cs": ["hrad"]}},
    "Find Types": {"priority": 8, "keywords": {"cs": ["keramika"]}},
    "Chronology": {"priority": 9, "keywords": {"cs": ["eneolit"]}},
    "Other": {"priority": -1, "in_prompt": False, "keywords": {}},
}


def _pair(cs, en="x", **kw):
    return {"cs": cs, "en": en, **kw}


def _nesting_mgr(tmp_path):
    return _mgr(tmp_path, taxonomy=NESTED_TAXONOMY)


def test_build_nested_is_pure_and_repeatable(tmp_path):
    m = _nesting_mgr(tmp_path)
    terms = {"starý hrad": _pair("starý hrad"), "nesmysl": _pair("nesmysl")}
    first = m.build_nested(terms)
    second = m.build_nested(terms)
    assert first == second
    # no disk write, no self-mutation
    assert m.vocab_data == {}
    assert not (tmp_path / "vocab.json").exists()


def test_build_nested_theme_order_is_priority_descending(tmp_path):
    """Load-bearing: build_system_prompt iterates this dict in insertion order and
    truncates a prefix, so key order decides which themes survive a tight budget."""
    built = _nesting_mgr(tmp_path).build_nested({})
    assert list(built) == ["Site Types", "Chronology", "Find Types", "Other"]


def test_heslar_membership_beats_keyword_match(tmp_path):
    m = _nesting_mgr(tmp_path)
    # "hradiště eneolitické" matches the Site Types keyword, but the term came from the
    # AMCR obdobi list, which is curated and wins.
    term = _pair("hrad eneolitický", source="amcr", scheme="obdobi")
    assert m.assign_theme(term) == ("Chronology", "heslar:obdobi")


def test_teater_branch_beats_keyword_match(tmp_path):
    m = _nesting_mgr(tmp_path)
    term = _pair("hrad", source="teater", source_id="9", broader=["1050", "1070"])
    assert m.assign_theme(term) == ("Chronology", "teater:1050")


def test_excluded_schemes_are_dropped_entirely(tmp_path):
    """Countries are a legitimate AMCR list and not a topic of interest. Excluding them
    by list membership is what finally keeps them out, regardless of which keyword their
    name happens to collide with."""
    m = _nesting_mgr(tmp_path)
    built = m.build_nested({"Zimbabwe": _pair("Zimbabwe", source="amcr", scheme="zeme")})
    assert all("Zimbabwe" not in terms for terms in built.values())


def test_rescue_places_terms_the_keyword_matcher_missed(tmp_path):
    m = _nesting_mgr(tmp_path)
    terms = {"magdalénien": _pair("magdalénien", source="amcr", scheme="unmapped")}
    assert "magdalénien" in m.build_nested(terms)["Other"]
    rescued = m.build_nested(terms, rescue={"magdalénien": "Chronology"})
    assert "magdalénien" in rescued["Chronology"]
    assert rescued["Other"] == {}


def test_audit_records_the_rule_behind_every_placement(tmp_path):
    m = _nesting_mgr(tmp_path)
    audit = []
    m.build_nested(
        {
            "starý hrad": _pair("starý hrad"),
            "magdalénien": _pair("magdalénien", source="amcr", scheme="obdobi"),
            "nesmysl": _pair("nesmysl"),
        },
        audit=audit,
    )
    assert {r["cs"]: r["placed_by"] for r in audit} == {
        "starý hrad": "keyword:cs:hrad",
        "magdalénien": "heslar:obdobi",
        "nesmysl": "other",
    }


def test_keep_limits_the_on_disk_entry_shape(tmp_path):
    m = _nesting_mgr(tmp_path)
    rich = _pair("starý hrad", source="amcr", scheme="lokalita_typ", source_id="HES-1")
    built = m.build_nested({"starý hrad": rich})
    assert built["Site Types"]["starý hrad"] == {"cs": "starý hrad", "en": "x"}
    full = m.build_nested({"starý hrad": rich}, keep=None)
    assert full["Site Types"]["starý hrad"]["scheme"] == "lokalita_typ"


def test_settings_block_is_not_a_theme(tmp_path):
    m = _nesting_mgr(tmp_path)
    assert "_settings" not in m.themes()
    assert "_settings" not in m.build_nested({})


def test_tie_break_resolves_equal_priorities(tmp_path):
    tied = {
        "_settings": {"tie_break": ["Find Types", "Site Types"]},
        "Site Types": {"priority": 5, "keywords": {"cs": ["objekt"]}},
        "Find Types": {"priority": 5, "keywords": {"cs": ["objekt"]}},
    }
    m = _mgr(tmp_path, taxonomy=tied)
    assert m.assign_theme({"cs": "objekt"})[0] == "Find Types"


def test_load_without_auto_sync_refuses_to_harvest(tmp_path):
    import pytest

    m = VocabularyManager(str(tmp_path / "missing.json"), str(tmp_path / "cfg.json"))
    with pytest.raises(FileNotFoundError, match="vocab_build.py"):
        m.load(auto_sync=False)


# ── facet taxonomy: depth-2 branches, labels, validation ────────────────────
FACET_TAXONOMY = {
    "_settings": {
        "tie_break": ["Chronology", "Artefact"],
        "heslar_map": {"obdobi": "Chronology", "zeme": "__exclude__"},
        "heslar_labels": {"obdobi": "období"},
        "teater_branch_map": {
            "1050": "Chronology",
            "2557": "Documentation",
            "2560": "__exclude__",
        },
    },
    "Chronology": {"priority": 9, "keywords": {}},
    "Artefact": {"priority": 6, "keywords": {}},
    "Documentation": {"priority": 2, "keywords": {}},
}


def _facet_mgr(tmp_path):
    return _mgr(tmp_path, taxonomy=FACET_TAXONOMY)


def test_teater_depth_two_branch_overrides_its_parent(tmp_path):
    """Branch 2557 mixes ethnic groups with historical regions. Resolving the broader
    chain most-specific-first is what lets one sub-branch be dropped while its siblings
    are kept, without a code change."""
    m = _facet_mgr(tmp_path)
    kept = _pair("Achájové", source="teater", source_id="9", broader=["2557", "2900"])
    dropped = _pair("Avaři", source="teater", source_id="8", broader=["2557", "2560"])
    assert m.assign_theme(kept) == ("Documentation", "teater:2557")
    assert m.assign_theme(dropped) == ("__exclude__", "teater:2560")


def test_teater_branch_map_can_key_on_the_node_itself(tmp_path):
    m = _facet_mgr(tmp_path)
    assert m.assign_theme(_pair("x", source="teater", source_id="1050"))[1] == "teater:1050"


def test_amcr_heslar_gets_a_readable_subtheme_label(tmp_path):
    """The prompt sub-header should read "období (kategorie)", not "obdobi_kat"."""
    m = _facet_mgr(tmp_path)
    built = m.build_nested(
        {"eneolit": _pair("eneolit", source="amcr", scheme="obdobi", sub="obdobi")}
    )
    assert built["Chronology"]["eneolit"]["sub"] == "období"


def test_unlabelled_subtheme_passes_through(tmp_path):
    """TEATER's depth-2 labels are already human-readable and must not be mangled."""
    m = _facet_mgr(tmp_path)
    term = _pair(
        "magdalénien", source="teater", source_id="9", broader=["1050"], sub="periodizace dějin"
    )
    built = m.build_nested({"magdalénien": term})
    assert built["Chronology"]["magdalénien"]["sub"] == "periodizace dějin"


def test_a_map_pointing_at_an_undeclared_facet_is_a_hard_error(tmp_path):
    """build_nested used to create any theme a map named, silently — so a typo produced
    a phantom facet with no priority and last place in the truncation order."""
    import pytest

    broken = json.loads(json.dumps(FACET_TAXONOMY))
    broken["_settings"]["heslar_map"]["obdobi"] = "Chronlogy"  # typo
    m = _mgr(tmp_path, taxonomy=broken)
    with pytest.raises(ValueError, match="undeclared facets"):
        m.build_nested({})


def test_exclude_is_always_a_valid_map_target(tmp_path):
    _facet_mgr(tmp_path).validate_settings()  # must not raise


# ── the shipped config, not a synthetic one ─────────────────────────────────
def _repo_root():
    from pathlib import Path

    return Path(__file__).resolve().parent.parent


def _shipped():
    return VocabularyManager(
        vocab_path=str(_repo_root() / "data_samples" / "vocab" / "union_nested.json"),
        config_path=str(_repo_root() / "data_samples" / "taxonomy_config.json"),
    )


def test_shipped_config_maps_only_to_declared_facets():
    _shipped().validate_settings()  # must not raise


def test_shipped_config_covers_every_source_list():
    """A new AMCR heslar or TEATER branch appearing upstream must fail here rather than
    silently landing in Other. Nothing else in the suite reads the real config."""
    import csv

    m = _shipped()
    audits = {
        "heslar_map": "amcr_placement_audit.csv",
        "teater_branch_map": "teater_placement_audit.csv",
    }
    for map_name, filename in audits.items():
        path = _repo_root() / "data_samples" / "vocab" / filename
        if not path.exists():  # artifacts are optional in a bare checkout
            continue
        with open(path, encoding="utf-8") as fh:
            schemes = {row["scheme"] for row in csv.DictReader(fh) if row["scheme"]}
        unmapped = schemes - set(m.settings.get(map_name) or {})
        assert not unmapped, f"{map_name} does not cover: {sorted(unmapped)}"


def test_shipped_vocabulary_has_no_unplaced_terms():
    """Other empty is the invariant that says every term was placed by an explicit rule
    rather than falling through to a bucket the prompt then hides."""
    m = _shipped()
    if not m.vocab_path.exists():
        return
    assert m.load(auto_sync=False).get("Other", {}) == {}
