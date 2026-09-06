"""
tests/test_vocab_manager.py – Unit tests for vocab_manager.py's
VocabularyManager: config/taxonomy loading, keyword theme assignment, injectable
LLM classification, and vocab persistence. Network paths (AMCR OAI-PMH sync) are
never exercised — every test provides an on-disk vocab file or a mock predictor.
"""

import json
from pathlib import Path

from vocab_manager import VocabularyManager, attach_same_as, find_composite_links

REPO_ROOT = Path(__file__).resolve().parent.parent

TAXONOMY = {
    "Site Types": {"priority": 10, "keywords": {"cs": ["hrad", "mohyla"]}},
    "Find Types": {"priority": 8, "keywords": {"cs": ["keramika", "nůž"]}},
}


def _mgr(tmp_path, taxonomy=TAXONOMY, llm_predictor=None, overrides=None):
    cfg = tmp_path / "taxonomy.json"
    cfg.write_text(json.dumps(taxonomy), encoding="utf-8")
    vocab = tmp_path / "vocab.json"
    if overrides is not None:
        (tmp_path / "taxonomy_overrides.json").write_text(
            json.dumps({"overrides": overrides}), encoding="utf-8"
        )
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
    # default keep: cs/en/sub plus the id-tracking fields build_system_prompt needs for
    # teater_category_ids (source/source_id/discarded_ids) and B3 stripping (bare_cs) —
    # not "scheme", which is only a placement input, never surfaced. discarded_ids and
    # bare_cs are absent here because this hand-built pair (unlike a real
    # to_term_pairs() output) never sets them.
    assert built["Site Types"]["starý hrad"] == {
        "cs": "starý hrad",
        "en": "x",
        "source": "amcr",
        "source_id": "HES-1",
    }
    full = m.build_nested({"starý hrad": rich}, keep=None)
    assert full["Site Types"]["starý hrad"]["scheme"] == "lokalita_typ"


def test_nested_keep_comes_from_the_config(tmp_path):
    """Which harvested keys reach an on-disk entry decides the prompt payload, so it
    belongs in the file a curator edits — not only in a function default."""
    taxonomy = json.loads(json.dumps(NESTED_TAXONOMY))
    taxonomy["_settings"]["nested_keep"] = ["cs", "en", "scheme"]
    m = _mgr(tmp_path, taxonomy=taxonomy)
    rich = _pair("starý hrad", source="amcr", scheme="lokalita_typ", source_id="HES-1")
    built = m.build_nested({"starý hrad": rich})
    assert built["Site Types"]["starý hrad"] == {
        "cs": "starý hrad",
        "en": "x",
        "scheme": "lokalita_typ",
    }


def test_an_explicit_keep_still_beats_the_config(tmp_path):
    """The config supplies the default, it does not take the parameter away — the
    single-source and union builds must stay free to ask for a different shape."""
    taxonomy = json.loads(json.dumps(NESTED_TAXONOMY))
    taxonomy["_settings"]["nested_keep"] = ["cs", "en", "scheme"]
    m = _mgr(tmp_path, taxonomy=taxonomy)
    rich = _pair("starý hrad", source="amcr", scheme="lokalita_typ", source_id="HES-1")
    assert m.build_nested({"starý hrad": rich}, keep=("cs",))["Site Types"]["starý hrad"] == {
        "cs": "starý hrad"
    }
    full = m.build_nested({"starý hrad": rich}, keep=None)  # None still means "every key"
    assert full["Site Types"]["starý hrad"]["source_id"] == "HES-1"


def test_admin_stop_words_come_from_the_config(tmp_path):
    """Terms containing a stop word sort to the BACK of their facet and prompt
    truncation drops a suffix — so this list decides what a small-context model still
    sees, and which words read as boilerplate is a per-corpus judgement about report
    language. Changing it must reorder the facet."""
    terms = {"hrad zpráva": _pair("hrad zpráva"), "zámecký hrad": _pair("zámecký hrad")}

    # default: "zpráva" is boilerplate, so "hrad zpráva" goes last
    assert list(_nesting_mgr(tmp_path).build_nested(terms)["Site Types"]) == [
        "zámecký hrad",
        "hrad zpráva",
    ]

    taxonomy = json.loads(json.dumps(NESTED_TAXONOMY))
    taxonomy["_settings"]["admin_stop_words"] = ["zámecký"]
    m = _mgr(tmp_path, taxonomy=taxonomy)
    assert m.admin_stop_words() == frozenset({"zámecký"})
    assert list(m.build_nested(terms)["Site Types"]) == ["hrad zpráva", "zámecký hrad"]


def test_an_empty_stop_word_list_falls_back_to_the_default(tmp_path):
    """An absent or empty key means "unset", not "no boilerplate at all" — the same
    reading every other _settings key gets, so deleting a line cannot silently change
    what the prompt truncates."""
    taxonomy = json.loads(json.dumps(NESTED_TAXONOMY))
    taxonomy["_settings"]["admin_stop_words"] = []
    from vocab_manager import ADMIN_STOP_WORDS

    assert _mgr(tmp_path, taxonomy=taxonomy).admin_stop_words() == ADMIN_STOP_WORDS
    assert _nesting_mgr(tmp_path).admin_stop_words() == ADMIN_STOP_WORDS


def test_shipped_config_states_the_defaults_it_relies_on():
    """nested_keep and admin_stop_words are in the shipped file so a curator can see
    and edit them; they must state exactly what the code default is, or the file
    documents one behaviour while the build does another."""
    from vocab_manager import ADMIN_STOP_WORDS, DEFAULT_NESTED_KEEP

    m = VocabularyManager(config_path=str(_repo_root() / "data_samples" / "taxonomy_config.json"))
    assert "nested_keep" in m.settings and m.nested_keep() == DEFAULT_NESTED_KEEP
    assert "admin_stop_words" in m.settings and m.admin_stop_words() == ADMIN_STOP_WORDS
    assert m.composite_separators() == ("/",)


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
    with pytest.raises(ValueError, match="undeclared facet 'Chronlogy'"):
        m.build_nested({})


def test_exclude_is_always_a_valid_map_target(tmp_path):
    _facet_mgr(tmp_path).validate_settings()  # must not raise


# ── taxonomy_overrides.json: per-term facet/qualifier corrections (B4) ──────────


def _facet_mgr_with_overrides(tmp_path, overrides):
    return _mgr(tmp_path, taxonomy=FACET_TAXONOMY, overrides=overrides)


def test_override_facet_wins_over_heslar_map(tmp_path):
    """kostel/kaple/mlýn/cesta: a single mis-sorted record inside an otherwise-correct
    list, fixed per-id rather than by touching the list's own placement."""
    m = _facet_mgr_with_overrides(
        tmp_path,
        [{"match": {"source": "amcr", "id": "HES-1"}, "facet": "Artefact", "reason": "x"}],
    )
    term = _pair("hrad eneolitický", source="amcr", scheme="obdobi", source_id="HES-1")
    assert m.assign_theme(term) == ("Artefact", "override:amcr:HES-1")
    # a different id from the same list is unaffected
    other = _pair("jiný", source="amcr", scheme="obdobi", source_id="HES-2")
    assert m.assign_theme(other) == ("Chronology", "heslar:obdobi")


def test_override_without_facet_does_not_change_placement(tmp_path):
    """A qualifier-only override (B3) is consumed by to_term_pairs, not assign_theme —
    the record still resolves to its normal facet via heslar_map/teater_branch_map."""
    m = _facet_mgr_with_overrides(
        tmp_path,
        [{"match": {"source": "teater", "id": "9"}, "qualifier_cs": "sídlo elity", "reason": "x"}],
    )
    term = _pair("zámek", source="teater", source_id="9", broader=["1050"])
    assert m.assign_theme(term) == ("Chronology", "teater:1050")


def test_qualifier_overrides_extracts_only_qualifier_entries(tmp_path):
    m = _facet_mgr_with_overrides(
        tmp_path,
        [
            {
                "match": {"source": "teater", "id": "1439"},
                "qualifier_cs": "sídlo elity",
                "reason": "x",
            },
            {"match": {"source": "amcr", "id": "HES-1"}, "facet": "Artefact", "reason": "x"},
        ],
    )
    assert m.qualifier_overrides() == {("teater", "1439"): "sídlo elity"}


def test_is_excluded_ignores_keyword_fallback_and_rescue(tmp_path):
    """is_excluded must answer purely from the curated placement rules, since it gates
    which records even reach dedup (finding 3) — a keyword or rescue match is an aid
    for an unclassified term, never a reason to treat a term as excluded."""
    m = _facet_mgr(tmp_path)
    excluded = _pair("Avaři", source="teater", source_id="8", broader=["2557", "2560"])
    kept = _pair("Achájové", source="teater", source_id="9", broader=["2557", "2900"])
    unmatched = _pair("nesmysl")
    assert m.is_excluded(excluded) is True
    assert m.is_excluded(kept) is False
    assert m.is_excluded(unmatched) is False  # falls through to Other, not excluded


def test_override_facet_can_itself_be_exclude(tmp_path):
    m = _facet_mgr_with_overrides(
        tmp_path,
        [{"match": {"source": "amcr", "id": "HES-1"}, "facet": "__exclude__", "reason": "x"}],
    )
    term = _pair("x", source="amcr", scheme="obdobi", source_id="HES-1")
    assert m.is_excluded(term) is True


def test_override_sub_relabels_the_moved_terms_header(tmp_path):
    """A `facet` override alone leaves the record under its *old* list's sub-header —
    `kostel` rendered as "Feature / areál aktivity", the facet a reviewer asked for
    under the header they moved it away from. `sub` closes that, and must land
    identically in the nested entry and the audit row so the CSV shows what the prompt
    renders."""
    m = _facet_mgr_with_overrides(
        tmp_path,
        [
            {
                "match": {"source": "amcr", "id": "HES-1"},
                "facet": "Artefact",
                "sub": "druh předmětu",
                "reason": "x",
            }
        ],
    )
    audit = []
    nested = m.build_nested(
        {
            "kostel": _pair(
                "kostel", source="amcr", scheme="obdobi", source_id="HES-1", sub="obdobi"
            )
        },
        audit=audit,
    )
    assert nested["Artefact"]["kostel"]["sub"] == "druh předmětu"
    assert audit[0]["sub"] == "druh předmětu"
    assert audit[0]["scheme"] == "obdobi"  # the raw list is still recorded, unrelabelled


def test_sub_falls_back_to_heslar_labels_without_an_override(tmp_path):
    """The override is an exception, not the path: an unoverridden record still gets the
    heslar_labels relabelling ('obdobi' -> 'období') it always had."""
    m = _facet_mgr(tmp_path)
    audit = []
    nested = m.build_nested(
        {
            "eneolit": _pair(
                "eneolit", source="amcr", scheme="obdobi", source_id="HES-9", sub="obdobi"
            )
        },
        audit=audit,
    )
    assert nested["Chronology"]["eneolit"]["sub"] == "období"
    assert audit[0]["sub"] == "období"


def test_override_sub_can_supply_a_header_the_source_omits(tmp_path):
    """TEATER records carry no heslar scheme, so a moved one has no sub-header at all.
    Assigning (rather than relabelling in place) lets `sub` fill that gap."""
    m = _facet_mgr_with_overrides(
        tmp_path,
        [
            {
                "match": {"source": "teater", "id": "7"},
                "facet": "Artefact",
                "sub": "druh předmětu",
                "reason": "x",
            }
        ],
    )
    nested = m.build_nested({"x": _pair("x", source="teater", source_id="7")})
    assert nested["Artefact"]["x"]["sub"] == "druh předmětu"


def test_override_sub_is_still_bound_by_keep(tmp_path):
    """`keep` is the on-disk entry contract (test_keep_limits_the_on_disk_entry_shape).
    An override must not smuggle a key past it — a caller that excluded `sub` gets no
    `sub`, override or not."""
    m = _facet_mgr_with_overrides(
        tmp_path,
        [
            {
                "match": {"source": "amcr", "id": "HES-1"},
                "facet": "Artefact",
                "sub": "druh předmětu",
                "reason": "x",
            }
        ],
    )
    nested = m.build_nested(
        {"x": _pair("x", source="amcr", scheme="obdobi", source_id="HES-1", sub="obdobi")},
        keep=("cs", "en"),
    )
    assert nested["Artefact"]["x"] == {"cs": "x", "en": "x"}


def test_override_facet_must_be_a_declared_theme(tmp_path):
    import pytest

    m = _facet_mgr_with_overrides(
        tmp_path,
        [{"match": {"source": "amcr", "id": "HES-1"}, "facet": "Nonexistent", "reason": "x"}],
    )
    with pytest.raises(ValueError, match="overrides"):
        m.validate_settings()


# ── _exclusions: keys, back-compat, and the review status (issue #6, O3/O4) ──────


def _exclusions_mgr(tmp_path, exclusions):
    taxonomy = json.loads(json.dumps(FACET_TAXONOMY))
    taxonomy["_settings"]["_exclusions"] = exclusions
    return _mgr(tmp_path, taxonomy=taxonomy)


def test_exclusion_notes_are_keyed_by_the_rule_that_excluded_the_term(tmp_path):
    """`assign_theme` reports `heslar:zeme` / `teater:2560`; `_exclusions` is keyed the
    same way, so a report looks a note up by the rule rather than by a second key
    convention that has to be kept in step with it."""
    m = _exclusions_mgr(
        tmp_path,
        {
            "heslar:zeme": {"status": "open_geo_ethnic", "reason": "country names"},
            "teater:2560": {"status": "open_geo_ethnic", "reason": "ethnonyms"},
        },
    )
    notes = m.exclusion_notes()
    assert notes["heslar:zeme"] == {"status": "open_geo_ethnic", "reason": "country names"}
    assert notes["teater:2560"]["reason"] == "ethnonyms"

    term = _pair("Albánie", source="amcr", scheme="zeme", source_id="HES-9")
    _theme, rule = m.assign_theme(term)
    assert rule in notes, "the rule assign_theme reports must be the key a note is under"


def test_legacy_exclusion_key_spellings_still_resolve(tmp_path):
    """The file shipped with bare AMCR list names and `TEATER 288`. Renaming the keys is
    not a reason to invalidate someone's local edit, so both spellings still fold onto
    the canonical rule string."""
    m = _exclusions_mgr(
        tmp_path,
        {"zeme": "country names", "TEATER 288": "neighbouring disciplines", "2560": "ethnonyms"},
    )
    notes = m.exclusion_notes()
    assert set(notes) == {"heslar:zeme", "teater:288", "teater:2560"}


def test_a_bare_reason_string_means_already_ruled_on(tmp_path):
    """Back-compat with a value shape, not just a key: a note written before statuses
    existed meant "already ruled on", which is what it must keep meaning."""
    m = _exclusions_mgr(tmp_path, {"heslar:zeme": "country names"})
    assert m.exclusion_notes()["heslar:zeme"] == {
        "status": "settled",
        "reason": "country names",
    }


def test_an_entry_without_a_status_defaults_to_settled(tmp_path):
    m = _exclusions_mgr(tmp_path, {"heslar:zeme": {"reason": "country names"}})
    assert m.exclusion_notes()["heslar:zeme"]["status"] == "settled"


# ── vocabulary provenance for paradata (issue #6, D3) ───────────────────────────


def _write_vocab_with_meta(tmp_path, meta):
    (tmp_path / "union_nested.json").write_text("{}", encoding="utf-8")
    (tmp_path / "union_nested.meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )
    return str(tmp_path / "union_nested.json")


def test_provenance_reads_the_sidecar_beside_the_artifact(tmp_path):
    """vocab_build.py has always written this file and nothing has ever read it, so an
    enrichment run recorded *that* it used a vocabulary but never *which* one."""
    from vocab_manager import vocabulary_provenance

    path = _write_vocab_with_meta(
        tmp_path,
        {
            "tool_version": "v0.20.0",
            "counts": {"total": 2074},
            "taxonomy_config": {"sha256": "aaa"},
            "taxonomy_overrides": {"sha256": "bbb"},
            "sources": [
                {"name": "amcr", "records": 1460, "license": "CC BY-NC 4.0", "endpoint": "oai"},
                {
                    "name": "teater",
                    "records": 4134,
                    "license": "CC BY-NC 4.0",
                    "snapshot_ref": "2106c59",
                },
            ],
        },
    )
    prov = vocabulary_provenance(path)
    assert prov["vocab"]["tool_version"] == "v0.20.0"
    assert prov["vocab"]["terms"] == 2074
    assert prov["taxonomy"] == {"config_sha256": "aaa", "overrides_sha256": "bbb"}
    # TEATER pins a commit; AMCR has no equivalent, so the endpoint stands in for one.
    assert prov["sources"]["teater"]["ref"] == "2106c59"
    assert prov["sources"]["amcr"]["ref"] == "oai"


def test_provenance_names_the_components_the_build_actually_used(tmp_path):
    """An AMCR-only artifact must not claim it used TEATER data — the licence block is
    an assertion about what a run depended on, not a list of what exists."""
    from vocab_manager import vocabulary_provenance

    path = _write_vocab_with_meta(
        tmp_path, {"sources": [{"name": "amcr", "records": 1460, "license": "CC BY-NC 4.0"}]}
    )
    assert vocabulary_provenance(path)["components"] == ["amcr_vocab"]


def test_provenance_is_empty_without_a_sidecar(tmp_path):
    """The legacy flat vocabulary and an auto-synced one have no sidecar. A run must
    not fail over missing provenance."""
    from vocab_manager import vocabulary_provenance

    (tmp_path / "legacy.json").write_text("{}", encoding="utf-8")
    assert vocabulary_provenance(str(tmp_path / "legacy.json")) == {}
    assert vocabulary_provenance(str(tmp_path / "absent.json")) == {}


def test_shipped_vocabulary_declares_both_non_commercial_sources():
    from vocab_manager import vocabulary_provenance

    prov = vocabulary_provenance(str(_repo_root() / "data_samples" / "vocab" / "union_nested.json"))
    assert prov["components"] == ["amcr_vocab", "teater_data"]
    assert {s["license"] for s in prov["sources"].values()} == {"CC BY-NC 4.0"}
    assert prov["taxonomy"]["config_sha256"] and prov["taxonomy"]["overrides_sha256"]


def test_every_paradata_component_name_is_declared_in_para_config():
    """log_component() falls back to "UNKNOWN" for a name para_config.txt does not
    declare — silently, and the run's effective licence is then computed without it.
    A typo here would under-report a CC BY-NC 4.0 dependency."""
    from vocab_manager import PARADATA_COMPONENTS

    text = (_repo_root() / "para_config.txt").read_text(encoding="utf-8")
    declared = {
        line.split("=", 1)[0].strip()
        for line in text.splitlines()
        if "=" in line and not line.strip().startswith("#")
    }
    assert set(PARADATA_COMPONENTS.values()) <= declared


# ── geo guardrail: one decision, two files, kept in step (issue #6, O4 / C1) ─────


def _guard_mgr(tmp_path, **guard):
    taxonomy = json.loads(json.dumps(FACET_TAXONOMY))
    taxonomy["_settings"]["geo_guardrail"] = {
        "active": True,
        "covers": ["teater:2560"],
        **guard,
    }
    return _mgr(tmp_path, taxonomy=taxonomy)


WITH_CLAUSE = "NEVER select a country name, language name, or geographic region name"
WITHOUT_CLAUSE = "Select the SINGLE most relevant category."


def test_the_shipped_config_and_the_shipped_prompt_agree():
    """The invariant the build gate exists to hold: nothing in the repo today offers a
    term the prompt forbids the model from selecting. Reads the guardrail through
    prompt_template rather than grepping llm_run.py — the wording is a config-selected
    block now, so the only honest source is the one the runtime renders from."""
    import prompt_template

    root = _repo_root()
    m = VocabularyManager(config_path=str(root / "data_samples" / "taxonomy_config.json"))
    config = prompt_template.load_run_config(root / "llm_config.txt")
    assert m.geo_guardrail_problems(prompt_template.guardrail_text(config)) == []


def test_reinstating_a_covered_branch_while_the_clause_stands_is_a_problem(tmp_path):
    """The failure mode the O3/O4 package warns about: the vocabulary offers ethnonyms,
    the prompt forbids selecting one, and the score measures the contradiction."""
    m = _guard_mgr(tmp_path)
    assert m.geo_guardrail_problems(WITH_CLAUSE) == []  # 2560 is excluded in the fixture

    m.taxonomy["_settings"]["teater_branch_map"]["2560"] = "Chronology"
    problems = m.geo_guardrail_problems(WITH_CLAUSE)
    assert len(problems) == 1
    assert "teater:2560 is offered to the model" in problems[0]


def test_relaxing_the_clause_without_saying_so_is_a_problem(tmp_path):
    """Dropping the wording quietly leaves five branches excluded for a reason that no
    longer exists — the vocabulary would be needlessly poorer and nothing would say
    why."""
    m = _guard_mgr(tmp_path)
    problems = m.geo_guardrail_problems(WITHOUT_CLAUSE)
    assert len(problems) == 1
    assert "the prompt no longer forbids" in problems[0]


def test_declaring_the_guardrail_off_while_the_prompt_still_carries_it_is_a_problem(tmp_path):
    m = _guard_mgr(tmp_path, active=False)
    problems = m.geo_guardrail_problems(WITH_CLAUSE)
    assert len(problems) == 1
    assert "the prompt still forbids" in problems[0]


def test_both_halves_flipped_together_is_the_clean_path(tmp_path):
    """The one outcome the package asks for: relax the wording, declare it relaxed, and
    reinstate the branch — all in the same change, and the build stays happy."""
    m = _guard_mgr(tmp_path, active=False)
    m.taxonomy["_settings"]["teater_branch_map"]["2560"] = "Chronology"
    assert m.geo_guardrail_problems(WITHOUT_CLAUSE) == []


def test_the_prompt_half_is_skipped_when_there_is_no_prompt_to_read(tmp_path):
    """These artifacts are copied into atrium-llm-enrich, where the prompt lives
    elsewhere; the vocabulary half must still be checked there."""
    m = _guard_mgr(tmp_path)
    assert m.geo_guardrail_problems(None) == []
    m.taxonomy["_settings"]["teater_branch_map"]["2560"] = "Chronology"
    assert len(m.geo_guardrail_problems(None)) == 1


def test_prompt_markers_are_config_not_code(tmp_path):
    """A reworded prompt must be an edit to the config, not a patch to the detector."""
    m = _guard_mgr(tmp_path, prompt_markers=["nikdy nevybírej název země"])
    assert m.geo_guardrail_problems("... nikdy nevybírej název země ...") == []
    assert len(m.geo_guardrail_problems(WITH_CLAUSE)) == 1


def test_an_armed_guardrail_with_an_empty_scope_is_refused(tmp_path):
    """`geo_guardrail_problems` loops over `covers`; an empty list makes the gate pass
    whatever the maps say. M11 relaxed the guardrail and emptied `covers` in the same
    change, which left re-arming it — the "retire Q1 if problems arise" move M11
    explicitly deferred — completely unguarded: `active: true` plus the strict wording
    plus all 982 geographic terms still offered produced zero complaints."""
    import pytest

    taxonomy = json.loads(json.dumps(FACET_TAXONOMY))
    taxonomy["_settings"]["geo_guardrail"] = {
        "active": True,
        "prompt_markers": ["country name"],
        "covers": [],
    }
    m = _mgr(tmp_path, taxonomy=taxonomy)
    with pytest.raises(ValueError) as excinfo:
        m.validate_settings()
    assert "covers is empty" in str(excinfo.value)
    assert "check nothing" in str(excinfo.value)


def test_a_relaxed_guardrail_may_keep_its_scope_listed(tmp_path):
    """The complement, and the shipped shape: `covers` names the branches that ARE
    geographic, which does not stop being true when they are reinstated. Keeping the
    list populated while `active` is false must stay legal — that is what makes the
    gate work the moment anyone re-arms it."""
    taxonomy = json.loads(json.dumps(FACET_TAXONOMY))
    taxonomy["_settings"]["teater_branch_map"]["2560"] = "Chronology"  # reinstated
    taxonomy["_settings"]["geo_guardrail"] = {
        "active": False,
        "prompt_markers": ["country name"],
        "covers": ["teater:2560"],
    }
    m = _mgr(tmp_path, taxonomy=taxonomy)
    m.validate_settings()
    m.settings["geo_guardrail"]["active"] = True
    problems = m.geo_guardrail_problems("NEVER select a country name")
    assert any("teater:2560" in p for p in problems), "re-arming must report the conflict"


def test_the_shipped_config_can_rearm_its_own_guardrail(tmp_path):
    """End to end on the real config, because this is the move M11 deferred: flipping
    `active` back on while the branches are still offered must name every one of them."""
    manager = VocabularyManager(
        config_path=str(REPO_ROOT / "data_samples" / "taxonomy_config.json")
    )
    manager.validate_settings()
    assert manager.geo_guardrail()["covers"], "covers must survive reinstatement"
    manager.settings["geo_guardrail"]["active"] = True
    problems = manager.geo_guardrail_problems(
        "NEVER select a country name, language name, or geographic region name"
    )
    assert len(problems) == len(manager.geo_guardrail()["covers"])


def test_guardrail_scope_and_the_exclusion_register_cannot_drift(tmp_path):
    """`covers` outlives an exclusion (it is what says a reinstated branch conflicts at
    all), so the two lists are cross-checked rather than merged."""
    import pytest

    taxonomy = json.loads(json.dumps(FACET_TAXONOMY))
    taxonomy["_settings"]["_exclusions"] = {
        "teater:2560": {"status": "open_geo_ethnic", "reason": "ethnonyms"}
    }
    taxonomy["_settings"]["geo_guardrail"] = {"active": True, "covers": []}
    with pytest.raises(ValueError, match="geo_guardrail.covers does not list it"):
        _mgr(tmp_path, taxonomy=taxonomy).validate_settings()

    taxonomy["_settings"]["geo_guardrail"]["covers"] = ["teater:2560", "teater:9999"]
    with pytest.raises(ValueError, match="which no map places"):
        _mgr(tmp_path, taxonomy=taxonomy).validate_settings()

    taxonomy["_settings"]["geo_guardrail"]["covers"] = ["teater:2560"]
    taxonomy["_settings"]["_exclusions"]["teater:2560"]["status"] = "open_other"
    with pytest.raises(ValueError, match="not 'open_geo_ethnic'"):
        _mgr(tmp_path, taxonomy=taxonomy).validate_settings()


# ── validate_settings: every hand-editable surface, reported at once ─────────────


def _broken(tmp_path, mutate, overrides=None):
    import pytest

    taxonomy = json.loads(json.dumps(FACET_TAXONOMY))
    mutate(taxonomy["_settings"])
    m = _mgr(tmp_path, taxonomy=taxonomy, overrides=overrides)
    with pytest.raises(ValueError) as excinfo:
        m.validate_settings()
    return str(excinfo.value)


def test_tie_break_naming_an_undeclared_facet_is_caught(tmp_path):
    """tie_break decides prompt truncation order; a name nothing declares is simply
    never consulted, so the ordering silently is not the one that was written."""
    message = _broken(tmp_path, lambda s: s["tie_break"].append("Chronlogy"))
    assert "tie_break lists undeclared facet 'Chronlogy'" in message


def test_facets_sharing_a_priority_must_be_ordered_in_tie_break(tmp_path):
    """Render order is truncation order, so where a facet sits among its
    equal-priority peers decides whether a 32k model sees it at all. A facet the
    tie_break omits is appended after every listed one by the len(tie_break) fallback
    — a real decision, made by nobody. This is the shape A1-facets takes: splitting the
    probation facet in two and leaving the new half unordered.
    """

    def mutate(settings):
        settings["teater_branch_map"]["3094"] = "Society"

    import pytest

    taxonomy = json.loads(json.dumps(FACET_TAXONOMY))
    taxonomy["Society"] = {"priority": 6, "keywords": {}}  # ties with Artefact
    mutate(taxonomy["_settings"])
    m = _mgr(tmp_path, taxonomy=taxonomy)
    with pytest.raises(ValueError) as excinfo:
        m.validate_settings()
    message = str(excinfo.value)
    assert "share priority 6" in message
    assert "'Society'" in message
    assert "tie_break" in message


def test_a_unique_priority_needs_no_tie_break_entry(tmp_path):
    """The guard must not force every facet into the list — only the ambiguous ones.
    Documentation sits alone at priority 2 and is absent from tie_break in the shipped
    fixture; that is unambiguous and must stay legal."""
    taxonomy = json.loads(json.dumps(FACET_TAXONOMY))
    assert "Documentation" not in taxonomy["_settings"]["tie_break"]
    _mgr(tmp_path, taxonomy=taxonomy).validate_settings()


def test_listing_both_tied_facets_satisfies_the_guard(tmp_path):
    """And the fix the message asks for actually works."""
    taxonomy = json.loads(json.dumps(FACET_TAXONOMY))
    taxonomy["Society"] = {"priority": 6, "keywords": {}}
    taxonomy["_settings"]["teater_branch_map"]["3094"] = "Society"
    taxonomy["_settings"]["tie_break"].append("Society")
    m = _mgr(tmp_path, taxonomy=taxonomy)
    m.validate_settings()
    order = m._theme_order()
    assert order.index("Artefact") < order.index("Society")


def test_a_label_for_an_unmapped_list_is_caught(tmp_path):
    """heslar_labels renames a list for the prompt sub-header. A key naming a list no
    map places is dead: the relabel never renders and nothing says so."""
    message = _broken(tmp_path, lambda s: s["heslar_labels"].update({"obdoby": "období"}))
    assert "heslar_labels['obdoby'] names nothing in heslar_map" in message


def test_a_reason_for_something_nobody_excludes_is_caught(tmp_path):
    """An _exclusions note whose rule is not `__exclude__` reads as a live exclusion
    to anyone auditing the file, while excluding nothing — the exact inverse of the
    traceability M7 asked for."""
    message = _broken(tmp_path, lambda s: s.update({"_exclusions": {"heslar:obdobi": "x"}}))
    assert "documents a rule that excludes nothing" in message


def test_an_unknown_exclusion_status_is_caught(tmp_path):
    message = _broken(
        tmp_path,
        lambda s: s.update({"_exclusions": {"heslar:zeme": {"status": "maybe", "reason": "x"}}}),
    )
    assert "status 'maybe' is not one of" in message


def test_an_unknown_override_key_is_caught(tmp_path):
    """The likeliest hand-edit mistake of all: a plausible key nobody reads. 'sub_cs'
    for 'sub', 'facets' for 'facet' — each does nothing, silently."""
    message = _broken(
        tmp_path,
        lambda s: None,
        overrides=[{"match": {"source": "amcr", "id": "HES-1"}, "sub_cs": "x", "reason": "y"}],
    )
    assert "unknown key 'sub_cs'" in message


def test_a_malformed_same_as_entry_is_caught(tmp_path):
    message = _broken(
        tmp_path,
        lambda s: None,
        overrides=[
            {
                "match": {"source": "amcr", "id": "HES-1"},
                "same_as": [{"source": "teater"}],  # no id
                "reason": "y",
            }
        ],
    )
    assert "needs both 'source' and 'id'" in message


def test_a_pair_both_linked_and_suppressed_is_caught(tmp_path):
    """attach_same_as resolves this the safe way (no link), but silently — and the two
    entries cannot both be what their author meant."""
    message = _broken(
        tmp_path,
        lambda s: None,
        overrides=[
            {
                "match": {"source": "amcr", "id": "HES-1"},
                "same_as": [{"source": "teater", "id": "9"}],
                "same_as_suppress": [{"source": "teater", "id": "9"}],
                "reason": "y",
            }
        ],
    )
    assert "both name the pair" in message


def test_a_stale_override_is_only_reported_when_records_are_supplied(tmp_path):
    """The manager is routinely used with no harvest at all (prompt building, the
    single-source builds), where "no such record" is normal — so this check is opt-in
    rather than something that would fire on every load."""
    import pytest

    m = _facet_mgr_with_overrides(
        tmp_path,
        [{"match": {"source": "amcr", "id": "HES-GONE"}, "facet": "Artefact", "reason": "y"}],
    )
    m.validate_settings()  # no records → not an error

    live = [{"source": "amcr", "source_id": "HES-1"}]
    with pytest.raises(ValueError, match="matches no harvested record"):
        m.validate_settings(records=live)

    m.validate_settings(records=live + [{"source": "amcr", "source_id": "HES-GONE"}])


def test_two_records_sharing_a_qualifier_is_caught(tmp_path):
    """@david-spacil, working the M8 review (issue #6, comment 5541507280): "two records
    sharing a qualifier build the same key, and the second one is then dropped without
    even keeping its id." He read it right — `to_term_pairs` appends the loser to the
    collisions *warning* list and skips it, so the id reaches no `discarded_ids`. His
    convention (qualify only the record that leaves the group) is now a rule."""
    import pytest

    m = _facet_mgr_with_overrides(
        tmp_path,
        [
            {"match": {"source": "teater", "id": "1"}, "qualifier_cs": "q", "reason": "x"},
            {"match": {"source": "teater", "id": "2"}, "qualifier_cs": "q", "reason": "x"},
        ],
    )
    records = [
        {"cs": "x", "source": "teater", "source_id": "1"},
        {"cs": "x", "source": "teater", "source_id": "2"},
        {"cs": "x", "source": "amcr", "source_id": "A1"},
    ]
    m.validate_settings()  # not knowable without the records
    with pytest.raises(ValueError, match="would build the same key"):
        m.validate_settings(records=records)


def test_distinct_qualifiers_in_one_group_are_fine(tmp_path):
    """The safe shape of the same edit: two homonyms split apart, each keeping its id."""
    m = _facet_mgr_with_overrides(
        tmp_path,
        [
            {"match": {"source": "teater", "id": "1"}, "qualifier_cs": "q1", "reason": "x"},
            {"match": {"source": "teater", "id": "2"}, "qualifier_cs": "q2", "reason": "x"},
        ],
    )
    m.validate_settings(
        records=[
            {"cs": "x", "source": "teater", "source_id": "1"},
            {"cs": "x", "source": "teater", "source_id": "2"},
            {"cs": "x", "source": "amcr", "source_id": "A1"},
        ]
    )


def test_qualifying_every_member_of_a_group_is_caught(tmp_path):
    """The mirror fault: qualify them all and the bare label is offered by nobody —
    which no reviewer splitting a homonym intends."""
    import pytest

    m = _facet_mgr_with_overrides(
        tmp_path,
        [
            {"match": {"source": "teater", "id": "1"}, "qualifier_cs": "q1", "reason": "x"},
            {"match": {"source": "teater", "id": "2"}, "qualifier_cs": "q2", "reason": "x"},
        ],
    )
    with pytest.raises(ValueError, match="bare label would not be offered"):
        m.validate_settings(
            records=[
                {"cs": "x", "source": "teater", "source_id": "1"},
                {"cs": "x", "source": "teater", "source_id": "2"},
            ]
        )


def test_a_qualifier_shared_across_DIFFERENT_groups_is_fine(tmp_path):
    """Two of the shipped verdicts both use "materiál" (`vejce`, `rostlinné
    makrozbytty`). Different labels build different keys, so this must not be flagged —
    the check is per group, not global."""
    m = _facet_mgr_with_overrides(
        tmp_path,
        [
            {"match": {"source": "amcr", "id": "A1"}, "qualifier_cs": "materiál", "reason": "x"},
            {"match": {"source": "amcr", "id": "B1"}, "qualifier_cs": "materiál", "reason": "x"},
        ],
    )
    m.validate_settings(
        records=[
            {"cs": "vejce", "source": "amcr", "source_id": "A1"},
            {"cs": "vejce", "source": "amcr", "source_id": "A2"},
            {"cs": "rostlinné makrozbytky", "source": "amcr", "source_id": "B1"},
            {"cs": "rostlinné makrozbytky", "source": "amcr", "source_id": "B2"},
        ]
    )


def test_every_problem_is_reported_at_once(tmp_path):
    """One typo per rebuild is how a config file stops being worth editing."""
    message = _broken(
        tmp_path,
        lambda s: (
            s["tie_break"].append("Chronlogy"),
            s["heslar_labels"].update({"obdoby": "období"}),
        ),
        overrides=[{"match": {"source": "amcr", "id": "HES-1"}, "sub_cs": "x", "reason": "y"}],
    )
    assert "tie_break lists undeclared facet" in message
    assert "names nothing in heslar_map" in message
    assert "unknown key 'sub_cs'" in message


def test_the_shipped_config_survives_the_widened_validation():
    """Every check above is a real invariant of the shipped files, not just of a
    fixture — including the stale-override check against the real harvest."""
    root = _repo_root()
    m = VocabularyManager(config_path=str(root / "data_samples" / "taxonomy_config.json"))
    m.validate_settings()

    import pytest

    import vocab_build as vb

    vocab_dir = root / "data_samples" / "vocab"
    if not (vocab_dir / "amcr_flat.json").exists():
        pytest.skip("flat artifacts not present in this checkout")
    records = [r.as_dict() for recs, _meta in vb._load_flat(vocab_dir).values() for r in recs]
    m.validate_settings(records=records)


def test_missing_overrides_file_is_not_an_error(tmp_path):
    m = _facet_mgr(tmp_path)  # no overrides= passed, no file written
    assert m.overrides == {}
    assert m.qualifier_overrides() == {}
    m.validate_settings()  # must not raise


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


def test_shipped_exclusions_document_every_excluded_list_and_branch():
    """Every `__exclude__` in the shipped maps must have a note, and every note must
    name a rule that actually excludes something — a reason for a list nobody excludes
    is dead config, and an exclusion with no reason is the thing M7 exists to prevent."""
    m = VocabularyManager(config_path=str(_repo_root() / "data_samples" / "taxonomy_config.json"))
    settings = m.settings
    excluded_rules = {
        f"heslar:{k}" for k, v in (settings.get("heslar_map") or {}).items() if v == "__exclude__"
    } | {
        f"teater:{k}"
        for k, v in (settings.get("teater_branch_map") or {}).items()
        if v == "__exclude__"
    }
    notes = m.exclusion_notes()
    assert set(notes) == excluded_rules
    assert all(note["reason"] for note in notes.values())


def test_shipped_exclusion_statuses_are_all_declared():
    from vocab_manager import EXCLUSION_STATUSES

    m = VocabularyManager(config_path=str(_repo_root() / "data_samples" / "taxonomy_config.json"))
    assert {n["status"] for n in m.exclusion_notes().values()} <= set(EXCLUSION_STATUSES)


def test_shipped_vocabulary_has_no_unplaced_terms():
    """Other empty is the invariant that says every term was placed by an explicit rule
    rather than falling through to a bucket the prompt then hides."""
    m = _shipped()
    if not m.vocab_path.exists():
        return
    assert m.load(auto_sync=False).get("Other", {}) == {}


# ── composite links / same_as (issue #6, O1 / F) ────────────────────────────────

NESTED_WITH_COMPOSITE = {
    "Activity Area": {
        "most/brod": {"cs": "most/brod", "en": "bridge/ford", "source": "amcr", "source_id": "A1"},
    },
    "Feature": {
        "most": {"cs": "most", "en": "bridge", "source": "amcr", "source_id": "A2"},
        "hrad": {"cs": "hrad", "en": "castle", "source": "teater", "source_id": "T9"},
    },
}


def _nested_copy():
    return json.loads(json.dumps(NESTED_WITH_COMPOSITE))


def test_find_composite_links_pairs_a_composite_with_its_standalone_component():
    links = find_composite_links(_nested_copy())
    assert links == [("Activity Area", "most/brod", "Feature", "most")]


def test_find_composite_links_ignores_a_component_nobody_offers():
    """ "brod" is not a standalone entry here, so only the "most" half is a pair —
    a composite whose components are all unoffered produces nothing at all."""
    nested = _nested_copy()
    del nested["Feature"]["most"]
    assert find_composite_links(nested) == []


def test_find_composite_links_never_pairs_a_label_with_itself():
    for _f1, cs1, _f2, cs2 in find_composite_links(_nested_copy()):
        assert cs1 != cs2


def test_attach_same_as_links_both_directions():
    """A scorer may hold either side as the gold label, so the link has to be
    findable from either entry without a second lookup table."""
    nested = _nested_copy()
    assert attach_same_as(nested) == 1

    composite = nested["Activity Area"]["most/brod"]
    component = nested["Feature"]["most"]
    assert composite["same_as"] == [{"source": "amcr", "id": "A2"}]
    assert component["same_as"] == [{"source": "amcr", "id": "A1"}]
    # an unrelated term is untouched
    assert "same_as" not in nested["Feature"]["hrad"]


def test_attach_same_as_is_idempotent():
    """vocab_build runs this on every build; running it twice must not double the
    links (the committed artifact would then differ from a fresh build and trip
    the --from-flat --check drift gate)."""
    nested = _nested_copy()
    attach_same_as(nested)
    first = json.loads(json.dumps(nested))
    assert attach_same_as(nested) == 0
    assert nested == first


def test_attach_same_as_is_a_noop_without_any_composite_overlap():
    nested = {"Feature": {"most": {"cs": "most", "en": "bridge", "source": "a", "source_id": "1"}}}
    assert attach_same_as(nested) == 0
    assert "same_as" not in nested["Feature"]["most"]


def test_composite_separators_come_from_the_config(tmp_path):
    """Only "/" is in use, but both sources are hand-maintained: a second convention
    appearing in a later harvest must be a config edit, not a code change."""
    nested = {
        "Feature": {
            "most-brod": {"cs": "most-brod", "en": "x", "source": "amcr", "source_id": "A1"},
            "most": {"cs": "most", "en": "bridge", "source": "amcr", "source_id": "A2"},
        }
    }
    assert find_composite_links(nested) == []  # "-" is not a separator by default
    assert find_composite_links(nested, ["-"]) == [("Feature", "most-brod", "Feature", "most")]

    taxonomy = json.loads(json.dumps(FACET_TAXONOMY))
    assert _mgr(tmp_path, taxonomy=taxonomy).composite_separators() == ("/",)
    taxonomy["_settings"]["composite_separators"] = ["/", " - "]
    assert _mgr(tmp_path, taxonomy=taxonomy).composite_separators() == ("/", " - ")


def test_same_as_overrides_reads_both_directions_of_correction(tmp_path):
    m = _facet_mgr_with_overrides(
        tmp_path,
        [
            {
                "match": {"source": "amcr", "id": "A1"},
                "same_as": [{"source": "teater", "id": "T9"}],
                "same_as_suppress": [{"source": "amcr", "id": "A2"}],
                "reason": "x",
            }
        ],
    )
    extra, suppress = m.same_as_overrides()
    assert extra == [[("amcr", "A1"), ("teater", "T9")]]
    assert suppress == [[("amcr", "A1"), ("amcr", "A2")]]


def test_extra_links_two_records_no_label_connects():
    """`hrad` and `most` share no label at all — only a reviewer can say they belong
    together, so the mechanism has to accept a link the detector cannot see."""
    nested = _nested_copy()
    links = attach_same_as(nested, extra=[[("amcr", "A2"), ("teater", "T9")]])
    assert links == 2  # the auto most/brod<->most pair, plus the declared one
    assert {"source": "teater", "id": "T9"} in nested["Feature"]["most"]["same_as"]
    assert {"source": "amcr", "id": "A2"} in nested["Feature"]["hrad"]["same_as"]


def test_suppress_drops_a_detected_link():
    nested = _nested_copy()
    assert attach_same_as(nested, suppress=[[("amcr", "A1"), ("amcr", "A2")]]) == 0
    assert "same_as" not in nested["Activity Area"]["most/brod"]
    assert "same_as" not in nested["Feature"]["most"]


def test_suppress_wins_over_extra():
    """Contradictory config resolves the safe way — no link — rather than by which
    argument the caller passed first."""
    nested = _nested_copy()
    pair = [("amcr", "A2"), ("teater", "T9")]
    assert attach_same_as(nested, extra=[pair], suppress=[pair]) == 1  # only the auto pair
    assert "same_as" not in nested["Feature"]["hrad"]


def test_extra_naming_a_record_this_build_does_not_offer_is_skipped():
    """One overrides file drives the AMCR-only, TEATER-only and union builds, so a
    declared link whose other side is absent is normal, not an error."""
    nested = _nested_copy()
    assert attach_same_as(nested, extra=[[("amcr", "A2"), ("teater", "GONE")]]) == 1
    assert nested["Feature"]["most"]["same_as"] == [{"source": "amcr", "id": "A1"}]


def test_extra_is_order_independent():
    """A set is a valid argument and the two sides of a pair are interchangeable —
    neither may change the bytes the build writes."""
    forward, backward = _nested_copy(), _nested_copy()
    attach_same_as(forward, extra=[[("amcr", "A2"), ("teater", "T9")]])
    attach_same_as(backward, extra={frozenset({("teater", "T9"), ("amcr", "A2")})})
    assert forward == backward


def test_shipped_same_as_baseline_is_unchanged():
    """169 terms carrying 102 links, and the number has moved twice for stated reasons:
    the reviewed baseline was 162/98; M13's `pastvina/louka (nálezové okolnosti)` split
    added one (the qualified label still contains "/", so it is still a composite of the
    same components); M11's reinstatement added three more, since composites and their
    components can now both be offered across the two new context facets. The
    configurable extra/suppress path must not move it further while no override uses
    it."""
    import json as _json

    nested = _json.loads(
        (_repo_root() / "data_samples" / "vocab" / "union_nested.json").read_text(encoding="utf-8")
    )
    linked = [e for terms in nested.values() for e in terms.values() if e.get("same_as")]
    assert len(linked) == 169
    assert sum(len(e["same_as"]) for e in linked) == 2 * 102


def test_shipped_vocabulary_same_as_links_are_symmetric():
    """Every same_as link in the committed artifact must be answered by one pointing
    back — a one-way link would score correct in one direction only, depending on
    which side the annotator happened to write down as gold."""
    m = _shipped()
    if not m.vocab_path.exists():
        return
    nested = m.load(auto_sync=False)

    by_id = {}
    for terms in nested.values():
        if not isinstance(terms, dict):
            continue
        for entry in terms.values():
            if isinstance(entry, dict) and entry.get("source_id"):
                by_id[(entry.get("source"), entry["source_id"])] = entry

    for terms in nested.values():
        if not isinstance(terms, dict):
            continue
        for cs, entry in terms.items():
            if not isinstance(entry, dict):
                continue
            for link in entry.get("same_as") or []:
                other = by_id.get((link["source"], link["id"]))
                assert other is not None, f"{cs}: same_as points at a term not offered"
                back = {"source": entry.get("source"), "id": entry.get("source_id")}
                assert back in (other.get("same_as") or []), f"{cs}: one-way same_as link"


# ── the module entry point (issue #6, gap 8) ─────────────────────────────────────


def test_running_vocab_manager_as_a_script_never_writes_the_vocabulary():
    """`python3 vocab_manager.py` used to call the legacy AMCR-only sync against the
    shipped union artifact and save the result. Those records carry only cs/en, so every
    one landed in `Other` — which is `in_prompt: false`, leaving the next pipeline run
    injecting an empty vocabulary. CI caught it as artifact drift; a curator running the
    module to look at the vocabulary did not. The entry point inspects and nothing else.

    Asserted on the source rather than by running it: importing under `__main__` is not
    something a test can do cleanly, and the property at stake is "this code path cannot
    write", which the absence of the call is exactly what establishes.
    """
    src = (Path(__file__).resolve().parent.parent / "vocab_manager.py").read_text(encoding="utf-8")
    main_block = src.split('if __name__ == "__main__":', 1)[1]
    # comments explain the old behaviour by name; only executable lines are the claim
    code = "\n".join(line for line in main_block.splitlines() if not line.lstrip().startswith("#"))

    assert "sync_and_build_nested_taxonomy" not in code, (
        "the entry point rebuilds the vocabulary again — it must only read it"
    )
    assert ".save()" not in code, "the entry point writes"
    assert "auto_sync=False" in code, (
        "load() must be told not to fall back to a harvest, or a missing artifact "
        "triggers the very rebuild this guard exists to prevent"
    )
