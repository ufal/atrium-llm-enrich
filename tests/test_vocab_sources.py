"""
tests/test_vocab_sources.py — offline unit tests for the vocabulary harvesters.

Every test here is offline: ``requests.Session`` is replaced by a stub that serves
recorded fixtures, so the suite never touches api.aiscr.cz or teater.aiscr.cz. That
mirrors the contract already stated in tests/test_vocab_manager.py.
"""

import json
from pathlib import Path

import pytest

import vocab_sources as vs

FIXTURES = Path(__file__).parent / "fixtures" / "vocab"


class _Response:
    def __init__(self, content: bytes = b"", payload=None):
        self.content = content
        self._payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload


class _StubSession:
    """Serves queued responses and records the URLs it was asked for."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.urls = []
        self.headers = {}

    def get(self, url, **kwargs):
        self.urls.append(url)
        if not self._responses:
            raise AssertionError(f"unexpected extra request: {url}")
        return self._responses.pop(0)


def _xml(name: str) -> _Response:
    return _Response(content=(FIXTURES / name).read_bytes())


# ── AMCR ─────────────────────────────────────────────────────────────────────


def test_amcr_follows_resumption_token_and_stops():
    session = _StubSession([_xml("amcr_oai_page1.xml"), _xml("amcr_oai_page2.xml")])
    records, meta = vs.harvest_amcr(delay=0, session=session)

    assert len(session.urls) == 2
    # The token contains an ampersand and must be percent-encoded, or the second
    # request silently truncates into a different query.
    assert "resumptionToken=tok%261" in session.urls[1]
    assert meta["pages"] == 2
    assert meta["records"] == len(records) == 4


def test_amcr_captures_the_whole_heslo_record():
    session = _StubSession([_xml("amcr_oai_page1.xml"), _xml("amcr_oai_page2.xml")])
    records, _ = vs.harvest_amcr(delay=0, session=session)
    by_id = {r.source_id: r for r in records}

    term = by_id["HES-001065"]
    assert term.cs == "terénní zásah"
    assert term.en == "field intervention"
    assert term.scheme == "akce_typ_kat"  # the field that dissolves most of "Other"
    assert term.broader == ("HES-000001",)
    assert term.sort == 30
    assert term.abbr == "tz"
    assert term.note_cs == "zásah v terénu"
    assert term.uri == "https://api.aiscr.cz/id/HES-001065"
    # only skos:exactMatch is promoted; broadMatch is not an identity claim
    assert term.exact_match == ("http://vocab.getty.edu/aat/300077463",)


def test_amcr_keeps_terms_without_an_english_gloss():
    """The flat archive records them; only the nesting stage drops them.

    The previous harvester discarded EN-less terms silently, so nobody could tell how
    many there were.
    """
    session = _StubSession([_xml("amcr_oai_page1.xml"), _xml("amcr_oai_page2.xml")])
    records, meta = vs.harvest_amcr(delay=0, session=session)

    without_en = [r for r in records if r.en is None]
    assert [r.cs for r in without_en] == ["bezejmenný předmět"]
    assert meta["without_en"] == 1
    assert "bezejmenný předmět" not in vs.to_term_pairs(records)


def test_amcr_network_error_returns_a_partial_harvest():
    import requests

    class _Boom(_StubSession):
        def get(self, url, **kwargs):
            self.urls.append(url)
            raise requests.RequestException("connection reset")

    records, meta = vs.harvest_amcr(delay=0, session=_Boom([]))
    assert records == []
    assert "network error" in meta["error"]


def test_amcr_max_pages_guard_trips():
    session = _StubSession([_xml("amcr_oai_page1.xml")] * 10)
    _, meta = vs.harvest_amcr(delay=0, session=session, max_pages=3)
    assert meta["pages"] == 3
    assert len(session.urls) == 3


def test_amcr_parser_does_not_expand_external_entities():
    """The hardened lxml parser is the reason this module exists rather than reusing
    ``xml.etree.ElementTree.fromstring``, which resolves entities by default."""
    session = _StubSession([_xml("amcr_xxe.xml")])
    records, meta = vs.harvest_amcr(delay=0, session=session)
    assert not any("root:" in (r.cs or "") for r in records)
    assert records == [] or all(r.cs != "pwned" for r in records)


# ── TEATER ───────────────────────────────────────────────────────────────────


def _teater_session():
    payload = json.loads((FIXTURES / "teater_import_sample.json").read_text(encoding="utf-8"))
    # the snapshot walk reads 12 files; serve the sample once and empty lists after
    return _StubSession([_Response(payload=payload)] + [_Response(payload=[]) for _ in range(11)])


def test_teater_snapshot_flattens_the_tree():
    records, meta = vs.harvest_teater(mode="snapshot", session=_teater_session())
    by_id = {r.source_id: r for r in records}

    assert meta["strategy"] == "snapshot"
    assert set(by_id) == {"1050", "1090", "3549", "3552"}

    mag = by_id["1090"]
    assert (mag.cs, mag.en, mag.de) == ("magdalénien", "magdalenian", "Magdalenien")
    assert mag.uri == "https://teater.aiscr.cz/id/1090"
    assert mag.broader == ("1050",)
    assert mag.scheme == "1050"  # the top-level branch, what teater_branch_map keys on
    assert mag.note_cs == "kulturní celek paleolitu"
    assert mag.note_en == "a palaeolithic culture"
    # the concept's own label is not repeated as one of its own alt labels
    assert mag.alt_cs == ("magdalenien",)


def test_teater_roots_are_their_own_branch():
    records, _ = vs.harvest_teater(mode="snapshot", session=_teater_session())
    root = next(r for r in records if r.source_id == "1050")
    assert root.broader == ()
    assert root.scheme == "1050"


def test_teater_carries_the_depth_two_label_as_sub():
    """TEATER's second level is the granularity the 7-theme rollup discards — 17 kinds
    of activity area, 18 kinds of feature. It rides along on every descendant."""
    records, _ = vs.harvest_teater(mode="snapshot", session=_teater_session())
    by_id = {r.source_id: r for r in records}
    assert by_id["1090"].sub == "magdalénien"  # a depth-2 node names itself
    assert by_id["1050"].sub is None  # a depth-1 root is a section title, not a group


def test_teater_branch_roots_never_become_selectable_terms():
    """ "5) Chronologie" and "12) Společnost" are numbered section titles. They belong in
    the flat archive for completeness but must not reach the model as categories."""
    records, _ = vs.harvest_teater(mode="snapshot", session=_teater_session())
    assert {r.source_id for r in records} >= {"1050", "3549"}

    pairs = vs.to_term_pairs(records)
    assert "5) Chronologie" not in pairs
    assert "12) Společnost" not in pairs
    assert "magdalénien" in pairs
    assert pairs["magdalénien"]["sub"] == "magdalénien"


def test_teater_live_falls_back_to_the_snapshot():
    import requests

    class _FailingLive(_StubSession):
        def __init__(self, inner):
            super().__init__([])
            self._inner = inner
            self._first = True

        def get(self, url, **kwargs):
            if self._first:
                self._first = False
                raise requests.RequestException("503")
            return self._inner.get(url, **kwargs)

    records, meta = vs.harvest_teater(mode="live", session=_FailingLive(_teater_session()))
    assert meta["strategy"] == "snapshot"
    assert records


# ── normalisation, merge, serialisation ──────────────────────────────────────


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("  Hradiště  ", "hradiště"),
        ("3D –\xa0VRML (*.wrl)", "3d – vrml (*.wrl)"),  # NBSP really occurs in the data
        ("Pohřebiště\n/hřbitov", "pohřebiště /hřbitov"),
        (None, ""),
    ],
)
def test_norm_label(raw, expected):
    assert vs.norm_label(raw) == expected


def test_record_sort_key_is_numeric_on_source_id():
    """String ordering puts "3552" before "562", which silently changed which of two
    same-labelled concepts survived a round-trip through the flat file."""
    small = vs.VocabRecord(cs="a", source="teater", source_id="562")
    large = vs.VocabRecord(cs="b", source="teater", source_id="3552")
    assert sorted([large, small], key=vs.record_sort_key) == [small, large]


def test_merge_prefers_amcr_and_counts_collisions():
    amcr = [vs.VocabRecord(cs="most", en="bridge", source="amcr", source_id="HES-1")]
    teater = [vs.VocabRecord(cs="Most", en="bridge (teater)", source="teater", source_id="1")]
    merged, collisions = vs.merge({"amcr": (amcr, {}), "teater": (teater, {})})
    assert [r.source for r in merged] == ["amcr"]
    assert collisions == 1


def test_to_term_pairs_resolves_label_collisions_deterministically():
    """Duplicate labels cannot both survive: the nested dict is keyed by the label and
    build_schema turns the list into an Enum, where duplicate values become aliases."""
    a = vs.VocabRecord(
        cs="komunita", en="community", source="teater", source_id="562", broader=("288",)
    )
    b = vs.VocabRecord(
        cs="komunita",
        en="community (sociology)",
        source="teater",
        source_id="3552",
        broader=("3549",),
    )
    collisions = []
    forward = vs.to_term_pairs([a, b], collisions=collisions)
    backward = vs.to_term_pairs([b, a], collisions=[])
    assert forward == backward
    assert forward["komunita"]["en"] == "community"
    assert len(collisions) == 1


# ── to_term_pairs: dedup carries ids, qualifiers opt a record out (B1-B5) ────────


def test_dedup_lists_every_discarded_record_on_the_survivor():
    """M7: a same-label group that dedups still records what it absorbed, rather than
    silently dropping every record but the winner."""
    a = vs.VocabRecord(cs="kostel", en="church", source="amcr", source_id="HES-000021")
    b = vs.VocabRecord(cs="kostel", en="church", source="amcr", source_id="HES-000465")
    c = vs.VocabRecord(
        cs="kostel", en="church", source="teater", source_id="1333", broader=("1267",)
    )
    pairs = vs.to_term_pairs([a, b, c])
    assert set(pairs) == {"kostel"}
    assert pairs["kostel"]["source_id"] == "HES-000021"  # amcr sorts first
    discarded = {d["id"] for d in pairs["kostel"]["discarded_ids"]}
    assert discarded == {"HES-000465", "1333"}


def test_a_same_label_collision_dedups_by_default_even_with_different_glosses():
    """Differing EN glosses alone must never trigger a qualifier split — most of the
    515 real collisions are translation variance (e.g. 'fortress' vs 'fort'), not a
    homonym, and only a human review (taxonomy_overrides.json) can tell them apart."""
    a = vs.VocabRecord(cs="kůl", en="pole", source="amcr", source_id="HES-000737")
    b = vs.VocabRecord(cs="kůl", en="stake", source="teater", source_id="1588", broader=("1481",))
    pairs = vs.to_term_pairs([a, b])
    assert set(pairs) == {"kůl"}
    assert pairs["kůl"]["discarded_ids"][0]["id"] == "1588"


def test_qualifier_override_pulls_a_record_into_its_own_bracketed_entry():
    """B3: an explicitly flagged record (taxonomy_overrides.json) gets its own enum
    entry instead of being silently dropped by the dedup — the concrete 'zámek' case
    (lock vs. château) motyc signed off on in comment 5439363875."""
    lock_amcr = vs.VocabRecord(cs="zámek", en="lock", source="amcr", source_id="HES-000817")
    lock_teater = vs.VocabRecord(
        cs="zámek", en="lock", source="teater", source_id="2358", broader=("1788",)
    )
    chateau = vs.VocabRecord(
        cs="zámek", en="châteaux", source="teater", source_id="1439", broader=("1267",)
    )
    qualifiers = {("teater", "1439"): "sídlo elity"}
    pairs = vs.to_term_pairs([lock_amcr, lock_teater, chateau], qualifiers=qualifiers)

    assert set(pairs) == {"zámek", "zámek (sídlo elity)"}
    assert pairs["zámek"]["source_id"] == "HES-000817"
    assert [d["id"] for d in pairs["zámek"]["discarded_ids"]] == ["2358"]
    assert pairs["zámek (sídlo elity)"]["source_id"] == "1439"
    assert pairs["zámek (sídlo elity)"]["discarded_ids"] == []
    assert pairs["zámek (sídlo elity)"]["bare_cs"] == "zámek"
    assert "bare_cs" not in pairs["zámek"]


def test_qualifier_override_alone_in_its_group_still_gets_bracketed():
    """A flagged record with no competing bare entry still needs bare_cs set, so the
    output-side stripping step (llm_run.main) can find it."""
    only = vs.VocabRecord(
        cs="zámek", en="châteaux", source="teater", source_id="1439", broader=("1267",)
    )
    pairs = vs.to_term_pairs([only], qualifiers={("teater", "1439"): "sídlo elity"})
    assert set(pairs) == {"zámek (sídlo elity)"}
    assert pairs["zámek (sídlo elity)"]["bare_cs"] == "zámek"


def test_flat_json_round_trip_is_lossless(tmp_path):
    session = _StubSession([_xml("amcr_oai_page1.xml"), _xml("amcr_oai_page2.xml")])
    records, meta = vs.harvest_amcr(delay=0, session=session)

    path = tmp_path / "amcr_flat.json"
    vs.write_flat_json(records, path, meta)
    restored, restored_meta = vs.read_flat_json(path)

    assert restored == sorted(records, key=vs.record_sort_key)
    assert restored_meta == meta
    # and re-serialising the restored records is byte-identical
    assert vs.flat_json_text(restored, restored_meta) == path.read_text(encoding="utf-8")


def test_flat_csv_is_a_valid_translator_glossary(tmp_path):
    session = _StubSession([_xml("amcr_oai_page1.xml"), _xml("amcr_oai_page2.xml")])
    records, _ = vs.harvest_amcr(delay=0, session=session)

    path = tmp_path / "amcr_flat.csv"
    vs.write_flat_csv(records, path)
    # Path.read_text() applies universal newlines; CRLF here would make every
    # regeneration look like a change to --check.
    assert b"\r\n" not in path.read_bytes()

    import csv as _csv

    rows = list(_csv.reader(path.read_text(encoding="utf-8").splitlines()))
    assert rows[0][:2] == ["source_lemma", "target_translation"]
    glossary = {r[0]: r[1] for r in rows[1:] if r[1]}
    assert glossary["terénní zásah"] == "field intervention"


def test_vocabulary_csv_is_strictly_two_columns(tmp_path):
    records = [
        vs.VocabRecord(cs="most, dřevěný", en="bridge, wooden", source="amcr", source_id="1"),
        vs.VocabRecord(cs="bez glosy", en=None, source="amcr", source_id="2"),
    ]
    path = tmp_path / "vocabulary.csv"
    vs.write_vocabulary_csv(records, path)

    import csv as _csv

    rows = list(_csv.reader(path.read_text(encoding="utf-8").splitlines()))
    assert all(len(r) == 2 for r in rows)
    assert rows[1] == ["most, dřevěný", "bridge, wooden"]  # comma survives quoting
    assert len(rows) == 2  # the EN-less term is omitted from a translation glossary
