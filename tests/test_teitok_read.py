# Adjust import path assuming pytest runs from the repository root
import sys
from pathlib import Path

import pytest

_api_util_path = str(Path(__file__).parent.parent / "api_util")
if _api_util_path not in sys.path:
    sys.path.insert(0, _api_util_path)

from api_util.teitok_read import (  # noqa: E402
    doc_id_from_path,
    parse_teitok,
    read_teitok_rows,
    read_teitok_text,
    read_teitok_tokens,
)

TEITOK_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<teiCorpus>
    <text>
        <pb n="1"/>
        <s text="První věta na stránce.">
            <tok id="w-1" lemma="první" pos="ADJ" join="right">První</tok>
            <tok id="w-2" lemma="věta" pos="NOUN">věta</tok>
            <tok id="w-3" lemma="na" pos="ADP">na</tok>
            <tok id="w-4" lemma="stránka" pos="NOUN" spaceAfter="No">stránce</tok>
            <tok id="w-5" lemma="." pos="PUNCT">.</tok>
        </s>
        <lb/>
        <s>
            <tok id="w-6" lemma="druhý" pos="ADJ">Druhá</tok>
            <tok id="w-7" lemma="chybí" pos="VERB">chybí</tok>
            <tok id="w-8" lemma="text" type="attr">text</tok>
        </s>
        <pb n="2"/>
        <s text="Věta na druhé straně.">
            <tok id="w-9" lemma="věta" pos="NOUN">Věta</tok>
        </s>
    </text>
</teiCorpus>
"""


@pytest.fixture
def sample_teitok(tmp_path):
    p = tmp_path / "doc.teitok.xml"
    p.write_text(TEITOK_SAMPLE, encoding="utf-8")
    return p


def test_doc_id_from_path():
    assert doc_id_from_path("CTX001.conllu") == "CTX001"
    assert doc_id_from_path("CTX001.teitok.xml") == "CTX001"
    assert doc_id_from_path("/path/to/CTX001.txt") == "CTX001"


# ── doc_id derivation: one answer, every entry point (atrium-project#10, D3) ──

#: Every multi-dot name the pipeline actually passes around. Each must resolve to the SAME
#: doc_id everywhere, because the accretion contract is keyed on it: a stage that disagrees
#: looks for `<other_id>.document.json`, does not find it, and silently emits an orphan with
#: every upstream block discarded (that is D1/D2, in llm-enrich and in alto's service).
_MULTI_DOT_NAMES = [
    "CTX000000001.alto.xml",
    "CTX000000001.teitok.xml",
    "CTX000000001.udpipe.conllu",
    "CTX000000001.conllu",
    "CTX000000001.document.json",
    "CTX000000001.categories.json",
    "CTX000000001.csv",
    "CTX000000001.md",
]


@pytest.mark.parametrize("name", _MULTI_DOT_NAMES)
def test_doc_id_from_path_matches_canonical_doc_id(name):
    """The regression that mattered is ``.udpipe.conllu``.

    The old implementation sliced ``.conllu`` off by literal length (7 chars), so
    ``CTX000000001.udpipe.conllu`` came back as ``CTX000000001.udpipe`` while
    ``canonical_doc_id()`` — whose ``KNOWN_PIPELINE_SUFFIXES`` lists the longer
    ``.udpipe.conllu`` first, deliberately — answers ``CTX000000001``. Latent only because
    this repo's input filters never feed it a ``.conllu`` today; the function is public and
    its own docstring advertised the suffix.
    """
    from atrium_document import canonical_doc_id

    assert doc_id_from_path(name) == canonical_doc_id(name) == "CTX000000001"
    # …and the same answer wherever the path came from.
    assert doc_id_from_path(f"/archive/2026/{name}") == "CTX000000001"


@pytest.mark.parametrize("name", _MULTI_DOT_NAMES)
def test_every_doc_id_entry_point_in_this_repo_agrees(name):
    """The cross-entry-point half of D3's gate.

    Three mutually-inconsistent derivations used to coexist here: this module's literal-slice
    stripper, ``service/api.py``'s ``_doc_id()``, and a bare ``Path.stem`` in both remote
    clients. They are now one function with three call sites, and this test is what keeps the
    fourth implementation from being written.
    """
    pytest.importorskip("fastapi")  # service/api.py imports FastAPI at module level
    from atrium_document import canonical_doc_id
    from service.api import _doc_id

    assert {doc_id_from_path(name), _doc_id(name), canonical_doc_id(name)} == {"CTX000000001"}


def _core(row):
    """The keys every reader has always given; ``page_idx``/``page_label`` are additive."""
    return {k: row[k] for k in ("page_num", "line_num", "text")}


WRITER_SAMPLE = (
    Path(__file__).resolve().parent / "fixtures" / "teitok" / "writer" / "CTX000000002.teitok.xml"
)


def test_a_sentence_over_a_page_break_is_one_row_per_page():
    """nlp-enrich's format-2 writer puts a <pb/> inside an <s> that runs over a page break
    (its released sample, vendored): the reader gives the two page parts as two rows, so
    xml_to_md never unions boxes of two pages into one line."""
    rows = read_teitok_rows(WRITER_SAMPLE)
    parts = [
        r for r in rows if r["text"] in ("qpqb dbqp uunn", "Soubor nálezů byl uložen v depozitáři.")
    ]
    assert [(r["page_idx"], r["line_num"]) for r in parts] == [(3, 2), (4, 1)]
    assert sorted({r["page_idx"] for r in rows}) == [1, 2, 3, 4]


def test_read_teitok_rows(sample_teitok):
    rows = read_teitok_rows(sample_teitok)
    assert len(rows) == 3

    # Check page and line tracking
    assert _core(rows[0]) == {"page_num": 1, "line_num": 1, "text": "První věta na stránce."}
    assert (rows[0]["page_idx"], rows[0]["page_label"]) == (1, "1")

    # Check fallback text reconstruction from <tok> elements if @text is missing
    assert _core(rows[1]) == {"page_num": 1, "line_num": 2, "text": "Druhá chybí text"}

    # Line numbers restart on every page (they ran on across pages before nlp-enrich #38).
    assert _core(rows[2]) == {"page_num": 2, "line_num": 1, "text": "Věta na druhé straně."}


def test_read_teitok_text(sample_teitok):
    text = read_teitok_text(sample_teitok)
    assert text == "První věta na stránce.\nDruhá chybí text\nVěta na druhé straně."


def test_read_teitok_tokens(sample_teitok):
    tokens = read_teitok_tokens(sample_teitok)
    assert len(tokens) == 9

    # Check standard token attributes
    assert tokens[0] == {"form": "První", "lemma": "první", "upos": "ADJ", "space_after": False}
    assert tokens[1] == {"form": "věta", "lemma": "věta", "upos": "NOUN", "space_after": True}

    # Check spaceAfter="No" mapped properly
    assert tokens[3]["space_after"] is False

    # @pos is the UPOS fallback when @upos is missing (TEITOK projects often call it so);
    # @type never is -- it is the word/punctuation flag ("w"/"pc"), so it used to turn
    # every token of real TEITOK output into upos "w" or "pc".
    assert tokens[7] == {"form": "text", "lemma": "text", "upos": "", "space_after": True}


# ── Regression: <pb n="..."> with a non-numeric label (atrium-project#24 TODO ──
# ── — TEITOK-2-MD must survive archival roman-numeral front matter) ────────────

ROMAN_PB_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<TEI>
<text><body><div>
<pb n="I"/>
<s text="Titulni strana."/>
<pb n="1"/>
<s text="Skutecny obsah."/>
</div></body></text>
</TEI>
"""


def test_read_teitok_rows_survives_non_numeric_pb_n(tmp_path):
    """A <pb n="I"> (roman-numeral front matter, a real archival pattern)
    used to raise an unhandled ValueError from int(elem.get("n", ...)).
    It must not crash, and text on both sides of the bad label must still
    come through."""
    p = tmp_path / "roman.teitok.xml"
    p.write_text(ROMAN_PB_SAMPLE, encoding="utf-8")

    rows = read_teitok_rows(p)

    assert [r["text"] for r in rows] == ["Titulni strana.", "Skutecny obsah."]
    # n="I" doesn't parse. The document's first <pb> is page 1 whatever its label; a later
    # unparseable label advances the running counter. The next <pb n="1"> parses and is
    # taken at face value -- the usual archival pattern (roman front matter, then arabic
    # numbering starting at 1). Page numbers aren't guaranteed unique across a non-numeric
    # label (full roman-numeral parsing is out of scope); not crashing is the contract.
    assert rows[0]["page_num"] == 1
    assert rows[1]["page_num"] == 1


def test_read_teitok_rows_pb_missing_n(tmp_path):
    """A missing @n: the first <pb> is page 1 (flexiconv writes <pb> without @n, and a
    one-page document used to come out as page 2), later ones advance the counter."""
    p = tmp_path / "missing_n.teitok.xml"
    p.write_text(
        '<TEI><text><body><div><pb/><s text="Only page."/></div></body></text></TEI>',
        encoding="utf-8",
    )
    rows = read_teitok_rows(p)
    assert [_core(r) for r in rows] == [{"page_num": 1, "line_num": 1, "text": "Only page."}]

    p.write_text(
        '<TEI><text><body><div><pb/><s text="One."/><pb/><s text="Two."/></div></body></text></TEI>',
        encoding="utf-8",
    )
    assert [r["page_num"] for r in read_teitok_rows(p)] == [1, 2]


# ── Regression: <name>...</n> mis-close (issue #13 §D.2's own flagged ──────
# ── TEITOK-correctness item; fix_name_close_tags existed but was dead code) ─

NAME_MISCLOSE_SAMPLE = """<?xml version="1.0" encoding="UTF-8"?>
<TEI>
<text><body><div>
<pb n="1"/>
<s text="Vyzkum odhalil zaklady gotickeho kostela"><name>gotickeho kostela</n></s>
</div></body></text>
</TEI>
"""


def test_read_teitok_rows_survives_name_misclose(tmp_path):
    """<name>...</n> is not well-formed XML (mismatched close tag) and used
    to raise an unhandled xml.etree.ElementTree.ParseError. bbox_scale's
    fix_name_close_tags already existed and was unit-tested in isolation,
    but wasn't wired into any reader -- this pins that it now is."""
    p = tmp_path / "misclose.teitok.xml"
    p.write_text(NAME_MISCLOSE_SAMPLE, encoding="utf-8")

    rows = read_teitok_rows(p)
    assert [_core(r) for r in rows] == [
        {"page_num": 1, "line_num": 1, "text": "Vyzkum odhalil zaklady gotickeho kostela"}
    ]


def test_read_teitok_tokens_survives_name_misclose(tmp_path):
    """read_teitok_tokens() parses independently of read_teitok_rows() and
    needs the same repair -- it doesn't walk through <pb>, so this isolates
    the parse-level fix from the pb-parsing fix."""
    p = tmp_path / "misclose.teitok.xml"
    p.write_text(NAME_MISCLOSE_SAMPLE, encoding="utf-8")
    # Must not raise. The fixture has no <tok>, so the tokens are the sentence text split
    # on whitespace (with empty lemma/upos), as for any untokenized TEITOK document.
    assert [t["form"] for t in read_teitok_tokens(p)] == [
        "Vyzkum",
        "odhalil",
        "zaklady",
        "gotickeho",
        "kostela",
    ]


def test_parse_teitok_is_noop_for_well_formed_input(sample_teitok):
    """The repair must not alter documents that don't have the quirk --
    guards against fix_name_close_tags' regex over-firing on unrelated
    "</n>"-shaped content and silently corrupting well-formed input."""
    root = parse_teitok(sample_teitok)
    # Same content read_teitok_rows() already exercises via ET.parse directly
    # elsewhere in this file -- three <s> texts/reconstructions, two pages.
    assert [e.get("n") for e in root.iter() if e.tag.split("}")[-1] == "pb"] == ["1", "2"]


# ── Upstream TEITOK shapes: real flexiconv v0.3.10 output (fixtures shared with nlp-enrich) ──
# flexiconv writes no <s>: plain formats become <p>/<head>/<item> text, layout formats
# (PAGE XML, hOCR, ALTO) become <tok bbox> + <lb/> without sentences. Before the canonical
# reader was vendored, every such document read as zero rows here.

FLEXICONV_FIXTURES = Path(__file__).parent / "fixtures" / "teitok" / "flexiconv"


@pytest.mark.parametrize(
    "name, expected",
    [
        ("txt", ["Výzkum proběhl v Praze. Nalezeno 12 střepů.", "Druhý odstavec textu."]),
        ("md", ["Nadpis zprávy", "První odstavec důležitý.", "položka seznamu"]),
        ("page", ["Výzkum proběhl v Praze.", "Nalezeno 12 střepů."]),
        ("hocr", ["METHODIUS", "AN (ZU) SISTELIUS, UBER DEN AUSSATZ", "Methodius. 99"]),
        ("alto", ["The cat sat here."]),
    ],
)
def test_flexiconv_output_yields_rows(name, expected):
    rows = read_teitok_rows(FLEXICONV_FIXTURES / f"{name}.teitok.xml")
    assert [r["text"] for r in rows] == expected


def test_text_faithful_spacing_and_multiword_tokens(tmp_path):
    """Whitespace between tokens is the space (no join attribute needed); <dtok> words of a
    multi-word token never add text."""
    p = tmp_path / "mwt.teitok.xml"
    p.write_text(
        '<TEI><text><s id="s-1"><tok id="w-1">Praze</tok><tok id="w-2">,</tok> '
        '<tok id="w-3">abych<dtok id="w-3.1" form="aby"/><dtok id="w-3.2" form="bych"/></tok> '
        '<tok id="w-4">šel</tok></s></text></TEI>',
        encoding="utf-8",
    )
    assert read_teitok_rows(p)[0]["text"] == "Praze, abych šel"
    tokens = read_teitok_tokens(p)
    assert [(t["form"], t["space_after"]) for t in tokens] == [
        ("Praze", False),
        (",", True),
        ("abych", True),
        ("šel", True),
    ]
