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


def test_read_teitok_rows(sample_teitok):
    rows = read_teitok_rows(sample_teitok)
    assert len(rows) == 3

    # Check page and line tracking
    assert rows[0] == {"page_num": 1, "line_num": 1, "text": "První věta na stránce."}

    # Check fallback text reconstruction from <tok> elements if @text is missing
    assert rows[1] == {"page_num": 1, "line_num": 2, "text": "Druhá chybí text"}

    assert rows[2] == {"page_num": 2, "line_num": 2, "text": "Věta na druhé straně."}


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

    # Check fallback to @type for UPOS if @pos is missing
    assert tokens[7] == {"form": "text", "lemma": "text", "upos": "attr", "space_after": True}


# ── Regression: <pb n="..."> with a non-numeric label (issue #13 TODO — ────
# ── TEITOK-2-MD must survive archival roman-numeral front matter) ──────────

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
    # n="I" doesn't parse -> falls back to incrementing the running counter
    # (1 -> 2), the same fallback already used for a missing @n. The next
    # <pb n="1"> then parses normally and is taken at face value -- page
    # numbers aren't guaranteed monotonic across a non-numeric label, which
    # is an accepted, documented limitation of the fallback (full roman-
    # numeral parsing is out of scope here); not crashing is the contract.
    assert rows[0]["page_num"] == 2
    assert rows[1]["page_num"] == 1


def test_read_teitok_rows_pb_missing_n_still_increments(tmp_path):
    """Existing behavior (missing @n falls back to page_num + 1) must be
    unchanged by the new non-numeric-label handling."""
    p = tmp_path / "missing_n.teitok.xml"
    p.write_text(
        '<TEI><text><body><div><pb/><s text="Only page."/></div></body></text></TEI>',
        encoding="utf-8",
    )
    rows = read_teitok_rows(p)
    assert rows == [{"page_num": 2, "line_num": 1, "text": "Only page."}]


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
    assert rows == [
        {"page_num": 1, "line_num": 1, "text": "Vyzkum odhalil zaklady gotickeho kostela"}
    ]


def test_read_teitok_tokens_survives_name_misclose(tmp_path):
    """read_teitok_tokens() parses independently of read_teitok_rows() and
    needs the same repair -- it doesn't walk through <pb>, so this isolates
    the parse-level fix from the pb-parsing fix."""
    p = tmp_path / "misclose.teitok.xml"
    p.write_text(NAME_MISCLOSE_SAMPLE, encoding="utf-8")
    # Must not raise; a <name> element has no <tok> children in this fixture
    # so the token list is legitimately empty -- the assertion that matters
    # is that parsing completes at all.
    assert read_teitok_tokens(p) == []


def test_parse_teitok_is_noop_for_well_formed_input(sample_teitok):
    """The repair must not alter documents that don't have the quirk --
    guards against fix_name_close_tags' regex over-firing on unrelated
    "</n>"-shaped content and silently corrupting well-formed input."""
    root = parse_teitok(sample_teitok)
    # Same content read_teitok_rows() already exercises via ET.parse directly
    # elsewhere in this file -- three <s> texts/reconstructions, two pages.
    assert [e.get("n") for e in root.iter() if e.tag.split("}")[-1] == "pb"] == ["1", "2"]
