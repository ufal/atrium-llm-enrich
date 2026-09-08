"""
vocab_sources.py — flat controlled-vocabulary harvesting (AMCR + TEATER)

Stage 1 of the two-stage vocabulary build:

    harvest (network)  ->  FLAT artifacts  ->  nest (pure)  ->  NESTED artifacts
    vocab_sources.py       vocab_flat_*.json  vocab_manager    teater_nested_vocab.json

Why this module exists separately from ``vocab_manager.py``:

  * ``vocab_manager.py`` is imported at module scope by the LLM services (llm-enrich's
    ``service/api.py``, ``ollama_client.py``, ``openrouter_client.py``). It must stay
    dependency-light — in particular ``lxml`` must NOT enter that import path. Every
    import of this module from ``vocab_manager`` is therefore deliberately lazy.
  * Only this stage needs the network. Splitting it out makes re-nesting a pure,
    offline, deterministic operation that can be re-run whenever the taxonomy changes.

Two sources, both CC BY-NC 4.0:

  AMCR    OAI-PMH ``api.aiscr.cz/2.2/oai?set=heslo``. The ``heslo`` record carries far
          more than a cs/en label pair: ``ident_cely`` (stable id), ``nazev_heslare``
          (which of the ~50 controlled lists the term belongs to), ``popis``/``popis_en``,
          ``zkratka``, ``razeni``, ``hierarchie_vyse`` (broader terms) and ``odkaz``
          (SKOS mappings to external vocabularies). All of it is captured here.

  TEATER  The Thesaurus of Archaeological Terminology, ``teater.aiscr.cz``. Its full
          content is published two ways, both used here:
            * ``snapshot`` (default) — the 12 static ``backend/json/import_N.json`` files
              committed to ``ARUP-CAS/aiscr-teater``, fetched from raw.githubusercontent
              at a pinned commit. Reproducible, auth-free, rate-limit-free, and reachable
              from networks that block ``aiscr.cz``.
            * ``live`` — ``GET https://teater.aiscr.cz/api/export``, which serves the same
              tree plus an ``lastImport`` date.
          Either way the result is trilingual labels and the real broader/narrower tree,
          so TEATER's nesting never has to be re-invented.
"""

from __future__ import annotations

import csv
import io
import json
import time
import unicodedata
import urllib.parse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

# ── endpoints ─────────────────────────────────────────────────────────────────

AMCR_OAI_BASE = "https://api.aiscr.cz/2.2/oai"
AMCR_ID_BASE = "https://api.aiscr.cz/id/"
AMCR_NS = {
    "oai": "http://www.openarchives.org/OAI/2.0/",
    "amcr": "https://api.aiscr.cz/schema/amcr/2.2/",
}

TEATER_ID_BASE = "https://teater.aiscr.cz/id/"
TEATER_EXPORT_URL = "https://teater.aiscr.cz/api/export"

# Pinned so a re-harvest is reproducible. Bump deliberately, and record the move in the
# artifact's .meta.json — an unpinned "master" would make every harvest a silent diff.
TEATER_SNAPSHOT_REF = "2106c59103556731c56dc116573c4b02e1199466"
TEATER_SNAPSHOT_URL = (
    "https://raw.githubusercontent.com/ARUP-CAS/aiscr-teater/{ref}/backend/json/import_{n}.json"
)
TEATER_SNAPSHOT_FILES = 12

USER_AGENT = "ATRIUM-vocabulary-harvester/2.0"
DEFAULT_DELAY = 0.3
MAX_PAGES = 500

# TEATER description blocks are keyed by these. The *_eq_title blocks hold the
# language equivalents (alt labels); comment_title holds the scope note.
_TEATER_ALT_KEYS = {"cz_eq_title": "cs", "en_eq_title": "en", "de_eq_title": "de"}
_TEATER_NOTE_KEY = "comment_title"

# AMCR <odkaz> mapping relations -> the VocabRecord field that carries them.
#
# This used to be a single `== "skos:exactMatch"` test, and everything else the
# heslar asserts was dropped on the floor. The narrowing was deliberate and its
# reason was sound -- broadMatch is not an identity claim, so promoting it into a
# field named `exact_match` would have been a lie -- but the fix for "this URI means
# something weaker" is a field for the weaker thing, not silence. SKOS already draws
# the distinction; we just have to keep it.
#
# The four SKOS mapping properties, and what each licenses:
#   exactMatch   - interchangeable across applications, and TRANSITIVE. The strong one.
#   closeMatch   - interchangeable in SOME applications; deliberately not transitive.
#   broadMatch   - the external concept is broader than ours.
#   relatedMatch - associative, no hierarchy claimed.
#
# `exact_match` keeps its name and its exact previous contents, so every existing
# consumer (vocab_review._aat_verdict, collision_review.csv) is untouched.
AMCR_MAPPING_RELATIONS: Dict[str, str] = {
    "skos:exactMatch": "exact_match",
    "skos:closeMatch": "close_match",
    "skos:broadMatch": "broad_match",
    "skos:relatedMatch": "related_match",
}


# ── the normalised record ─────────────────────────────────────────────────────


@dataclass(frozen=True)
class VocabRecord:
    """One controlled-vocabulary term, normalised across both sources.

    ``cs`` is kept verbatim (not case-folded): the nested vocabulary and the LLM's
    ``teater_category`` enum both use it as the surface label. Use :func:`norm_label`
    when matching.

    ``scheme`` is the source's coarse grouping — the AMCR heslar name, or the TEATER
    top-level branch id — and is what the taxonomy maps a term to a facet by. ``sub`` is
    the source's own *second* level: TEATER's depth-2 label, or the AMCR heslar again.
    It carries the granularity both thesauri curate and the 7-theme rollup discards, and
    it is what the prompt renders as a sub-header inside each facet.
    """

    cs: str
    en: Optional[str] = None
    de: Optional[str] = None
    source: str = ""
    source_id: str = ""
    uri: str = ""
    scheme: Optional[str] = None
    sub: Optional[str] = None
    broader: Tuple[str, ...] = ()
    sort: Optional[int] = None
    abbr: Optional[str] = None
    alt_cs: Tuple[str, ...] = ()
    alt_en: Tuple[str, ...] = ()
    note_cs: Optional[str] = None
    note_en: Optional[str] = None
    exact_match: Tuple[str, ...] = field(default=())
    close_match: Tuple[str, ...] = field(default=())
    broad_match: Tuple[str, ...] = field(default=())
    related_match: Tuple[str, ...] = field(default=())
    citation_uri: Tuple[str, ...] = field(default=())

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for key in _TUPLE_FIELDS:
            d[key] = list(d[key])
        return d

    def mappings(self) -> Dict[str, Tuple[str, ...]]:
        """The SKOS mapping properties this record asserts, keyed by property name.

        Empty tuples are omitted, so a caller can iterate without checking. Note that
        :attr:`citation_uri` is deliberately NOT included: see its note below.
        """
        out = {
            relation: getattr(self, attr)
            for relation, attr in AMCR_MAPPING_RELATIONS.items()
            if getattr(self, attr)
        }
        return out


#: Every VocabRecord field that is a tuple on the dataclass and a list in JSON.
#: as_dict() and read_flat_json() both walk this, so adding a field means editing
#: one list rather than three call sites that silently disagree.
_TUPLE_FIELDS: Tuple[str, ...] = (
    "broader",
    "alt_cs",
    "alt_en",
    "exact_match",
    "close_match",
    "broad_match",
    "related_match",
    "citation_uri",
)


def record_sort_key(record: "VocabRecord") -> Tuple[str, int, str, str]:
    """The one canonical record order.

    Numeric-aware on ``source_id`` so TEATER's "10" sorts after "2" rather than before
    it. Used by every writer *and* by :func:`to_term_pairs`, because a label collision's
    winner must not depend on the order records happened to arrive in — that is the
    difference between a reproducible artifact and one that changes when you round-trip
    it through a flat file.
    """
    ident = record.source_id or ""
    numeric = int(ident) if ident.isdigit() else 0
    return (record.source, numeric, ident, record.cs)


def norm_label(value: Optional[str]) -> str:
    """Fold a label for cross-source matching.

    NFC, case-folded, whitespace-collapsed. The NBSP step is not hypothetical: the
    shipped AMCR vocabulary contains ``'3D –\xa0VRML (*.wrl)'``.
    """
    if not value:
        return ""
    text = unicodedata.normalize("NFC", value).replace("\xa0", " ")
    return " ".join(text.casefold().split())


def _secure_xml_parser():
    """lxml parser with entity expansion and network access disabled.

    ``vocab_manager.fetch_amcr_vocab`` historically used ``xml.etree`` defaults, which
    offer no protection against entity-expansion payloads in a remote document.
    """
    from lxml import etree

    return etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        load_dtd=False,
        huge_tree=False,
    )


def _session(session: Optional[requests.Session] = None) -> requests.Session:
    if session is not None:
        return session
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


# ── AMCR ──────────────────────────────────────────────────────────────────────


def _amcr_text(block, tag: str, ns: str) -> str:
    el = block.find(f"{{{ns}}}{tag}")
    return (el.text or "").strip() if el is not None else ""


def _amcr_record(block, ns: str) -> Optional[VocabRecord]:
    xml_lang = "{http://www.w3.org/XML/1998/namespace}lang"

    cs = en = ""
    for child in block:
        if child.tag == f"{{{ns}}}heslo" and child.get(xml_lang) == "cs":
            cs = (child.text or "").strip()
        elif child.tag == f"{{{ns}}}heslo_en":
            en = (child.text or "").strip()
    if not cs:
        return None

    ident = _amcr_text(block, "ident_cely", ns)
    razeni = _amcr_text(block, "razeni", ns)

    broader: List[str] = []
    for hv in block.findall(f"{{{ns}}}hierarchie_vyse"):
        parent = hv.find(f"{{{ns}}}heslo_nadrazene")
        if parent is not None and parent.get("id"):
            broader.append(parent.get("id"))

    # One bucket per SKOS mapping property AMCR is known to assert. An <odkaz> whose
    # relation is not in AMCR_MAPPING_RELATIONS is skipped rather than guessed at:
    # inventing a relation for an unrecognised one would put a claim in the graph that
    # the source never made, and a mapping nobody asserted is worse than none.
    matches: Dict[str, List[str]] = {attr: [] for attr in AMCR_MAPPING_RELATIONS.values()}
    for odkaz in block.findall(f"{{{ns}}}odkaz"):
        relation = _amcr_text(odkaz, "skos_mapping_relation", ns)
        uri = _amcr_text(odkaz, "uri", ns)
        attr = AMCR_MAPPING_RELATIONS.get(relation)
        if uri and attr:
            matches[attr].append(uri)

    return VocabRecord(
        cs=cs,
        en=en or None,
        source="amcr",
        source_id=ident,
        uri=f"{AMCR_ID_BASE}{ident}" if ident else "",
        scheme=_amcr_text(block, "nazev_heslare", ns) or None,
        sub=_amcr_text(block, "nazev_heslare", ns) or None,
        broader=tuple(broader),
        sort=int(razeni) if razeni.lstrip("-").isdigit() else None,
        abbr=_amcr_text(block, "zkratka", ns) or None,
        note_cs=_amcr_text(block, "popis", ns) or None,
        note_en=_amcr_text(block, "popis_en", ns) or None,
        exact_match=tuple(matches["exact_match"]),
        close_match=tuple(matches["close_match"]),
        broad_match=tuple(matches["broad_match"]),
        related_match=tuple(matches["related_match"]),
    )


def harvest_amcr(
    delay: float = DEFAULT_DELAY,
    session: Optional[requests.Session] = None,
    max_pages: int = MAX_PAGES,
    oai_set: str = "heslo",
) -> Tuple[List[VocabRecord], Dict[str, Any]]:
    """Harvest the AMCR heslář over OAI-PMH.

    Returns ``(records, meta)``. A network or parse failure mid-walk returns what was
    collected so far rather than raising — the caller decides whether a partial harvest
    is acceptable (``vocab_build`` refuses to overwrite a good artifact with one).
    """
    from lxml import etree

    sess = _session(session)
    parser = _secure_xml_parser()
    records: List[VocabRecord] = []
    seen: set[str] = set()

    url: Optional[str] = f"{AMCR_OAI_BASE}?verb=ListRecords&metadataPrefix=oai_amcr&set={oai_set}"
    page = 0
    error: Optional[str] = None

    print("[AMCR] Starting OAI-PMH harvest…")
    while url and page < max_pages:
        page += 1
        print(f"  [AMCR] Fetching page {page}…")
        try:
            resp = sess.get(url, timeout=60)
            resp.raise_for_status()
        except requests.RequestException as exc:
            error = f"network error on page {page}: {exc}"
            print(f"  [AMCR] {error}")
            break
        try:
            root = etree.fromstring(resp.content, parser=parser)
        except etree.XMLSyntaxError as exc:
            error = f"XML parse error on page {page}: {exc}"
            print(f"  [AMCR] {error}")
            break

        ns = AMCR_NS["amcr"]
        for record in root.iter(f"{{{AMCR_NS['oai']}}}record"):
            for block in record.iter(f"{{{ns}}}heslo"):
                # The record's own <heslo> label element shares the tag name with the
                # wrapper; only the wrapper has children.
                if len(block) == 0:
                    continue
                item = _amcr_record(block, ns)
                if item is None:
                    continue
                key = item.source_id or norm_label(item.cs)
                if key in seen:
                    continue
                seen.add(key)
                records.append(item)

        token_el = root.find(f".//{{{AMCR_NS['oai']}}}resumptionToken")
        if token_el is not None and token_el.text and token_el.text.strip():
            token = urllib.parse.quote(token_el.text.strip())
            url = f"{AMCR_OAI_BASE}?verb=ListRecords&resumptionToken={token}"
            time.sleep(delay)
        else:
            url = None

    meta = {
        "name": "amcr",
        "endpoint": AMCR_OAI_BASE,
        "metadata_prefix": "oai_amcr",
        "set": oai_set,
        "strategy": "oai-pmh",
        "pages": page,
        "records": len(records),
        "without_en": sum(1 for r in records if not r.en),
        "license": "CC BY-NC 4.0",
    }
    if error:
        meta["error"] = error
    print(f"[AMCR] Harvest complete. {len(records)} terms from {page} page(s).")
    return records, meta


# ── TEATER ────────────────────────────────────────────────────────────────────


def _teater_labels(node: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """TEATER names are ``{cz, en, de}`` in the snapshot and ``{cs, en, de}`` in the
    live export DTO. Accept both."""
    name = node.get("name") or {}
    return {
        "cs": (name.get("cz") or name.get("cs") or "").strip() or None,
        "en": (name.get("en") or "").strip() or None,
        "de": (name.get("de") or "").strip() or None,
    }


def _teater_descriptions(node: Dict[str, Any]) -> Dict[str, Any]:
    """Pull alt labels and the scope note out of a node's ``description`` blocks.

    ``*_eq_title`` blocks carry a plain string per content entry; ``comment_title``
    carries a ``{cz, en, de}`` dict. Both shapes occur, and mixing them up is the
    easiest way to get this parser wrong.
    """
    alt: Dict[str, List[str]] = {"cs": [], "en": [], "de": []}
    note_cs = note_en = None
    citations: List[str] = []

    for desc in node.get("description") or []:
        if not isinstance(desc, dict):
            continue
        title = desc.get("title")
        key = title.get("text") if isinstance(title, dict) else None
        contents = desc.get("content")
        if not isinstance(contents, list):
            continue

        for entry in contents:
            if not isinstance(entry, dict):
                continue
            text = entry.get("text")
            if key in _TEATER_ALT_KEYS and isinstance(text, str) and text.strip():
                alt[_TEATER_ALT_KEYS[key]].append(text.strip())
            elif key == _TEATER_NOTE_KEY and isinstance(text, dict):
                note_cs = note_cs or (text.get("cz") or "").strip() or None
                note_en = note_en or (text.get("en") or "").strip() or None
            citations.extend(_teater_quote_urls(entry))

    return {
        "alt": alt,
        "note_cs": note_cs,
        "note_en": note_en,
        "citations": tuple(dict.fromkeys(citations)),
    }


def _teater_quote_urls(entry: Dict[str, Any]) -> List[str]:
    """URLs cited in support of one label equivalent.

    TEATER hangs a ``quotes`` list off each ``description[].content[]`` entry, and some
    of those quotes point at Getty AAT (``title_page: "AAT"``,
    ``location.url: "http://vocab.getty.edu/aat/..."``). The parser walked straight past
    them, so alignment data TEATER already publishes was never harvested.

    THESE ARE CITATIONS, NOT MAPPINGS. A quote is bibliographic support for a label --
    "this English equivalent is attested in AAT" -- not an assertion that the two
    concepts are the same thing. Promoting them to ``skos:exactMatch`` would manufacture
    identity claims TEATER never made, which is exactly the over-claim the AMCR side
    avoids by refusing to promote ``broadMatch``. They land in ``citation_uri`` and are
    serialised as ``dcterms:source``. Upgrading any of them to a real mapping is an
    editorial act for the vocabulary curators, not a parsing decision.
    """
    urls: List[str] = []
    for quote in entry.get("quotes") or []:
        if not isinstance(quote, dict):
            continue
        location = quote.get("location")
        if not isinstance(location, dict):
            continue
        url = (location.get("url") or "").strip()
        if url:
            urls.append(url)
    return urls


def _teater_walk(
    node: Dict[str, Any],
    ancestors: Tuple[str, ...],
    out: Dict[str, VocabRecord],
    children_key: str,
    sub: Optional[str] = None,
) -> None:
    """Flatten one TEATER subtree.

    ``sub`` is the depth-2 label inherited down the branch: a depth-2 node names itself
    and every descendant under it. Depth-1 roots keep ``sub=None`` — they are numbered
    section titles ("5) Chronologie"), not concepts, and :func:`to_term_pairs` drops
    them so they never reach the model as selectable terms.
    """
    node_id = str(node.get("id") or "").strip()
    if not node_id:
        return

    labels = _teater_labels(node)
    if labels["cs"] and node_id not in out:
        extra = _teater_descriptions(node)
        # The concept's own preferred label often repeats as its first cs equivalent.
        alt_cs = tuple(
            a
            for a in dict.fromkeys(extra["alt"]["cs"])
            if norm_label(a) != norm_label(labels["cs"])
        )
        alt_en = tuple(
            a
            for a in dict.fromkeys(extra["alt"]["en"])
            if norm_label(a) != norm_label(labels["en"])
        )
        out[node_id] = VocabRecord(
            cs=labels["cs"],
            en=labels["en"] or (extra["alt"]["en"][0] if extra["alt"]["en"] else None),
            de=labels["de"],
            source="teater",
            source_id=node_id,
            uri=f"{TEATER_ID_BASE}{node_id}",
            scheme=ancestors[0] if ancestors else node_id,
            sub=sub,
            broader=ancestors,
            alt_cs=alt_cs,
            alt_en=alt_en,
            note_cs=extra["note_cs"],
            note_en=extra["note_en"],
            citation_uri=extra["citations"],
        )

    for child in node.get(children_key) or []:
        if isinstance(child, dict):
            # A depth-1 root's children are the depth-2 groups; each names its subtree.
            child_sub = sub if ancestors else _teater_labels(child)["cs"]
            _teater_walk(child, ancestors + (node_id,), out, children_key, child_sub)


def harvest_teater(
    mode: str = "snapshot",
    session: Optional[requests.Session] = None,
    ref: str = TEATER_SNAPSHOT_REF,
) -> Tuple[List[VocabRecord], Dict[str, Any]]:
    """Harvest the TEATER thesaurus, hierarchy included.

    ``mode="snapshot"`` (default) reads the 12 pinned JSON files from GitHub;
    ``mode="live"`` reads ``GET /api/export``. Falls back from live to snapshot so a
    transient outage does not fail the build.
    """
    sess = _session(session)
    out: Dict[str, VocabRecord] = {}
    meta: Dict[str, Any] = {"name": "teater", "license": "CC BY-NC 4.0"}

    if mode == "live":
        try:
            resp = sess.get(TEATER_EXPORT_URL, timeout=120)
            resp.raise_for_status()
            payload = resp.json()
            for root in payload.get("categories") or []:
                _teater_walk(root, (), out, "children")
            meta.update(
                strategy="export",
                endpoint=TEATER_EXPORT_URL,
                upstream_last_import=payload.get("lastImport"),
            )
        except (requests.RequestException, ValueError) as exc:
            print(f"  [TEATER] live export failed ({exc}); falling back to the pinned snapshot.")
            out.clear()
            mode = "snapshot"

    if mode == "snapshot":
        print(f"[TEATER] Reading pinned snapshot {ref[:12]}…")
        for n in range(1, TEATER_SNAPSHOT_FILES + 1):
            url = TEATER_SNAPSHOT_URL.format(ref=ref, n=n)
            resp = sess.get(url, timeout=120)
            resp.raise_for_status()
            for root in resp.json():
                _teater_walk(root, (), out, "subcategories")
        meta.update(
            strategy="snapshot",
            endpoint=TEATER_SNAPSHOT_URL.format(ref=ref, n="{n}"),
            snapshot_ref=ref,
            files=TEATER_SNAPSHOT_FILES,
        )

    records = sorted(out.values(), key=record_sort_key)
    roots = sorted({r.scheme for r in records if r.scheme})
    meta.update(
        records=len(records), roots=len(roots), without_en=sum(1 for r in records if not r.en)
    )
    print(f"[TEATER] Harvest complete. {len(records)} concepts in {len(roots)} branches.")
    return records, meta


# ── orchestration ─────────────────────────────────────────────────────────────


def harvest(
    sources: Sequence[str] = ("amcr", "teater"),
    delay: float = DEFAULT_DELAY,
    teater_mode: str = "snapshot",
    session: Optional[requests.Session] = None,
) -> Dict[str, Tuple[List[VocabRecord], Dict[str, Any]]]:
    """Harvest each requested source. Returns ``{source: (records, meta)}``."""
    result: Dict[str, Tuple[List[VocabRecord], Dict[str, Any]]] = {}
    for name in sources:
        if name == "amcr":
            result["amcr"] = harvest_amcr(delay=delay, session=session)
        elif name == "teater":
            result["teater"] = harvest_teater(mode=teater_mode, session=session)
        else:
            raise ValueError(f"unknown vocabulary source: {name!r}")
    return result


def merge(
    per_source: Dict[str, Tuple[List[VocabRecord], Dict[str, Any]]],
    precedence: Sequence[str] = ("amcr", "teater"),
) -> Tuple[List[VocabRecord], int]:
    """Merge sources into one label-unique list. Earlier sources win collisions.

    Label uniqueness is not cosmetic: ``build_schema`` turns the term list into an
    ``enum.Enum``, where duplicate values silently become aliases and collapse two
    distinct concepts into one.

    Records are sorted by :func:`record_sort_key` before deduping, exactly as
    :func:`to_term_pairs` does. Without that, the winner of an intra-source label
    collision depends on the order records happened to arrive in — OAI page order during
    a live harvest, canonical order on a ``--from-flat`` re-nest — so the same inputs
    produced two different vocabularies. That made 116 AMCR labels resolve to a
    different record between the two paths, moved 57 of them to a different theme, and
    silently broke the "nesting is reproducible from the committed flat files" guarantee
    that ``vocab_build --check`` and the refresh workflow both rest on.
    """
    chosen: Dict[str, VocabRecord] = {}
    collisions = 0
    for name in precedence:
        for record in sorted(per_source.get(name, ([], {}))[0], key=record_sort_key):
            key = norm_label(record.cs)
            if not key:
                continue
            if key in chosen:
                collisions += 1
                continue
            chosen[key] = record
    return sorted(chosen.values(), key=record_sort_key), collisions


# ── serialisation ─────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    "source_lemma",
    "target_translation",
    "source",
    "source_id",
    "uri",
    "scheme",
    "sub",
    "broader",
    "sort",
]


def flat_json_text(records: Iterable[VocabRecord], meta: Dict[str, Any]) -> str:
    """Serialise the flat archive.

    ``terms`` is a list, so ordering is explicit rather than an artifact of dict key
    order; keys inside each record are sorted. This is the single implementation — every
    caller goes through it, because a second inline copy of the sort is exactly how the
    round-trip stopped being reproducible once already.
    """
    ordered = sorted(records, key=record_sort_key)
    payload = {"_meta": meta, "terms": [r.as_dict() for r in ordered]}
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n"


def flat_csv_text(records: Iterable[VocabRecord]) -> str:
    """Serialise the rich CSV view.

    Columns 1-2 are the ``source_lemma,target_translation`` pair atrium-translator's
    vocabulary loader reads positionally, so this file doubles as a translator glossary;
    columns 3+ are additive provenance.
    """
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(CSV_COLUMNS)
    for r in sorted(records, key=record_sort_key):
        writer.writerow(
            [
                r.cs.lower(),
                r.en or "",
                r.source,
                r.source_id,
                r.uri,
                r.scheme or "",
                r.sub or "",
                " ".join(r.broader),
                "" if r.sort is None else r.sort,
            ]
        )
    return buf.getvalue()


def vocabulary_csv_text(records: Iterable[VocabRecord]) -> str:
    """Serialise the strict two-column glossary atrium-translator documents."""
    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["source_lemma", "target_translation"])
    for r in sorted(records, key=record_sort_key):
        if r.en:
            writer.writerow([r.cs.lower(), r.en])
    return buf.getvalue()


def write_flat_json(records: Iterable[VocabRecord], path: Path, meta: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(flat_json_text(records, meta), encoding="utf-8")


def read_flat_json(path: Path) -> Tuple[List[VocabRecord], Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    records = []
    for item in payload.get("terms", []):
        records.append(
            VocabRecord(
                cs=item.get("cs", ""),
                en=item.get("en"),
                de=item.get("de"),
                source=item.get("source", ""),
                source_id=item.get("source_id", ""),
                uri=item.get("uri", ""),
                scheme=item.get("scheme"),
                sub=item.get("sub"),
                broader=tuple(item.get("broader") or ()),
                sort=item.get("sort"),
                abbr=item.get("abbr"),
                alt_cs=tuple(item.get("alt_cs") or ()),
                alt_en=tuple(item.get("alt_en") or ()),
                note_cs=item.get("note_cs"),
                note_en=item.get("note_en"),
                exact_match=tuple(item.get("exact_match") or ()),
                close_match=tuple(item.get("close_match") or ()),
                broad_match=tuple(item.get("broad_match") or ()),
                related_match=tuple(item.get("related_match") or ()),
                citation_uri=tuple(item.get("citation_uri") or ()),
            )
        )
    return records, payload.get("_meta", {})


def write_flat_csv(records: Iterable[VocabRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(flat_csv_text(records), encoding="utf-8")


def write_vocabulary_csv(records: Iterable[VocabRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(vocabulary_csv_text(records), encoding="utf-8")


# ── SKOS serialisation ────────────────────────────────────────────────────────
#
# WHAT IS AND IS NOT ASSERTED HERE, AND WHY
# =========================================
# Concept URIs are the SOURCES' OWN. AMCR and TEATER both already mint resolvable
# identifiers, they are already in `VocabRecord.uri`, and per ufal/atrium-project#51
# they are the identifiers that will become PID references to the eventually-published
# SKOS version. ATRIUM mints nothing for them. The only ATRIUM-minted URIs in this
# output are the two ConceptSchemes below, which describe *the harvest* -- an artifact
# ATRIUM really does own -- and never the concepts inside it.
#
# `broader` MEANS DIFFERENT THINGS IN THE TWO SOURCES, and neither maps to
# `skos:broader` directly. Measured over the shipped artifacts:
#
#   AMCR   1,176 `hierarchie_vyse` edges. ALL of them cross heslar boundaries; NONE
#          stay inside one. `obývání` (heslar `aktivita`, an activity) declares 22
#          "superior" terms and every one is an `areal` -- a site type. That is an
#          ASSOCIATIVE relation ("this activity is recorded for these site types"),
#          not a thesaurus hierarchy, and `skos:broader` would both misstate it and
#          give one concept 22 parents. It is emitted as `skos:related`.
#
#   TEATER 14,684 edges, ALL inside one top-level branch, none crossing. That is a
#          real hierarchy -- but the field holds the WHOLE ancestor chain root-first
#          (up to 12 deep), not the parent. `skos:broader` is defined as the DIRECT
#          link, so only the last element becomes `skos:broader`; the rest become
#          `skos:broaderTransitive`, which is exactly what SKOS provides for a
#          transitive ancestor claim.
#
# Getting this wrong would not fail any validator. `skos:broader` on 22 cross-facet
# parents is perfectly well-formed RDF; it is just false. That asymmetry -- syntax
# checks pass, meaning is wrong -- is the whole argument for writing the semantics
# down next to the code that emits it.

#: ATRIUM-minted scheme URIs for the two harvests. The concepts they contain keep the
#: sources' own URIs; only the harvest is ATRIUM's.
SKOS_HARVEST_SCHEMES = {
    "amcr": "https://w3id.org/atrium/scheme/amcr-harvest",
    "teater": "https://w3id.org/atrium/scheme/teater-harvest",
}

_SKOS_PREFIXES = {
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "dct": "http://purl.org/dc/terms/",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "atrium": "https://w3id.org/atrium/",
}

_SKOS_SCHEME_LABELS = {
    "amcr": (
        "ATRIUM harvest of the AMCR heslar",
        "ATRIUM's harvested view of the AMCR controlled lists (heslare), gathered over "
        "OAI-PMH. The concepts are AMCR's and keep their own api.aiscr.cz identifiers; "
        "this scheme describes the harvest, which is ATRIUM's artifact.",
    ),
    "teater": (
        "ATRIUM harvest of TEATER",
        "ATRIUM's harvested view of TEATER, the Thesaurus of Archaeological "
        "Terminology. The concepts are TEATER's and keep their own teater.aiscr.cz "
        "identifiers; this scheme describes the harvest.",
    ),
}


def _ttl_lit(value: str, lang: Optional[str] = None) -> str:
    escaped = (
        str(value)
        .replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"@{lang}' if lang else f'"{escaped}"'


def _source_id_to_uri(source: str, ident: str) -> str:
    base = AMCR_ID_BASE if source == "amcr" else TEATER_ID_BASE
    return f"{base}{ident}"


def skos_triples(
    records: Sequence[VocabRecord],
) -> List[Tuple[str, str, Tuple[str, Any, Optional[str]]]]:
    """Every SKOS triple for *records*, once, sorted.

    Objects are ``("uri", value, None)`` or ``("lit", value, lang_or_None)``. Both
    serialisers below render this and nothing else, so the Turtle and the JSON-LD
    cannot describe different graphs -- a property worth engineering rather than
    hoping for, since a hand-written pair of renderers over one table is precisely
    how a label set drifts.
    """
    out: List[Tuple[str, str, Tuple[str, Any, Optional[str]]]] = []
    present = sorted({r.source for r in records if r.source in SKOS_HARVEST_SCHEMES})

    for source in present:
        scheme = SKOS_HARVEST_SCHEMES[source]
        title, description = _SKOS_SCHEME_LABELS[source]
        out.append((scheme, "rdf:type", ("uri", "skos:ConceptScheme", None)))
        out.append((scheme, "dct:title", ("lit", title, "en")))
        out.append((scheme, "dct:description", ("lit", description, "en")))
        out.append(
            (
                scheme,
                "dct:license",
                ("uri", "https://creativecommons.org/licenses/by-nc/4.0/", None),
            )
        )

    for record in sorted(records, key=record_sort_key):
        if not record.uri:
            # No source identifier means no subject to hang triples off. Minting one
            # would invent an identity for a concept that has none upstream.
            continue
        uri = record.uri
        out.append((uri, "rdf:type", ("uri", "skos:Concept", None)))
        if record.source in SKOS_HARVEST_SCHEMES:
            out.append((uri, "skos:inScheme", ("uri", SKOS_HARVEST_SCHEMES[record.source], None)))
        if record.source_id:
            out.append((uri, "skos:notation", ("lit", record.source_id, None)))
        for lang, value in (("cs", record.cs), ("en", record.en), ("de", record.de)):
            if value:
                out.append((uri, "skos:prefLabel", ("lit", value, lang)))
        for value in record.alt_cs:
            out.append((uri, "skos:altLabel", ("lit", value, "cs")))
        for value in record.alt_en:
            out.append((uri, "skos:altLabel", ("lit", value, "en")))
        if record.abbr:
            # An abbreviation is a label the source curates, not a second notation:
            # `zkratka` is not unique across heslare, and skos:notation is meant to be.
            out.append((uri, "skos:altLabel", ("lit", record.abbr, "cs")))
        if record.note_cs:
            out.append((uri, "skos:scopeNote", ("lit", record.note_cs, "cs")))
        if record.note_en:
            out.append((uri, "skos:scopeNote", ("lit", record.note_en, "en")))

        # See the long note above: the two sources' `broader` fields are different
        # relations and are emitted as different properties.
        if record.broader:
            if record.source == "teater":
                chain = [b for b in record.broader if b]
                if chain:
                    parent = _source_id_to_uri(record.source, chain[-1])
                    out.append((uri, "skos:broader", ("uri", parent, None)))
                for ancestor in chain[:-1]:
                    out.append(
                        (
                            uri,
                            "skos:broaderTransitive",
                            ("uri", _source_id_to_uri(record.source, ancestor), None),
                        )
                    )
            else:
                for related in record.broader:
                    if related:
                        out.append(
                            (
                                uri,
                                "skos:related",
                                ("uri", _source_id_to_uri(record.source, related), None),
                            )
                        )

        for relation, uris in record.mappings().items():
            for target in uris:
                out.append((uri, relation, ("uri", target, None)))
        for target in record.citation_uri:
            # dcterms:source, never a skos match -- see _teater_quote_urls().
            out.append((uri, "dct:source", ("uri", target, None)))

    seen: set = set()
    unique = []
    for triple in out:
        if triple in seen:
            continue
        seen.add(triple)
        unique.append(triple)
    return sorted(unique, key=lambda t: (t[0], t[1], str(t[2][1]), t[2][2] or ""))


def _skos_meta_comment(meta: Dict[str, Any]) -> List[str]:
    """Provenance header lines. ``generated_utc`` is deliberately omitted.

    ``vocab_build._normalise_for_check`` blanks that one line before comparing, and it
    only knows how to do so for the ``"generated_utc"`` JSON key. A timestamp in a
    Turtle comment would be invisible to it and would make every ``--check`` run report
    drift, which is how a drift gate gets switched off.
    """
    lines = ["# ATRIUM SKOS view of the harvested source vocabularies.", "#"]
    for entry in meta.get("sources", []) or []:
        bits = [f"name={entry.get('name')}", f"records={entry.get('records')}"]
        for key in ("strategy", "set", "snapshot_ref", "endpoint", "license"):
            if entry.get(key):
                bits.append(f"{key}={entry[key]}")
        lines.append("# source: " + " ".join(bits))
    lines.append("#")
    lines.append("# Concept URIs are the sources' own; see skos_triples() for what is asserted.")
    lines.append("")
    return lines


def skos_turtle_text(records: Sequence[VocabRecord], meta: Dict[str, Any]) -> str:
    """The SKOS view as Turtle. Deterministic; safe for ``--check``."""
    lines = _skos_meta_comment(meta)
    lines.append("@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .")
    for prefix in sorted(_SKOS_PREFIXES):
        lines.append(f"@prefix {prefix}: <{_SKOS_PREFIXES[prefix]}> .")
    lines.append("")

    grouped: Dict[str, List[Tuple[str, Tuple[str, Any, Optional[str]]]]] = {}
    order: List[str] = []
    for subject, prop, obj in skos_triples(records):
        if subject not in grouped:
            grouped[subject] = []
            order.append(subject)
        grouped[subject].append((prop, obj))

    for subject in order:
        rendered = []
        for prop, obj in grouped[subject]:
            kind, value, lang = obj
            if kind == "uri":
                rendered.append(
                    f"    {prop} "
                    + (value if ":" in value and not value.startswith("http") else f"<{value}>")
                )
            else:
                rendered.append(f"    {prop} {_ttl_lit(value, lang)}")
        head = f"<{subject}> {rendered[0].lstrip()}"
        lines.append(" ;\n".join([head] + rendered[1:]) + " .")
        lines.append("")
    return "\n".join(lines).rstrip("\n") + "\n"


def skos_jsonld_text(records: Sequence[VocabRecord], meta: Dict[str, Any]) -> str:
    """The SKOS view as JSON-LD. Renders the same triples as :func:`skos_turtle_text`."""
    grouped: Dict[str, Dict[str, List[Any]]] = {}
    order: List[str] = []
    for subject, prop, obj in skos_triples(records):
        if subject not in grouped:
            grouped[subject] = {}
            order.append(subject)
        kind, value, lang = obj
        key = "@type" if prop == "rdf:type" else prop
        if kind == "uri":
            rendered: Any = value if key == "@type" else {"@id": value}
        elif lang:
            rendered = {"@language": lang, "@value": value}
        else:
            rendered = value
        grouped[subject].setdefault(key, []).append(rendered)

    graph = []
    for subject in order:
        node: Dict[str, Any] = {"@id": subject}
        for key in sorted(grouped[subject]):
            values = grouped[subject][key]
            node[key] = values[0] if len(values) == 1 else values
        graph.append(node)

    context = dict(_SKOS_PREFIXES)
    payload = {
        "@context": context,
        "_meta": {k: v for k, v in sorted(meta.items()) if k != "generated_utc"},
        "@graph": graph,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"


def _discarded_id(r: "VocabRecord") -> Dict[str, str]:
    return {
        "source": r.source,
        "id": r.source_id,
        "scheme": r.scheme or "",
        "cs": r.cs,
        "en": r.en or "",
    }


def _term_pair(
    record: "VocabRecord", discarded: Sequence["VocabRecord"], bare_cs: Optional[str] = None
) -> Dict[str, Any]:
    pair: Dict[str, Any] = {
        "cs": record.cs,
        "en": record.en,
        "source": record.source,
        "source_id": record.source_id,
        "scheme": record.scheme or "",
        "sub": record.sub or (record.scheme or ""),
        "broader": list(record.broader),
        "sort": record.sort,
        "discarded_ids": [_discarded_id(d) for d in discarded],
    }
    if bare_cs is not None:
        pair["bare_cs"] = bare_cs
    return pair


def group_by_label(records: Iterable[VocabRecord]) -> Dict[str, List[VocabRecord]]:
    """Group records that would collide under the same ``cs`` enum key.

    This is the single definition of "what counts as a collision group" — the same
    grouping :func:`to_term_pairs` dedups or splits, and what ``vocab_review.py``
    reports on for human review. Records without ``cs``/``en``, and TEATER depth-1
    branch roots (numbered section titles, not concepts — "5) Chronologie"), never
    enter a group at all, matching what actually reaches the vocabulary.

    Sorted by :func:`record_sort_key` before grouping, so the group's own member
    order — and therefore which member :func:`to_term_pairs` picks as its default
    winner — never depends on harvest order.
    """
    groups: Dict[str, List[VocabRecord]] = {}
    for r in sorted(records, key=record_sort_key):
        if not r.cs or not r.en:
            continue
        if r.source == "teater" and not r.broader:
            continue
        groups.setdefault(norm_label(r.cs), []).append(r)
    return groups


def to_term_pairs(
    records: Iterable[VocabRecord],
    collisions: Optional[List[Tuple[str, str, str]]] = None,
    qualifiers: Optional[Dict[Tuple[str, str], str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Adapt flat records to the ``{cs: {...}}`` mapping ``VocabularyManager`` nests.

    Carries ``scheme``/``broader``/``source``/``source_id`` through so the nesting stage
    can place a term by curated list membership rather than by substring luck.

    Two records sharing a Czech label cannot both become enum entries under the same
    key (``build_schema`` turns the term list into an ``enum.Enum``, where a repeated
    value silently becomes an alias and collapses two concepts into one), so every
    label group resolves to exactly one of two outcomes:

    * **Dedup** (the default). Every member without an entry in ``qualifiers`` is
      treated as the same concept — a genuine duplicate or, more often, ordinary
      translation variance between AMCR and TEATER. The lowest-sorting record
      (:func:`record_sort_key`, i.e. AMCR before TEATER, then id) wins the bare ``cs``
      key; every other member's identity is recorded in its ``discarded_ids`` (issue
      #6, M7) rather than silently dropped.
    * **Qualified split** (opt-in, B3). A record whose ``(source, source_id)`` appears
      in ``qualifiers`` is pulled out of the group into its own entry keyed
      ``"{cs} ({qualifier})"``, with ``bare_cs`` set so the emitted keyword can be
      stripped back to the plain label after inference. This never happens
      automatically from a raw label collision — a same-label group is assumed to be
      the same concept unless a human has reviewed it and added a qualifier (see
      ``taxonomy_overrides.json``); guessing homonym-hood from an EN gloss mismatch
      alone would mistake ordinary translation variance for a semantic split far more
      often than it would catch a real one.
    """
    if collisions is None:
        collisions = []
    qualifiers = qualifiers or {}

    groups = group_by_label(records)
    pairs: Dict[str, Dict[str, Any]] = {}
    for members in groups.values():
        flagged = [m for m in members if (m.source, m.source_id) in qualifiers]
        plain = [m for m in members if (m.source, m.source_id) not in qualifiers]

        for m in flagged:
            key = f"{m.cs} ({qualifiers[(m.source, m.source_id)]})"
            if key in pairs:
                collisions.append((key, pairs[key]["source_id"], m.source_id))
                continue
            pairs[key] = _term_pair(m, discarded=(), bare_cs=m.cs)

        if plain:
            winner, *discarded = plain  # already sorted by record_sort_key
            key = winner.cs
            if key in pairs:
                collisions.append((key, pairs[key]["source_id"], winner.source_id))
                continue
            pairs[key] = _term_pair(winner, discarded=discarded)
            for d in discarded:
                collisions.append((d.cs, winner.source_id, d.source_id))

    return pairs
