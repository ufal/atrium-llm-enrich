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

    def as_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        for key in ("broader", "alt_cs", "alt_en", "exact_match"):
            d[key] = list(d[key])
        return d


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

    exact: List[str] = []
    for odkaz in block.findall(f"{{{ns}}}odkaz"):
        relation = _amcr_text(odkaz, "skos_mapping_relation", ns)
        uri = _amcr_text(odkaz, "uri", ns)
        if uri and relation == "skos:exactMatch":
            exact.append(uri)

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
        exact_match=tuple(exact),
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

    return {"alt": alt, "note_cs": note_cs, "note_en": note_en}


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
            )
        )
    return records, payload.get("_meta", {})


def write_flat_csv(records: Iterable[VocabRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(flat_csv_text(records), encoding="utf-8")


def write_vocabulary_csv(records: Iterable[VocabRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(vocabulary_csv_text(records), encoding="utf-8")


def to_term_pairs(
    records: Iterable[VocabRecord],
    collisions: Optional[List[Tuple[str, str, str]]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Adapt flat records to the ``{cs: {...}}`` mapping ``VocabularyManager`` nests.

    Carries ``scheme``/``broader``/``source`` through so the nesting stage can place a
    term by curated list membership rather than by substring luck.
    """
    if collisions is None:
        collisions = []
    pairs: Dict[str, Dict[str, Any]] = {}
    seen: Dict[str, str] = {}
    for r in sorted(records, key=record_sort_key):
        if not r.cs or not r.en:
            continue
        if r.source == "teater" and not r.broader:
            # A depth-1 branch root — "5) Chronologie", "8) Předmět". These are numbered
            # section titles in the thesaurus, not concepts, and offering them to the
            # model as categories invites a shrug-answer that is technically in-vocabulary.
            continue
        key = norm_label(r.cs)
        if key in seen:
            # Two concepts share a surface label. Keeping both is not an option: the
            # nested dict is keyed by the label, and build_schema() turns the term list
            # into an enum.Enum where duplicate values silently become aliases and
            # collapse two concepts into one. First-in-canonical-order wins.
            collisions.append((r.cs, seen[key], r.source_id))
            continue
        seen[key] = r.source_id
        pairs[r.cs] = {
            "cs": r.cs,
            "en": r.en,
            "source": r.source,
            "source_id": r.source_id,
            "scheme": r.scheme or "",
            "sub": r.sub or (r.scheme or ""),
            "broader": list(r.broader),
            "sort": r.sort,
        }
    return pairs
