"""
vocab_manager.py — TEATER/AMCR Vocabulary Manager

Handles:
  • OAI-PMH harvesting of controlled-vocabulary term pairs (Czech ↔ English)
    from the AMCR API via paginated HTTP GET requests.
  • Thematic grouping of raw terms into the nested taxonomy structure required
    for LLM system-prompt injection.
  • Thematic priority sorting: prevents administrative terms from displacing
    content-rich archaeological keywords.
  • Optional LLM-assisted fallback classification for unclassified terms.
  • Deterministic on-disk caching of the nested vocabulary.
  • Memoised, lazily-built prompt string.
"""

import json
import re
import time
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import requests

# Administrative words that should sort to the back of their theme, so that prompt
# truncation drops boilerplate before it drops archaeological vocabulary. Module level
# because build_nested() is now a pure function of (taxonomy, terms) and must not depend
# on state built inside the sync path.
#
# The default, not the rule: `_settings.admin_stop_words` replaces this list, so which
# words read as boilerplate — a judgement about report language, made per corpus — is a
# config edit. Which terms a small-context model actually sees is decided here, so this
# list is load-bearing (see build_nested's note on theme order).
ADMIN_STOP_WORDS = frozenset(
    {
        "zpráva",
        "projekt",
        "číslo",
        "datum",
        "rok",
        "strana",
        "tabulka",
        "příloha",
        "text",
        "obsah",
    }
)

# Reserved taxonomy_config.json keys. Anything starting with "_" is file-level settings
# (heslar_map, teater_branch_map, tie_break), never a theme.
SETTINGS_KEY = "_settings"
EXCLUDE_THEME = "__exclude__"
OTHER_THEME = "Other"

# Review status of one `_exclusions` entry — whose call the exclusion still is, not how
# it was reached. Config-authored so a reviewer can move a branch between "already
# ruled on" and "still open" by editing the file, rather than a set in vocab_review.py
# that only a developer can reach (issue #6, O3/O4).
#
#   settled           already ruled on (M4/M3) and correctly excluded; not an open
#                     question. The default, so an entry that states only a reason
#                     keeps its current meaning.
#   open_geo_ethnic   O3/O4 "Q1": conflicts with the prompt's standing geographic
#                     guardrail ("country name, language name, or geographic region
#                     name"). Reinstating one REQUIRES relaxing that wording in the
#                     same change, or the prompt forbids what the vocabulary offers.
#   open_other        O4 "Q2": excluded on an a priori read with no guardrail conflict
#                     at all. Under M3 the default is keep, so the burden here runs
#                     the other way.
EXCLUSION_STATUSES = ("settled", "open_geo_ethnic", "open_other")
DEFAULT_EXCLUSION_STATUS = "settled"

# What counts as a composite label when looking for "X/Y" entries whose components are
# also offered on their own (issue #6, O1/F). Only "/" is in use, but both sources are
# hand-maintained and a second convention appearing in a later harvest is a config
# edit, not a code change -- override with `_settings.composite_separators`.
DEFAULT_COMPOSITE_SEPARATORS = ("/",)

# Every key a taxonomy_overrides.json entry may carry, besides the "match" block that
# names the record. Closed on purpose: an unrecognised key is almost always a typo or a
# convention someone invented, and either way it silently does nothing.
OVERRIDE_KEYS = frozenset({"facet", "sub", "qualifier_cs", "same_as", "same_as_suppress", "reason"})

# Which para_config.txt [components] entry each vocabulary source is declared as. Both
# are CC BY-NC 4.0 and both are declared *conditional*, meaning they only constrain a
# run's effective licence when log_component() actually names them — so a run that
# injects this vocabulary into its prompt and never logs the component under-reports
# its own licence. Kept here, beside the sources themselves, so the mapping travels
# with the vocabulary core into atrium-llm-enrich rather than being re-derived there.
PARADATA_COMPONENTS = {"amcr": "amcr_vocab", "teater": "teater_data"}


def vocabulary_provenance(vocab_path: str) -> Dict[str, Any]:
    """Identity of a built vocabulary, read from the ``*.meta.json`` beside it.

    ``vocab_build.py`` has always written this sidecar — the taxonomy files' sha256s,
    the tool version, and per-source harvest facts including TEATER's pinned
    ``snapshot_ref`` — and nothing has ever read it, so an enrichment run recorded
    *that* it used a vocabulary but never *which* one (issue #6, D3).

    Returns ``{}`` when there is no sidecar: the legacy flat vocabulary and an
    auto-synced one have none, and a run must not fail over missing provenance. Keys:

      ``vocab``        the artifact's own path, tool version and term count
      ``taxonomy``     ``{config_sha256, overrides_sha256}`` — the two files that
                       decide every placement
      ``sources``      per source: records, licence and pinned ref
      ``components``   para_config component names to pass to ``log_component``
    """
    meta_path = Path(vocab_path).with_suffix(".meta.json")
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as fh:
        meta = json.load(fh)

    sources = {}
    components = []
    for source in meta.get("sources") or []:
        name = str(source.get("name") or "")
        if not name:
            continue
        sources[name] = {
            "records": source.get("records"),
            "license": source.get("license"),
            # TEATER pins a commit; AMCR has no equivalent, so the harvest endpoint is
            # the closest thing to a version it can offer.
            "ref": source.get("snapshot_ref") or source.get("endpoint"),
        }
        if name in PARADATA_COMPONENTS:
            components.append(PARADATA_COMPONENTS[name])

    return {
        "vocab": {
            "path": str(vocab_path),
            "meta_path": str(meta_path),
            "tool_version": meta.get("tool_version"),
            "terms": (meta.get("counts") or {}).get("total"),
        },
        "taxonomy": {
            "config_sha256": (meta.get("taxonomy_config") or {}).get("sha256"),
            "overrides_sha256": (meta.get("taxonomy_overrides") or {}).get("sha256"),
        },
        "sources": sources,
        "components": sorted(components),
    }


# The prompt has forbidden the model from selecting a country, language or region name
# since long before the vocabulary excluded those branches. The two are one decision
# held in two places, and either half can be changed without the other: relax the prompt
# and nothing offers the terms; reinstate the branches and the prompt forbids what the
# vocabulary offers, so the scores measure the contradiction rather than the model.
#
# `_settings.geo_guardrail` keeps them in step, and is the only place either half is
# declared: `active` says whether the prompt carries the clause, `prompt_markers` is how
# to recognise it (a reworded prompt is then a config edit, not a code change), and
# `covers` names the rules the wording actually reaches. Enforced at BUILD time only —
# the prompt text itself stays in llm_run.py, this never rewrites it.
DEFAULT_GEO_GUARDRAIL_MARKERS = ("country name", "language name", "geographic region name")
GEO_ETHNIC_STATUS = "open_geo_ethnic"

# Which keys of a harvested record survive into an on-disk entry. `sub` is the source's
# own second level, and is what lets the prompt render facets with sub-headers;
# `source`/`source_id` identify the surviving record and `discarded_ids` every record
# dedup absorbed into it (M7), which together let build_system_prompt assemble the full
# id set for `teater_category_ids`; `bare_cs` names the unqualified label of a
# qualifier-split entry (B3). Overridable with `_settings.nested_keep` — dropping a key
# shrinks the prompt payload, adding one exposes a harvested field the entry does not
# carry today, and neither needs a code change. `discarded_ids` and `bare_cs` are the
# two the enrichment output reads back, so dropping either changes behaviour, not just
# size.
DEFAULT_NESTED_KEEP = ("cs", "en", "sub", "source", "source_id", "discarded_ids", "bare_cs")

# Sentinel for "take the value from _settings". Distinct from None, which several of
# these parameters already use to mean something specific (`keep=None` = keep every
# key), so it cannot be spelled as a default of None.
_FROM_CONFIG: Any = object()


def _canonical_exclusion_key(key: str) -> str:
    """Fold an ``_exclusions`` key onto the rule string :meth:`assign_theme` reports.

    The canonical spelling is that rule string — ``heslar:<list>`` or ``teater:<id>``
    — so a note can be looked up by the exact rule that excluded a term, with no
    second convention to keep in step. Two legacy spellings from the file's first
    version still resolve, because rewriting a hand-maintained file is not a reason
    to invalidate someone's local edit: a bare AMCR list name (``zeme``) and the
    space-separated branch form (``TEATER 288``).
    """
    text = str(key).strip()
    lowered = text.lower()
    if lowered.startswith("teater:") or lowered.startswith("heslar:"):
        kind, _, ident = text.partition(":")
        return f"{kind.lower()}:{ident.strip()}"
    if lowered.startswith("teater "):
        return "teater:" + text.split(None, 1)[1].strip()
    if text.isdigit():
        return f"teater:{text}"
    return f"heslar:{text}"


def _norm(value: Optional[str]) -> str:
    """Fold a Czech label for cross-source matching (NFC, casefold, NBSP, whitespace)."""
    if not value:
        return ""
    text = unicodedata.normalize("NFC", value).replace("\xa0", " ")
    return " ".join(text.casefold().split())


def _term_sort_key(
    item: Tuple[str, Dict[str, Any]],
    stop_words: Optional[Iterable[str]] = None,
) -> Tuple[int, int, str]:
    """Order terms inside a theme: content before boilerplate, then the curators' own
    ``razeni`` where AMCR supplied one, then the label.

    ``stop_words`` defaults to :data:`ADMIN_STOP_WORDS`; ``build_nested`` passes the
    config's own list.
    """
    cs_key, pair = item
    words = ADMIN_STOP_WORDS if stop_words is None else stop_words
    is_admin = 1 if any(aw in cs_key.lower() for aw in words) else 0
    sort = pair.get("sort") if isinstance(pair, dict) else None
    return (is_admin, sort if isinstance(sort, int) else 10**9, cs_key)


class VocabularyManager:
    # API constants
    AMCR_OAI_BASE = "https://api.aiscr.cz/2.2/oai"
    AMCR_NS = {
        "oai": "http://www.openarchives.org/OAI/2.0/",
        "amcr": "https://api.aiscr.cz/schema/amcr/2.2/",
    }

    def __init__(
        self,
        vocab_path: str = "data_samples/vocab/union_nested.json",
        config_path: str = "data_samples/taxonomy_config.json",
        overrides_path: Optional[str] = None,
        llm_predictor: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.vocab_path = Path(vocab_path)
        self.config_path = Path(config_path)
        self.overrides_path = (
            Path(overrides_path)
            if overrides_path is not None
            else self.config_path.with_name("taxonomy_overrides.json")
        )
        self.taxonomy: Dict[str, Any] = self._load_config()
        self.overrides: Dict[Tuple[str, str], Dict[str, Any]] = self._load_overrides()
        self.vocab_data: Dict[str, Any] = {}
        self.llm_predictor = llm_predictor
        self._prompt_string_cache: Optional[str] = None

    def _invalidate_cache(self) -> None:
        self._prompt_string_cache = None

    def _load_config(self) -> Dict[str, Any]:
        if self.config_path.exists():
            with open(self.config_path, "r", encoding="utf-8") as f:
                return json.load(f)

        print(f"[vocab] Warning: {self.config_path} not found. Using built-in default taxonomy.")
        return {
            "Site Types": {
                "priority": 10,
                "keywords": {
                    "cs": [
                        "hradiště",
                        "pohřebiště",
                        "sídliště",
                        "hrad",
                        "tvrz",
                        "kostel",
                        "mohyla",
                        "studna",
                        "depot",
                        "jáma",
                        "příkop",
                        "val",
                        "sklep",
                        "zaniklá",
                        "opevnění",
                        "areál",
                        "objekt",
                        "zásobní",
                    ]
                },
            },
            "Find Types": {
                "priority": 8,
                "keywords": {
                    "cs": [
                        "keramika",
                        "kost",
                        "hrob",
                        "záušnice",
                        "nůž",
                        "brousek",
                        "bronz",
                        "kámen",
                        "sklo",
                        "mazanice",
                        "nádoba",
                        "střep",
                        "oštěp",
                        "jehlice",
                        "mlat",
                        "zásobnice",
                        "kachel",
                        "konstrukční prvek",
                        "navážka",
                        "malta",
                        "cihla",
                        "glazura",
                        "zlomek",
                        "fragment",
                        "dno",
                        "okraj",
                        "ucho",
                        "výduť",
                    ]
                },
            },
            "Methods": {
                "priority": 9,
                "keywords": {
                    "cs": [
                        "povrchový sběr",
                        "plošný odkryv",
                        "sonda",
                        "výkop",
                        "průzkum",
                        "dokumentace",
                        "geodetický",
                        "stavebně-historický",
                        "záchranný",
                        "badatelský",
                        "dohled",
                        "terénní",
                        "revize",
                    ]
                },
            },
            "Chronology": {
                "priority": 11,
                "keywords": {
                    "cs": [
                        "středověk",
                        "eneolit",
                        "paleolit",
                        "neolit",
                        "bronzová",
                        "halštatská",
                        "laténská",
                        "novověk",
                        "pravěk",
                        "datum",
                        "přesné datum",
                        "někdy v letech",
                        "stol",
                        "století",
                    ]
                },
            },
            "Location & Admin": {
                "priority": 6,
                "keywords": {
                    "cs": [
                        "katastrální",
                        "parcela",
                        "okres",
                        "obec",
                        "lokalita",
                        "poloha",
                        "mapa",
                        "mapový",
                        "sekce",
                    ]
                },
            },
            "Documentation": {
                "priority": 7,
                "keywords": {
                    "cs": [
                        "fotografie",
                        "plán",
                        "kresba",
                        "zpráva",
                        "hlášení",
                        "nálezová",
                        "příloha",
                        "plánek",
                        "negativy",
                        "diapozitiv",
                    ]
                },
            },
            "Finds Context": {
                "priority": 8,
                "keywords": {
                    "cs": [
                        "ojedinělý nález",
                        "náhodný nález",
                        "nález v druhotné",
                        "záchranný nález",
                        "pohřeb",
                        "kostrový",
                        "žárový",
                    ]
                },
            },
        }

    def _load_overrides(self) -> Dict[Tuple[str, str], Dict[str, Any]]:
        """Per-(source, id) corrections from ``taxonomy_overrides.json`` (issue #6,
        workstream B4). Absent file means no overrides — every consumer already treats
        ``self.overrides`` as an empty mapping in that case, so this is not an error.
        """
        if not self.overrides_path.exists():
            return {}
        with open(self.overrides_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        out: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for entry in payload.get("overrides", []):
            match = entry.get("match") or {}
            source = str(match.get("source") or "")
            source_id = str(match.get("id") or "")
            if not source or not source_id:
                continue
            out[(source, source_id)] = {k: v for k, v in entry.items() if k != "match"}
        return out

    def qualifier_overrides(self) -> Dict[Tuple[str, str], str]:
        """``(source, id) -> qualifier_cs`` for records flagged to become their own
        bracketed enum entry (B3), for :func:`vocab_sources.to_term_pairs`. Kept as a
        plain dict rather than passing ``self`` into vocab_sources, which must stay
        import-light (see the module docstring there)."""
        return {
            key: value["qualifier_cs"]
            for key, value in self.overrides.items()
            if value.get("qualifier_cs")
        }

    def geo_guardrail(self) -> Dict[str, Any]:
        """The ``_settings.geo_guardrail`` block, with defaults filled in. See
        :data:`DEFAULT_GEO_GUARDRAIL_MARKERS`."""
        block = self.settings.get("geo_guardrail")
        block = block if isinstance(block, dict) else {}
        markers = tuple(str(m) for m in (block.get("prompt_markers") or ()) if str(m))
        return {
            "active": bool(block.get("active", True)),
            "prompt_markers": markers or DEFAULT_GEO_GUARDRAIL_MARKERS,
            "covers": tuple(str(r) for r in (block.get("covers") or ()) if str(r)),
        }

    def geo_guardrail_problems(self, prompt_source: Optional[str] = None) -> List[str]:
        """Every way the guardrail and the vocabulary currently contradict each other.

        ``prompt_source`` is the text of the module that builds the system prompt; pass
        ``None`` to check only the vocabulary half (the prompt is not this package's to
        read in every context). Returns an empty list when the two agree.
        """
        guard = self.geo_guardrail()
        active = guard["active"]
        problems: List[str] = []

        if prompt_source is not None:
            present = all(m.lower() in prompt_source.lower() for m in guard["prompt_markers"])
            if active and not present:
                problems.append(
                    "geo_guardrail.active is true, but the prompt no longer forbids "
                    f"selecting {', '.join(guard['prompt_markers'])} — the excluded "
                    "geographic branches are now excluded for no stated reason"
                )
            elif present and not active:
                problems.append(
                    "geo_guardrail.active is false, but the prompt still forbids "
                    f"selecting {', '.join(guard['prompt_markers'])} — relaxing the "
                    "wording is what makes reinstating those branches meaningful"
                )

        if active:
            maps = {
                "heslar": self.settings.get("heslar_map") or {},
                "teater": self.settings.get("teater_branch_map") or {},
            }
            for rule in guard["covers"]:
                kind, _, ident = rule.partition(":")
                if maps.get(kind, {}).get(ident) != EXCLUDE_THEME:
                    problems.append(
                        f"{rule} is offered to the model while the prompt still forbids "
                        "selecting one — reinstate it and relax the guardrail wording in "
                        "the same change, or neither"
                    )
        return problems

    def admin_stop_words(self) -> frozenset:
        """Words that sort a term to the back of its theme, from ``_settings``. See
        :data:`ADMIN_STOP_WORDS`."""
        raw = self.settings.get("admin_stop_words")
        return frozenset(str(w).lower() for w in raw if str(w)) if raw else ADMIN_STOP_WORDS

    def nested_keep(self) -> Tuple[str, ...]:
        """Keys that survive into an on-disk entry, from ``_settings``. See
        :data:`DEFAULT_NESTED_KEEP`."""
        raw = self.settings.get("nested_keep")
        keys = tuple(str(k) for k in (raw or ()) if str(k))
        return keys or DEFAULT_NESTED_KEEP

    def composite_separators(self) -> Tuple[str, ...]:
        """What splits a composite ``"X/Y"`` label, from ``_settings``. See
        :data:`DEFAULT_COMPOSITE_SEPARATORS`."""
        raw = self.settings.get("composite_separators")
        seps = tuple(str(s) for s in (raw or ()) if str(s))
        return seps or DEFAULT_COMPOSITE_SEPARATORS

    def same_as_overrides(
        self,
    ) -> Tuple[List[List[Tuple[str, str]]], List[List[Tuple[str, str]]]]:
        """``(extra, suppress)`` id-pairs for :func:`attach_same_as`, from
        ``taxonomy_overrides.json``.

        An override entry may carry ``same_as`` (link this record to each listed
        ``{source, id}``, even though no label connects them) and ``same_as_suppress``
        (do not link it to these, whatever the labels look like). Both are declared
        from one side only — a link is symmetric, so stating it twice would just be
        two places to forget.
        """
        extra: List[List[Tuple[str, str]]] = []
        suppress: List[List[Tuple[str, str]]] = []
        for (source, source_id), override in self.overrides.items():
            for key, bucket in (("same_as", extra), ("same_as_suppress", suppress)):
                for other in override.get(key) or []:
                    pair = sorted(
                        {
                            (source, source_id),
                            (str(other.get("source") or ""), str(other.get("id") or "")),
                        }
                    )
                    if len(pair) == 2:  # a self-link is a no-op, not an error
                        bucket.append(pair)
        return sorted(extra), sorted(suppress)

    def exclusion_notes(self) -> Dict[str, Dict[str, str]]:
        """``rule -> {"reason", "status"}`` for every documented exclusion.

        Keyed by the exact rule string :meth:`assign_theme` returns, so a report can
        look a note up by the rule that excluded the term instead of guessing at a key
        spelling (see :func:`_canonical_exclusion_key`). An entry may be a bare reason
        string — the shape the file shipped with — in which case the status is
        ``DEFAULT_EXCLUSION_STATUS``, i.e. "already ruled on"; that is what a note
        written before statuses existed meant.
        """
        out: Dict[str, Dict[str, str]] = {}
        for key, value in (self.settings.get("_exclusions") or {}).items():
            if isinstance(value, dict):
                note = {
                    "reason": str(value.get("reason") or ""),
                    "status": str(value.get("status") or DEFAULT_EXCLUSION_STATUS),
                }
            else:
                note = {"reason": str(value or ""), "status": DEFAULT_EXCLUSION_STATUS}
            out[_canonical_exclusion_key(key)] = note
        return out

    def is_excluded(self, term_pair: Dict[str, Any]) -> bool:
        """True if this record's own placement rule — override, heslar_map, or
        teater_branch_map — resolves to ``__exclude__``, ignoring the keyword fallback
        and cross-source rescue (neither ever yields ``__exclude__``, both are opt-in
        aids for a term no curated list already claims).

        Used to filter records out of a source's list *before* label-collision dedup
        (issue #6, finding 3 of comment 5395681950): without this, a term whose winning
        record sits in an excluded list is lost entirely even when a kept list also
        offers it under the same label.
        """
        return self.assign_theme(term_pair)[0] == EXCLUDE_THEME

    def fetch_amcr_vocab(self, delay: float = 0.3) -> Dict[str, Dict[str, str]]:
        term_mapping: Dict[str, Dict[str, str]] = {}
        url = f"{self.AMCR_OAI_BASE}?verb=ListRecords&metadataPrefix=oai_amcr&set=heslo"
        page = 0
        MAX_PAGES = 500

        print("[AMCR] Starting OAI-PMH harvest via GET requests…")
        session = requests.Session()
        session.headers.update({"User-Agent": "ATRIUM-vocabulary-manager/1.3"})

        while url and page < MAX_PAGES:
            page += 1
            print(f"  [AMCR] Fetching page {page}…")

            try:
                resp = session.get(url, timeout=60)
                resp.raise_for_status()
            except requests.RequestException as exc:
                print(f"  [AMCR] Network error on page {page}: {exc}")
                break

            try:
                root = ET.fromstring(resp.content)
            except ET.ParseError as exc:
                print(f"  [AMCR] XML parse error on page {page}: {exc}")
                break

            amcr_ns = self.AMCR_NS["amcr"]
            xml_lang = "{http://www.w3.org/XML/1998/namespace}lang"

            for record in root.iter(f"{{{self.AMCR_NS['oai']}}}record"):
                for heslo_block in record.iter(f"{{{amcr_ns}}}heslo"):
                    cs_text = en_text = ""
                    for child in heslo_block:
                        if child.tag == f"{{{amcr_ns}}}heslo" and child.get(xml_lang) == "cs":
                            cs_text = (child.text or "").strip()
                        elif child.tag == f"{{{amcr_ns}}}heslo_en":
                            en_text = (child.text or "").strip()

                    if cs_text and en_text:
                        term_mapping[cs_text] = {"cs": cs_text, "en": en_text}

            rt_elem = root.find(f".//{{{self.AMCR_NS['oai']}}}resumptionToken")
            if rt_elem is not None and rt_elem.text and rt_elem.text.strip():
                token = rt_elem.text.strip()
                url = f"{self.AMCR_OAI_BASE}?verb=ListRecords&resumptionToken={urllib.parse.quote(token)}"
                time.sleep(delay)
            else:
                url = None  # type: ignore[assignment]

        print(f"[AMCR] Harvest complete. {len(term_mapping)} terms collected.")
        return term_mapping

    # ── taxonomy accessors ──────────────────────────────────────────────────
    @property
    def settings(self) -> Dict[str, Any]:
        """File-level settings from taxonomy_config.json's "_settings" block."""
        value = self.taxonomy.get(SETTINGS_KEY)
        return value if isinstance(value, dict) else {}

    def themes(self) -> Dict[str, Any]:
        """The theme definitions, excluding "_"-prefixed settings keys."""
        return {k: v for k, v in self.taxonomy.items() if not k.startswith("_")}

    def _theme_order(self) -> List[str]:
        """Themes in the order they are emitted and matched: priority descending.

        An explicit "tie_break" list in "_settings" resolves equal priorities; without
        it, equal priorities fall back to theme name so the result never depends on JSON
        key order.
        """
        tie_break = self.settings.get("tie_break") or []
        themes = self.themes()

        def rank(name: str) -> Tuple[int, int, str]:
            priority = themes[name].get("priority", 0)
            explicit = tie_break.index(name) if name in tie_break else len(tie_break)
            return (-priority, explicit, name)

        return sorted(themes, key=rank)

    def assign_theme(self, term_pair: Dict[str, Any]) -> Tuple[str, str]:
        """Place one term, returning ``(theme, rule)``.

        Rules are tried in precedence order and the winning one is reported, so every
        placement in the built vocabulary can be traced back to the reason for it —
        which is what makes the result reviewable by a domain expert rather than
        something an LLM produced and nobody can audit.

          override the term's own (source, id) in taxonomy_overrides.json — checked
                   first, since it exists precisely to correct one record's placement
                   without changing the list/branch rule every other member follows
          heslar   the AMCR controlled list the term came from (nazev_heslare)
          teater   the term's TEATER top-level branch
          keyword  the legacy substring match against taxonomy_config keywords
          other    nothing matched
        """
        scheme = str(term_pair.get("scheme") or "")
        source = str(term_pair.get("source") or "")
        source_id = str(term_pair.get("source_id") or "")

        override = self.overrides.get((source, source_id))
        if override and override.get("facet"):
            return override["facet"], f"override:{source}:{source_id}"

        if scheme and source == "amcr":
            mapped = (self.settings.get("heslar_map") or {}).get(scheme)
            if mapped:
                return mapped, f"heslar:{scheme}"

        if source == "teater":
            branches = self.settings.get("teater_branch_map") or {}
            # Walk the ancestor chain most-specific-first, so a depth-2 sub-branch
            # overrides its depth-1 parent. That is what lets branch 2557 be judged per
            # part — its regions and dynasties dropped while `etnika` could be kept —
            # without the map needing a code change to express it.
            chain = list(term_pair.get("broader") or ())
            candidates = [str(term_pair.get("source_id") or "")] + list(reversed(chain))
            for node in candidates:
                mapped = branches.get(node)
                if mapped:
                    return mapped, f"teater:{node}"

        for theme in self._theme_order():
            config = self.taxonomy[theme]
            for lang, keywords in config.get("keywords", {}).items():
                term_value = str(term_pair.get(lang, "") or "").lower()
                if not term_value:
                    continue
                for kw in keywords:
                    if kw.lower() in term_value:
                        return theme, f"keyword:{lang}:{kw}"

        return OTHER_THEME, "other"

    def _assign_theme(self, term_pair: Dict[str, str]) -> str:
        """Back-compat wrapper — the theme name only."""
        return self.assign_theme(term_pair)[0]

    def classify_with_llm(self, term_pair: Dict[str, str]) -> Optional[str]:
        if not self.llm_predictor:
            return None
        categories = list(self.taxonomy.keys())
        prompt = (
            f"Categorize this archaeological term: '{term_pair.get('cs', '')}' "
            f"(English: '{term_pair.get('en', '')}') "
            f"into one of the following exact categories: {categories}. "
            "Reply ONLY with the exact category name and nothing else."
        )
        try:
            response_text = self.llm_predictor(prompt).strip()
            for key in categories:
                if key.lower() == response_text.lower():
                    return key
        except Exception as exc:
            print(f"  [LLM] Classification error during taxonomy sync: {exc}")
        return None

    def validate_settings(self, records: Optional[Iterable[Dict[str, Any]]] = None) -> None:
        """Fail loudly on a hand-edit that would otherwise do nothing.

        ``build_nested`` used to create any theme a map named, via ``setdefault`` — so a
        typo like "Site Type" produced a phantom facet with no priority, no config, and
        last place in the truncation order, with nothing printed. Silent is the wrong
        failure mode for a file people are expected to hand-edit, and every check below
        guards the same shape of mistake: an edit that looks applied and is not.

        Every problem found is reported at once. A curator fixing one typo per run,
        each caught only after a full rebuild, is how a config file stops being worth
        editing.

        ``records`` — the harvested records this config will be applied to, when the
        caller has them — additionally reports overrides whose ``(source, id)`` matches
        nothing. That check is opt-in because the manager is routinely used without a
        harvest (prompt building, the single-source builds), where "no such record" is
        normal rather than a mistake.
        """
        known_facets = set(self.themes()) | {EXCLUDE_THEME}
        settings = self.settings
        bad: List[str] = []

        # ── facets: every name that claims to be one must be declared ──────────────
        for map_name in ("heslar_map", "teater_branch_map"):
            for key, value in (settings.get(map_name) or {}).items():
                if value not in known_facets:
                    bad.append(f"{map_name}[{key!r}] -> undeclared facet {value!r}")
        for name in settings.get("tie_break") or []:
            if name not in self.themes():
                bad.append(f"tie_break lists undeclared facet {name!r}")

        # ── labels: a relabel for a list nobody maps never renders ─────────────────
        for label_map, source_map in (
            ("heslar_labels", "heslar_map"),
            ("teater_branch_labels", "teater_branch_map"),
        ):
            mapped = set(settings.get(source_map) or {})
            for key in settings.get(label_map) or {}:
                if key not in mapped:
                    bad.append(f"{label_map}[{key!r}] names nothing in {source_map}")

        # ── exclusions: a reason must belong to a rule that excludes something ─────
        excluded_rules = {
            f"heslar:{k}"
            for k, v in (settings.get("heslar_map") or {}).items()
            if v == EXCLUDE_THEME
        } | {
            f"teater:{k}"
            for k, v in (settings.get("teater_branch_map") or {}).items()
            if v == EXCLUDE_THEME
        }
        for rule, note in self.exclusion_notes().items():
            if rule not in excluded_rules:
                bad.append(f"_exclusions[{rule!r}] documents a rule that excludes nothing")
            if note["status"] not in EXCLUSION_STATUSES:
                bad.append(
                    f"_exclusions[{rule!r}] status {note['status']!r} "
                    f"is not one of {list(EXCLUSION_STATUSES)}"
                )

        # ── the guardrail's scope and the exclusion register must agree ───────────
        # Two lists, on purpose: the register describes things that ARE excluded, and a
        # reinstated branch drops out of it — while `covers` has to outlive that, since
        # it is what says the branch conflicts with the prompt at all. Cross-checked
        # here so they can never drift, which is the only real cost of holding both.
        all_rules = {f"heslar:{k}" for k in (settings.get("heslar_map") or {})} | {
            f"teater:{k}" for k in (settings.get("teater_branch_map") or {})
        }
        covers = set(self.geo_guardrail()["covers"])
        for rule in sorted(covers - all_rules):
            bad.append(f"geo_guardrail.covers names {rule!r}, which no map places")
        for rule, note in self.exclusion_notes().items():
            if note["status"] == GEO_ETHNIC_STATUS and rule not in covers:
                bad.append(
                    f"_exclusions[{rule!r}] is {GEO_ETHNIC_STATUS} but geo_guardrail.covers "
                    "does not list it"
                )
        for rule in sorted(covers & excluded_rules):
            note = self.exclusion_notes().get(rule)
            if note and note["status"] != GEO_ETHNIC_STATUS:
                bad.append(
                    f"geo_guardrail.covers lists {rule!r} but _exclusions calls it "
                    f"{note['status']!r}, not {GEO_ETHNIC_STATUS!r}"
                )

        # ── overrides: keys, values, and the pair-level contradictions ─────────────
        known_ids = None
        if records is not None:
            known_ids = {
                (str(r.get("source") or ""), str(r.get("source_id") or "")) for r in records
            }
        extra, suppress = self.same_as_overrides()
        contradictory = {frozenset(map(tuple, p)) for p in extra} & {
            frozenset(map(tuple, p)) for p in suppress
        }
        for (source, source_id), override in self.overrides.items():
            where = f"overrides[{source}:{source_id}]"
            for key in override:
                if key not in OVERRIDE_KEYS:
                    bad.append(f"{where} has unknown key {key!r}; expected {sorted(OVERRIDE_KEYS)}")
            facet = override.get("facet")
            if facet is not None and facet not in known_facets:
                bad.append(f"{where} -> undeclared facet {facet!r}")
            for key in ("same_as", "same_as_suppress"):
                for other in override.get(key) or []:
                    if not (isinstance(other, dict) and other.get("source") and other.get("id")):
                        bad.append(f"{where}.{key} entry {other!r} needs both 'source' and 'id'")
            if known_ids is not None and (source, source_id) not in known_ids:
                bad.append(f"{where} matches no harvested record")
        for pair in sorted(contradictory, key=lambda p: sorted(p)):
            bad.append(f"same_as and same_as_suppress both name the pair {sorted(pair)}")

        if bad:
            raise ValueError(
                f"{self.config_path.name} / {self.overrides_path.name} will not do what "
                "they say:\n  - "
                + "\n  - ".join(sorted(bad))
                + f"\nDeclared facets: {sorted(self.themes())}"
            )

    def build_nested(
        self,
        raw_terms: Dict[str, Dict[str, Any]],
        use_llm_fallback: bool = False,
        keep: Any = _FROM_CONFIG,
        audit: Optional[List[Dict[str, Any]]] = None,
        rescue: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Group flat terms into the nested taxonomy. Pure: no network, no disk.

        ``keep`` limits which keys survive into each entry: left unset it is
        ``_settings.nested_keep``, falling back to :data:`DEFAULT_NESTED_KEEP`, which
        documents what each key is for; an explicit sequence overrides the config and
        ``None`` keeps every key. ``audit``, when given, receives one row per term
        recording the rule that placed it. ``rescue`` maps a normalised Czech label to
        a theme, used to place AMCR terms that only a cross-source TEATER match can
        explain.

        Theme order is priority-descending (never alphabetical): ``build_system_prompt``
        iterates this dict in insertion order and truncates a *prefix* of the resulting
        term list, so key order decides which themes survive a tight context budget.
        """
        themed: Dict[str, Dict[str, Any]] = {theme: {} for theme in self._theme_order()}
        themed.setdefault(OTHER_THEME, {})
        self.validate_settings()
        heslar_labels = self.settings.get("heslar_labels") or {}
        stop_words = self.admin_stop_words()
        if keep is _FROM_CONFIG:
            keep = self.nested_keep()

        for cs_key, pair in raw_terms.items():
            theme, rule = self.assign_theme(pair)

            if theme == OTHER_THEME and rescue:
                mapped = rescue.get(_norm(cs_key))
                if mapped:
                    theme, rule = mapped, "rescue:teater"

            if theme == OTHER_THEME and use_llm_fallback and self.llm_predictor:
                llm_theme = self.classify_with_llm(pair)
                if llm_theme and llm_theme in themed:
                    theme, rule = llm_theme, "llm"
                    print(f"  [LLM] Re-classified '{cs_key}' → {theme}")

            # Sub-header resolution, in the same precedence order as the facet itself:
            # a per-(source, id) override wins over the source's own scheme/branch name,
            # which is then relabelled through `heslar_labels`. Without the override
            # step a term moved by `facet` keeps its original list's sub-header, so the
            # prompt renders e.g. `kostel` under "Feature / areál aktivity" — the facet
            # a reviewer asked for, under the sub-header they moved it away from.
            # Computed once here, before the EXCLUDE_THEME guard, so the audit CSV and
            # the nested entry can never disagree about what the prompt renders.
            override = self.overrides.get(
                (str(pair.get("source") or ""), str(pair.get("source_id") or ""))
            )
            raw_sub = pair.get("sub") or ""
            sub = (override or {}).get("sub") or heslar_labels.get(raw_sub, raw_sub)

            if audit is not None:
                audit.append(
                    {
                        "cs": cs_key,
                        "en": pair.get("en", ""),
                        "source": pair.get("source", ""),
                        "source_id": pair.get("source_id", ""),
                        "scheme": pair.get("scheme", ""),
                        "sub": sub,
                        "theme": theme,
                        "placed_by": rule,
                    }
                )

            if theme == EXCLUDE_THEME:
                continue

            entry = dict(pair) if keep is None else {k: pair[k] for k in keep if k in pair}
            if sub and (keep is None or "sub" in keep):
                # Assigned rather than relabelled in place, so an override can also
                # *supply* a sub-header for a record whose source offers none.
                entry["sub"] = sub
            themed.setdefault(theme, {})[cs_key] = entry

        for theme in list(themed.keys()):
            themed[theme] = dict(
                sorted(themed[theme].items(), key=lambda item: _term_sort_key(item, stop_words))
            )

        return themed

    def sync_and_build_nested_taxonomy(self, use_llm_fallback: bool = False) -> None:
        print("[vocab] Syncing remote vocabularies…")
        raw_terms = self.fetch_amcr_vocab()
        self.vocab_data = self.build_nested(raw_terms, use_llm_fallback=use_llm_fallback)
        self._invalidate_cache()
        self.save()

    def load(self, auto_sync: bool = True) -> Dict[str, Any]:
        if not self.vocab_path.exists():
            if not auto_sync:
                raise FileNotFoundError(
                    f"Vocabulary {self.vocab_path} not found. Build it with "
                    "`python3 vocab_build.py` on a machine with network access, or point "
                    "VOCAB_PATH at an existing artifact. Refusing to start a multi-minute "
                    "OAI-PMH harvest in the middle of a pipeline run."
                )
            print(f"[vocab] {self.vocab_path} not found — triggering auto-sync.")
            self.sync_and_build_nested_taxonomy()
            return self.vocab_data

        with open(self.vocab_path, "r", encoding="utf-8") as f:
            self.vocab_data = json.load(f)

        self._invalidate_cache()

        known_old_keys = {"Archaeological Terms (AMCR)"}
        if set(self.vocab_data.keys()) <= known_old_keys:
            print(
                "[vocab] WARNING: Cached vocabulary is in the old flat format. "
                "Re-syncing to build thematic grouping based on external config."
            )
            self.sync_and_build_nested_taxonomy()

        return self.vocab_data

    def save(self) -> None:
        self.vocab_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.vocab_path, "w", encoding="utf-8") as f:
            json.dump(
                self.vocab_data,
                f,
                indent=4,
                ensure_ascii=False,
                sort_keys=False,
            )
        self._invalidate_cache()
        print(f"[vocab] Vocabulary cached to {self.vocab_path}")

    def vocab_statistics(self) -> Dict[str, int]:
        if not self.vocab_data:
            self.load()
        return {
            theme: len(terms) if isinstance(terms, dict) else 0
            for theme, terms in self.vocab_data.items()
            if not theme.startswith("_")
        }

    def get_prompt_string(self) -> str:
        if self._prompt_string_cache is not None:
            return self._prompt_string_cache

        if not self.vocab_data:
            self.load()

        self._prompt_string_cache = json.dumps(
            self.vocab_data,
            indent=2,
            ensure_ascii=False,
            sort_keys=False,
        )
        return self._prompt_string_cache


# ── composite-label equivalence (issue #6, finding 2 / workstream F, O1) ───────────
#
# AMCR packs related concepts into one entry — "most/brod", "muzeum/skanzen",
# "budova/stavba" — and where a component also exists as its own standalone entry,
# both are offered to the model as distinct, independently selectable answers to the
# same line. Under P1 dropping either is not an option (it would be editing what a
# source vocabulary offers), so the two are linked instead: `same_as` records that a
# term stands in for the same real-world thing another term does, purely so a scorer
# can treat either as correct. It changes nothing about the prompt or the schema —
# both entries were already independently selectable before this ran.
#
# Pure functions of the already-built nested shape, not of the flat records: a
# composite-component match can span two different source lists or even two
# different sources (AMCR "most/brod" vs. AMCR "most"; "muzeum/skanzen" vs. TEATER
# "skanzen"), so it can only be found once nesting has already resolved dedup and
# placement. find_composite_links() is the single definition both the actual
# attachment (below) and vocab_review.py's reviewable CSV read from, so the report
# a human reads and the vocabulary a model is offered can never quietly disagree
# about what counts as a pair.


def find_composite_links(
    nested: Dict[str, Dict[str, Any]],
    separators: Sequence[str] = DEFAULT_COMPOSITE_SEPARATORS,
) -> List[Tuple[str, str, str, str]]:
    """Every offered ``"X/Y[/Z]"`` label paired with a component also offered as its
    own standalone entry. Returns ``(composite_facet, composite_cs, component_facet,
    component_cs)`` tuples, one per pair, deduplicated and ordered by label so the
    result is deterministic regardless of dict iteration order.

    ``separators`` is what splits a composite label; see
    :data:`DEFAULT_COMPOSITE_SEPARATORS`.
    """
    seps = [str(s) for s in separators if str(s)] or list(DEFAULT_COMPOSITE_SEPARATORS)
    splitter = re.compile("|".join(re.escape(s) for s in seps))

    by_label: Dict[str, Tuple[str, str]] = {}
    for facet, terms in nested.items():
        if facet.startswith("_"):
            continue
        for cs in terms:
            by_label[_norm(cs)] = (facet, cs)

    seen: set = set()
    links: List[Tuple[str, str, str, str]] = []
    for facet, terms in nested.items():
        if facet.startswith("_"):
            continue
        for cs in terms:
            if not any(sep in cs for sep in seps):
                continue
            parts = [p.strip() for p in splitter.split(cs) if p.strip()]
            if len(parts) < 2:
                continue
            for part in parts:
                key = _norm(part)
                if key not in by_label or key == _norm(cs):
                    continue
                comp_facet, comp_cs = by_label[key]
                pair = (facet, cs, comp_facet, comp_cs)
                if pair in seen:
                    continue
                seen.add(pair)
                links.append(pair)

    links.sort(key=lambda t: (t[1], t[3]))
    return links


def attach_same_as(
    nested: Dict[str, Dict[str, Any]],
    separators: Sequence[str] = DEFAULT_COMPOSITE_SEPARATORS,
    extra: Iterable[Iterable[Tuple[str, str]]] = (),
    suppress: Iterable[Iterable[Tuple[str, str]]] = (),
) -> int:
    """Mutate ``nested`` in place: for every pair :func:`find_composite_links` finds,
    add a bidirectional ``same_as`` list of ``{source, id}`` to both entries. Returns
    the number of links attached (0 on a build with no composite/component overlap
    at all, e.g. a single-source nesting that never sees both sides of a pair).

    ``extra`` and ``suppress`` are the reviewer's two corrections to what the label
    shape alone can tell, each an iterable of ``((source, id), (source, id))`` pairs
    (order within a pair is irrelevant — a link is symmetric):

      extra      link two records the detector cannot see, because neither label
                 contains the other. Sorted before use, so a set is a valid argument
                 and the output stays byte-reproducible.
      suppress   drop a link the detector does find but a reviewer has judged wrong —
                 two senses that merely share a word. Applied to ``extra`` too, so a
                 contradictory pair of entries resolves the safe way (no link)
                 rather than by argument order.

    Reversible by construction: this only ever appends to a ``same_as`` list nothing
    else reads yet, so removing the call site drops the field from the next rebuild
    with no other change required — the same reversibility M7 asked for on
    ``discarded_ids``, extended to this field on the same reasoning.
    """
    pairs = list(find_composite_links(nested, separators))

    blocked = {frozenset(p) for p in suppress}
    if extra:
        index: Dict[Tuple[str, str], Tuple[str, str]] = {}
        for facet, terms in nested.items():
            if facet.startswith("_"):
                continue
            for cs, entry in terms.items():
                index[(entry.get("source", ""), entry.get("source_id", ""))] = (facet, cs)
        for pair in sorted(sorted(tuple(p)) for p in extra):
            if len(pair) != 2:
                continue
            loc_a, loc_b = index.get(tuple(pair[0])), index.get(tuple(pair[1]))
            # A pair naming a record this build does not offer (excluded, or from the
            # other source in a single-source nesting) is skipped, not an error: the
            # same overrides file drives the AMCR-only, TEATER-only and union builds.
            if loc_a and loc_b and loc_a != loc_b:
                pairs.append((*loc_a, *loc_b))

    count = 0
    for facet_a, cs_a, facet_b, cs_b in pairs:
        entry_a = nested[facet_a][cs_a]
        entry_b = nested[facet_b][cs_b]
        id_a = {"source": entry_a.get("source", ""), "id": entry_a.get("source_id", "")}
        id_b = {"source": entry_b.get("source", ""), "id": entry_b.get("source_id", "")}
        if frozenset({(id_a["source"], id_a["id"]), (id_b["source"], id_b["id"])}) in blocked:
            continue

        same_a = entry_a.setdefault("same_as", [])
        if id_b not in same_a:
            same_a.append(id_b)
            count += 1

        same_b = entry_b.setdefault("same_as", [])
        if id_a not in same_b:
            same_b.append(id_a)

    return count


if __name__ == "__main__":
    manager = VocabularyManager(
        vocab_path="data_samples/vocab/union_nested.json",
        config_path="data_samples/taxonomy_config.json",
        llm_predictor=None,
    )
    manager.sync_and_build_nested_taxonomy(use_llm_fallback=False)
    prompt_str = manager.get_prompt_string()
    print("\n[Preview of serialised LLM prompt string]")
    print(prompt_str[:500] + "\n… [truncated]")
    print("\n[Vocabulary statistics]")
    for theme, count in manager.vocab_statistics().items():
        print(f"  {theme}: {count} terms")
