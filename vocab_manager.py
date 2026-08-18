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
import time
import unicodedata
import urllib.parse
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import requests

# Administrative words that should sort to the back of their theme, so that prompt
# truncation drops boilerplate before it drops archaeological vocabulary. Module level
# because build_nested() is now a pure function of (taxonomy, terms) and must not depend
# on state built inside the sync path.
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


def _norm(value: Optional[str]) -> str:
    """Fold a Czech label for cross-source matching (NFC, casefold, NBSP, whitespace)."""
    if not value:
        return ""
    text = unicodedata.normalize("NFC", value).replace("\xa0", " ")
    return " ".join(text.casefold().split())


def _term_sort_key(item: Tuple[str, Dict[str, Any]]) -> Tuple[int, int, str]:
    """Order terms inside a theme: content before boilerplate, then the curators' own
    ``razeni`` where AMCR supplied one, then the label."""
    cs_key, pair = item
    is_admin = 1 if any(aw in cs_key.lower() for aw in ADMIN_STOP_WORDS) else 0
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
        vocab_path: str = "data_samples/teater_nested_vocab.json",
        config_path: str = "data_samples/taxonomy_config.json",
        llm_predictor: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.vocab_path = Path(vocab_path)
        self.config_path = Path(config_path)
        self.taxonomy: Dict[str, Any] = self._load_config()
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

          heslar   the AMCR controlled list the term came from (nazev_heslare)
          teater   the term's TEATER top-level branch
          keyword  the legacy substring match against taxonomy_config keywords
          other    nothing matched
        """
        scheme = str(term_pair.get("scheme") or "")
        source = str(term_pair.get("source") or "")

        if scheme and source == "amcr":
            mapped = (self.settings.get("heslar_map") or {}).get(scheme)
            if mapped:
                return mapped, f"heslar:{scheme}"

        if source == "teater":
            branches = self.settings.get("teater_branch_map") or {}
            chain = list(term_pair.get("broader") or ())
            root = chain[0] if chain else str(term_pair.get("source_id") or "")
            mapped = branches.get(root)
            if mapped:
                return mapped, f"teater:{root}"

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

    def build_nested(
        self,
        raw_terms: Dict[str, Dict[str, Any]],
        use_llm_fallback: bool = False,
        keep: Optional[Sequence[str]] = ("cs", "en"),
        audit: Optional[List[Dict[str, Any]]] = None,
        rescue: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """Group flat terms into the nested taxonomy. Pure: no network, no disk.

        ``keep`` limits which keys survive into each entry — the default ``("cs", "en")``
        reproduces the on-disk entry shape every downstream consumer expects. ``audit``,
        when given, receives one row per term recording the rule that placed it.
        ``rescue`` maps a normalised Czech label to a theme, used to place AMCR terms
        that only a cross-source TEATER match can explain.

        Theme order is priority-descending (never alphabetical): ``build_system_prompt``
        iterates this dict in insertion order and truncates a *prefix* of the resulting
        term list, so key order decides which themes survive a tight context budget.
        """
        themed: Dict[str, Dict[str, Any]] = {theme: {} for theme in self._theme_order()}
        themed.setdefault(OTHER_THEME, {})

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

            if audit is not None:
                audit.append(
                    {
                        "cs": cs_key,
                        "en": pair.get("en", ""),
                        "source": pair.get("source", ""),
                        "source_id": pair.get("source_id", ""),
                        "scheme": pair.get("scheme", ""),
                        "theme": theme,
                        "placed_by": rule,
                    }
                )

            if theme == EXCLUDE_THEME:
                continue

            entry = dict(pair) if keep is None else {k: pair[k] for k in keep if k in pair}
            themed.setdefault(theme, {})[cs_key] = entry

        for theme in list(themed.keys()):
            themed[theme] = dict(sorted(themed[theme].items(), key=_term_sort_key))

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


if __name__ == "__main__":
    manager = VocabularyManager(
        vocab_path="data_samples/teater_nested_vocab.json",
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
