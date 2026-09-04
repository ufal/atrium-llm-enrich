#!/usr/bin/env python3
"""
corpus_review.py — corpus-occurrence evidence for issue #6's open vocabulary
questions, and the annotation workbook the D1/D2 evaluation rubric needs.

@motyc's own standard for excluding a TEATER branch is evidence from enrichment
results, not a priori judgement ("I would exclude branches primarily on the basis of
evidence from the enrichment results" — M3). This module produces exactly that
evidence, from whatever corpus is on disk when it runs.

How much that is worth depends entirely on what is present. The repository itself
tracks only three synthetic demo documents
(``data_samples/{ALTO,DOC_LINE_CATEG,UDP,...}/CTX00000000{1,2,3}`` — 69 words), on
which these reports are a smoke test and nothing more. The real reports
(``CTX192100040``, ``CTX195603828``, … — the 16 documents / ~618 lines the thread's
numbers refer to, ``agent_dev_logs/issues/2026-07-17.19.issue.open.md``) arrive as a
zip attachment on issue #19 and land in ``data_samples/DOC_LINE_CATEG`` +
``data_samples/UDP`` without being committed. When they are present the same commands
produce real evidence over the full corpus; nothing here needs changing between the
two cases, and no report states a fixed hit count, precisely so a number computed
over three placeholders can never be mistaken for one computed over real reports.

    python3 corpus_review.py --term-evidence     # per-term corpus hits
    python3 corpus_review.py --branch-evidence   # per-excluded-branch roll-up (O3/O4)
    python3 corpus_review.py --gold-workbook     # per-line annotation instrument (D1/D2)
    python3 corpus_review.py --all

Matches on LEMMA, not surface form — ``data_samples/UDP/*.conllu`` already carries
real UDPipe lemmas (``czech-pdt-ud-2.15-241121``), so no model run is needed to
compute this. Surface matching would both miss real hits (Czech is heavily inflected:
"středověku" never equals the vocabulary's "středověk") and risks manufacturing false
ones on a larger corpus by substring luck. See :func:`_form_lemma_map` for the
mechanism, and single-word vocabulary terms only — phrase-level matching
("zlomek keramiky") is future work; that limitation is reported in the output, not
hidden.

Every run prints its own corpus size (documents, tokens, distinct lemmas, terms hit)
before writing anything, so any figure taken from these sheets can be quoted together
with the corpus it came from. Read that line first: the same sheet over three
placeholder documents and over the full 19-document set are not comparable numbers,
and only the second is evidence in the sense M3 means.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import vocab_build as vb
import vocab_review as vr
import vocab_sources as vs
from vocab_manager import VocabularyManager

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_VOCAB_DIR = REPO_ROOT / "data_samples" / "vocab"
DEFAULT_CONFIG = REPO_ROOT / "data_samples" / "taxonomy_config.json"
DEFAULT_UDP_DIR = REPO_ROOT / "data_samples" / "UDP"
DEFAULT_LINES_DIR = REPO_ROOT / "data_samples" / "DOC_LINE_CATEG"

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


# ── CoNLL-U reading (pure stdlib — no new dependency) ────────────────────────────


@dataclass(frozen=True)
class ConlluToken:
    form: str
    lemma: str
    upos: str


def read_conllu_tokens(path: Path) -> List[ConlluToken]:
    """Minimal CoNLL-U reader: FORM/LEMMA/UPOS only, pure stdlib — the same manual
    tab-split parsing ``keywords.py._extract_lemmas_conllu`` already uses, so this
    module adds no new dependency. Multi-word-token and empty-node rows (ids
    containing ``-`` or ``.``) are skipped, matching UD convention: those regroup
    already-listed single-token rows and would double-count words if kept.
    """
    tokens: List[ConlluToken] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line or line.startswith("#"):
                continue
            cols = line.split("\t")
            if len(cols) < 10:
                continue
            tok_id = cols[0]
            if "-" in tok_id or "." in tok_id:
                continue
            tokens.append(ConlluToken(form=cols[1], lemma=cols[2], upos=cols[3]))
    return tokens


def _form_lemma_map(tokens: Sequence[ConlluToken]) -> Dict[str, str]:
    """casefolded FORM -> the most common LEMMA that FORM took in this document.

    Built once per document so a DOC_LINE_CATEG line's own raw text (which carries no
    morphological annotation of its own) can still be lemma-matched, without needing
    sentence-level alignment between the CoNLL-U file's own sentence boundaries and
    the line CSV's — they do not coincide even in the tiny synthetic corpus: CTX1's
    first UDPipe sentence spans what ``DOC_LINE_CATEG`` splits into two separate
    lines ("Výzkumná zpráva…Mezí" + "Terénní výzkum…2024." as one ``# text``, but two
    ``line_num`` rows).

    This is why lemma matching, not surface matching, is required: the line
    "Nalezená keramika pochází z raného středověku." never contains the vocabulary's
    own dictionary-form label "středověk" as a substring — only its genitive
    "středověku". Lemma matching finds it; surface matching does not.
    """
    counts: Dict[str, Counter] = defaultdict(Counter)
    for tok in tokens:
        if tok.lemma and tok.lemma != "_":
            counts[vs.norm_label(tok.form)][tok.lemma] += 1
    return {form: counter.most_common(1)[0][0] for form, counter in counts.items()}


def _line_lemmas(text: str, form_lemma: Dict[str, str]) -> List[str]:
    """The lemma for each word-like token in ``text``, via ``form_lemma`` — falling
    back to the casefolded surface word itself when the document's own UDPipe pass
    never saw that exact form (the parser can mis-segment, or a word can appear only
    once with an ambiguous lemma)."""
    out = []
    for word in _WORD_RE.findall(text):
        key = vs.norm_label(word)
        out.append(form_lemma.get(key, key))
    return out


# ── the offered vocabulary, single-word terms only ───────────────────────────────


def shipped_nested(vocab_dir: Path, manager: VocabularyManager) -> Dict[str, Dict[str, Any]]:
    """The same baseline the other O3/O4 sheets build from — today's actually-offered
    union vocabulary, exclusion and dedup already applied."""
    per_source = vb._load_flat(vocab_dir)
    filtered = {
        name: vb._filter_excluded(records, manager) for name, (records, _meta) in per_source.items()
    }
    qualifiers = manager.qualifier_overrides()
    records = filtered.get("amcr", []) + filtered.get("teater", [])
    nested, _audit, _collisions = vb._nest(manager, records, qualifiers=qualifiers)
    return nested


def _single_word_terms(nested: Dict[str, Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """``norm_label(word) -> [{cs, en, facet, source, source_id}, ...]`` for every
    OFFERED term that is exactly one word (a bracketed qualifier, e.g. "zámek (sídlo
    elity)", is stripped back to its bare label first via ``bare_cs``). Multi-word
    terms are excluded — matching them needs phrase-level detection this generator
    does not attempt (see the module docstring)."""
    index: Dict[str, List[Dict[str, Any]]] = {}
    for facet, terms in nested.items():
        if facet.startswith("_"):
            continue
        for cs, entry in terms.items():
            bare = entry.get("bare_cs") or cs
            if len(bare.split()) != 1:
                continue
            key = vs.norm_label(bare)
            index.setdefault(key, []).append(
                {
                    "cs": cs,
                    "en": entry.get("en", ""),
                    "facet": facet,
                    "source": entry.get("source", ""),
                    "source_id": entry.get("source_id", ""),
                }
            )
    return index


# ── report 1: per-term corpus evidence ───────────────────────────────────────────


def corpus_term_evidence_rows(
    nested: Dict[str, Dict[str, Any]], udp_dir: Path, lines_dir: Path
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """One row per single-word offered term: how many documents its lemma occurs in,
    how many times, and up to three example lines. Returns ``(rows, corpus_stats)`` —
    ``corpus_stats`` (document/token/distinct-lemma counts) belongs in the same issue
    comment as this sheet, so nobody mistakes 4 hits on 3 documents for a real signal.
    """
    term_index = _single_word_terms(nested)
    occurrences: Counter = Counter()
    doc_hits: Dict[str, set] = defaultdict(set)
    examples: Dict[str, List[str]] = defaultdict(list)

    conllu_files = sorted(udp_dir.glob("*.conllu")) if udp_dir.exists() else []
    all_lemmas: set = set()
    total_tokens = 0

    for conllu_path in conllu_files:
        doc_id = conllu_path.stem
        tokens = read_conllu_tokens(conllu_path)
        total_tokens += len(tokens)
        form_lemma = _form_lemma_map(tokens)

        doc_lemma_counts: Counter = Counter()
        for t in tokens:
            if t.lemma and t.lemma != "_":
                lemma_key = vs.norm_label(t.lemma)
                all_lemmas.add(lemma_key)
                doc_lemma_counts[lemma_key] += 1
        for key, cnt in doc_lemma_counts.items():
            if key in term_index:
                occurrences[key] += cnt
                doc_hits[key].add(doc_id)

        line_path = lines_dir / f"{doc_id}.csv"
        if line_path.exists():
            with open(line_path, "r", encoding="utf-8", newline="") as fh:
                for row in csv.DictReader(fh):
                    text = row.get("text", "") or ""
                    for lemma in _line_lemmas(text, form_lemma):
                        key = vs.norm_label(lemma)
                        if key in term_index and len(examples[key]) < 3:
                            examples[key].append(
                                f"{doc_id} p{row.get('page_num', '')}"
                                f" l{row.get('line_num', '')}: {text}"
                            )

    rows: List[Dict[str, Any]] = []
    for key, terms in term_index.items():
        for t in terms:
            rows.append(
                {
                    "cs": t["cs"],
                    "en": t["en"],
                    "facet": t["facet"],
                    "source": t["source"],
                    "source_id": t["source_id"],
                    "doc_count": len(doc_hits.get(key, ())),
                    "occurrences": occurrences.get(key, 0),
                    "example_lines": "; ".join(examples.get(key, [])),
                }
            )
    rows.sort(key=lambda r: (-r["occurrences"], r["cs"]))

    corpus_stats = {
        "documents": len(conllu_files),
        "total_tokens": total_tokens,
        "distinct_lemmas": len(all_lemmas),
        "terms_with_hits": sum(1 for r in rows if r["occurrences"] > 0),
        "vocabulary_single_word_terms": len(term_index),
    }
    return rows, corpus_stats


TERM_EVIDENCE_COLUMNS = [
    "cs",
    "en",
    "facet",
    "source",
    "source_id",
    "doc_count",
    "occurrences",
    "example_lines",
]


# ── report 2: per-excluded-branch roll-up (O3/O4) ────────────────────────────────


def corpus_branch_evidence_rows(
    per_source: Dict[str, Tuple[List[vs.VocabRecord], Any]],
    manager: VocabularyManager,
    udp_dir: Path,
    nested: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """The same lemma evidence, rolled up per currently-EXCLUDED rule instead of per
    offered term — the shape Q1/Q2 actually needs: "does this branch's content show up
    in real reports at all". Reuses :func:`vocab_review._excluded_buckets` so this can
    never disagree with ``exclusion_impact.csv``/``reinstatement_preview.csv`` about
    which records a rule covers.

    **The unique/shared split is the whole point, and it is not cosmetic.** A branch is
    only credited with evidence for a label no kept list already offers. Without that
    split the sheet argues the opposite of the truth on the single most sensitive row:
    ``heslar:zeme`` (the 249 country names, O3/O4 Q1, the branch the geographic
    guardrail exists for) scores 7 occurrences on the real corpus — every one of them
    the word ``malta``. In those lines ``malta`` is *mortar*
    ("spojovaných vápennou maltou"), an offered Material term (AMCR HES-000910 /
    HES-000992, TEATER 2499); the excluded record it is being credited to is the
    country ``Malta`` (HES-001366), which ``norm_label``'s casefold makes
    indistinguishable. Read raw, that row says country names are attested in
    archaeological reports. They are not.

    So ``unique_*`` is the honest headline and the sort key; ``shared_*`` is retained
    beside it because a reviewer should be able to see the contamination rather than
    take its removal on trust. Same trap, milder, in ``heslar:letiste``
    (``kladno``/``kolín`` are towns) and ``heslar:nalez_typ``
    (``objekt``/``předmět`` — the two most generic words in the domain).

    ``nested`` is the currently-offered vocabulary; when omitted nothing can be
    classified as shared and every hit is reported as unique, so callers that want the
    split must pass it (``main()`` always does).
    """
    buckets = vr._excluded_buckets(per_source, manager)
    offered = set(_single_word_terms(nested)) if nested else set()

    conllu_files = sorted(udp_dir.glob("*.conllu")) if udp_dir.exists() else []
    lemma_occurrences: Counter = Counter()
    lemma_docs: Dict[str, set] = defaultdict(set)
    for conllu_path in conllu_files:
        doc_id = conllu_path.stem
        for t in read_conllu_tokens(conllu_path):
            if t.lemma and t.lemma != "_":
                key = vs.norm_label(t.lemma)
                lemma_occurrences[key] += 1
                lemma_docs[key].add(doc_id)

    def _tally(keys: Sequence[str]) -> Tuple[int, int]:
        docs: set = set()
        for k in keys:
            docs |= lemma_docs.get(k, set())
        return sum(lemma_occurrences.get(k, 0) for k in keys), len(docs)

    labels = manager.settings.get("teater_branch_labels") or {}
    rows: List[Dict[str, Any]] = []
    for rule, members in buckets.items():
        single_word = {vs.norm_label(m.cs) for m in members if m.cs and len(m.cs.split()) == 1}
        hit_terms = sorted(k for k in single_word if lemma_occurrences.get(k, 0) > 0)
        unique_terms = [k for k in hit_terms if k not in offered]
        shared_terms = [k for k in hit_terms if k in offered]

        occurrences, doc_count = _tally(hit_terms)
        unique_occurrences, unique_doc_count = _tally(unique_terms)
        shared_occurrences, _shared_docs = _tally(shared_terms)

        kind, ident = rule.split(":", 1)
        label = labels.get(ident, "") if kind == "teater" else ident
        rows.append(
            {
                "rule": rule,
                "label": label or ident,
                "branch_single_word_terms": len(single_word),
                "hit_term_count": len(hit_terms),
                "unique_hit_term_count": len(unique_terms),
                "unique_occurrences": unique_occurrences,
                "unique_doc_count": unique_doc_count,
                "shared_hit_term_count": len(shared_terms),
                "shared_occurrences": shared_occurrences,
                "occurrences": occurrences,
                "doc_count": doc_count,
                "sample_unique_terms": "; ".join(unique_terms[:10]),
                "sample_shared_terms": "; ".join(shared_terms[:10]),
            }
        )
    # Sorted by the unique figure: a branch whose every hit is explained by an
    # already-offered label must not head the sheet.
    rows.sort(key=lambda r: (-r["unique_occurrences"], -r["occurrences"], r["rule"]))
    return rows


BRANCH_EVIDENCE_COLUMNS = [
    "rule",
    "label",
    "branch_single_word_terms",
    # Read these first: hits on labels no kept list already offers.
    "unique_hit_term_count",
    "unique_occurrences",
    "unique_doc_count",
    # Retained so the contamination is visible rather than silently removed.
    "shared_hit_term_count",
    "shared_occurrences",
    # Totals, kept last: unique + shared. Quoting these alone is how heslar:zeme
    # ends up looking like evidence that country names appear in reports.
    "hit_term_count",
    "occurrences",
    "doc_count",
    "sample_unique_terms",
    "sample_shared_terms",
]


# ── report 3: gold annotation workbook (D1/D2) ───────────────────────────────────


def gold_workbook_rows(
    nested: Dict[str, Dict[str, Any]], lines_dir: Path, udp_dir: Path, top_n: int = 5
) -> List[Dict[str, Any]]:
    """One row per ``DOC_LINE_CATEG`` line, with a picklist of candidate vocabulary
    terms whose lemma occurs in that line. The whole point is that an annotator PICKS
    from ``candidate_terms`` rather than typing a label freehand — a free-text gold
    field invites labels outside the enum, which is exactly what makes a run
    unscoreable later. ``model_category``/``model_keywords_cs`` are left blank: no
    ``KW_PER_DOC_LLM_*`` run exists in this repository to populate them from (D2's own
    corpus gap) — the columns exist so a future run can be merged in without changing
    this sheet's shape.
    """
    rows: List[Dict[str, Any]] = []
    line_files = sorted(lines_dir.glob("*.csv")) if lines_dir.exists() else []
    for line_path in line_files:
        doc_id = line_path.stem
        conllu_path = udp_dir / f"{doc_id}.conllu"
        form_lemma = (
            _form_lemma_map(read_conllu_tokens(conllu_path)) if conllu_path.exists() else {}
        )
        with open(line_path, "r", encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                text = row.get("text", "") or ""
                seen: List[str] = []
                for lemma in _line_lemmas(text, form_lemma):
                    key = vs.norm_label(lemma)
                    for t in _single_word_terms(nested).get(key, []):
                        if t["cs"] not in seen:
                            seen.append(t["cs"])
                rows.append(
                    {
                        "file_id": row.get("file", doc_id),
                        "page": row.get("page_num", ""),
                        "line": row.get("line_num", ""),
                        "original_text": text,
                        "categ": row.get("categ", ""),
                        "quality_score": row.get("quality_score", ""),
                        "model_category": "",
                        "model_keywords_cs": "",
                        "candidate_terms": "; ".join(seen[:top_n]),
                        "gold_category": "",
                        "gold_keywords_cs": "",
                        "annotator_note": "",
                    }
                )

    def _sort_key(r: Dict[str, Any]) -> Tuple[str, int, int]:
        page = int(r["page"]) if str(r["page"]).isdigit() else 0
        line = int(r["line"]) if str(r["line"]).isdigit() else 0
        return (r["file_id"], page, line)

    rows.sort(key=_sort_key)
    return rows


GOLD_WORKBOOK_COLUMNS = [
    "file_id",
    "page",
    "line",
    "original_text",
    "categ",
    "quality_score",
    "model_category",
    "model_keywords_cs",
    "candidate_terms",
    "gold_category",
    "gold_keywords_cs",
    "annotator_note",
]


# ── shared CSV writer (same shape as vocab_review._write_csv) ───────────────────


CORPUS_META_NAME = "corpus_review.meta.json"


def _read_corpus_meta(vocab_dir: Path) -> Optional[Dict[str, Any]]:
    """Corpus provenance of the sheets currently on disk, or None if unrecorded."""
    path = vocab_dir / CORPUS_META_NAME
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError):
        # A corrupt sidecar must not block a rebuild — it only guards, it is not data.
        return None


def _write_corpus_meta(vocab_dir: Path, stats: Dict[str, Any], sheets: Sequence[str]) -> None:
    """Record which corpus these sheets came from, so a later run over a smaller one
    is refused rather than silently overwriting real evidence with placeholder
    numbers (the real reports are untracked, so a clean checkout has only the three
    demo documents — running the generator there used to destroy the committed
    sheets). ``generated_utc`` is deliberately absent: it would make every rebuild a
    diff, the same reason vocab_build.py normalises timestamps out of its --check.
    """
    payload = {
        "documents": stats["documents"],
        "total_tokens": stats["total_tokens"],
        "distinct_lemmas": stats["distinct_lemmas"],
        "terms_with_hits": stats["terms_with_hits"],
        "vocabulary_single_word_terms": stats["vocabulary_single_word_terms"],
        "sheets": sorted(sheets),
    }
    path = vocab_dir / CORPUS_META_NAME
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"  [write] {path}")


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> None:
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"  [write] {path}  ({len(rows)} rows)")


# ── CLI ───────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="corpus_review.py",
        description=(
            "Corpus-occurrence evidence for issue #6's open questions, and the D1/D2 "
            "gold annotation workbook. A generator: see the module docstring for why "
            "it currently has almost no real text to run against."
        ),
    )
    p.add_argument("--term-evidence", action="store_true", help="per-term corpus hits")
    p.add_argument(
        "--branch-evidence", action="store_true", help="O3/O4 per-excluded-branch roll-up"
    )
    p.add_argument("--gold-workbook", action="store_true", help="D1/D2 per-line annotation sheet")
    p.add_argument("--all", action="store_true", help="all three reports")
    p.add_argument("--vocab-dir", type=Path, default=DEFAULT_VOCAB_DIR)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--overrides", type=Path, default=None)
    p.add_argument("--udp-dir", type=Path, default=DEFAULT_UDP_DIR)
    p.add_argument("--lines-dir", type=Path, default=DEFAULT_LINES_DIR)
    p.add_argument("--top-n", type=int, default=5, help="candidate terms per gold-workbook line")
    p.add_argument(
        "--force",
        action="store_true",
        help=(
            "overwrite sheets even when the corpus on disk is smaller than the one "
            "they were built from"
        ),
    )
    return p


def main(argv: Any = None) -> int:
    args = build_parser().parse_args(argv)
    if not (args.term_evidence or args.branch_evidence or args.gold_workbook or args.all):
        print("Nothing to do — pass --term-evidence, --branch-evidence, --gold-workbook, or --all.")
        return 2

    manager = VocabularyManager(
        config_path=str(args.config),
        overrides_path=str(args.overrides) if args.overrides else None,
    )
    manager.validate_settings()

    if (
        not (args.vocab_dir / "amcr_flat.json").exists()
        or not (args.vocab_dir / "teater_flat.json").exists()
    ):
        print(f"[error] no *_flat.json artifacts found in {args.vocab_dir}")
        return 1

    if not args.udp_dir.exists():
        print(f"[warn] {args.udp_dir} not found — reports will show zero occurrences")
    if not args.lines_dir.exists():
        print(f"[warn] {args.lines_dir} not found — line-level reports will be empty")

    nested = shipped_nested(args.vocab_dir, manager)

    # Printed for EVERY report, not just --term-evidence: a figure from any of these
    # sheets is only interpretable next to the corpus it was computed over, and the
    # difference between three placeholder documents and the full report set is the
    # difference between a smoke test and the evidence M3 asks for.
    rows, stats = corpus_term_evidence_rows(nested, args.udp_dir, args.lines_dir)
    print(
        f"  [corpus] {stats['documents']} document(s), {stats['total_tokens']} tokens, "
        f"{stats['distinct_lemmas']} distinct lemmas, "
        f"{stats['terms_with_hits']}/{stats['vocabulary_single_word_terms']} "
        "single-word terms hit"
    )

    if not args.force:
        previous = _read_corpus_meta(args.vocab_dir)
        if previous and previous.get("documents", 0) > stats["documents"]:
            print(
                f"[refuse] the committed sheets were built from "
                f"{previous['documents']} document(s); only {stats['documents']} "
                f"are on disk now.\n"
                f"         Writing would replace real evidence with a smaller "
                f"corpus's numbers.\n"
                f"         The real reports are not tracked in git (issue #19 "
                f"attachment) — restore them to\n"
                f"         {args.udp_dir} and {args.lines_dir}, or pass --force if "
                f"shrinking is intended."
            )
            return 1

    written: List[str] = []

    if args.term_evidence or args.all:
        _write_csv(args.vocab_dir / "corpus_term_evidence.csv", rows, TERM_EVIDENCE_COLUMNS)
        written.append("corpus_term_evidence.csv")

    if args.branch_evidence or args.all:
        per_source = vb._load_flat(args.vocab_dir)
        rows = corpus_branch_evidence_rows(per_source, manager, args.udp_dir, nested=nested)
        _write_csv(args.vocab_dir / "corpus_branch_evidence.csv", rows, BRANCH_EVIDENCE_COLUMNS)
        written.append("corpus_branch_evidence.csv")

    if args.gold_workbook or args.all:
        rows = gold_workbook_rows(nested, args.lines_dir, args.udp_dir, top_n=args.top_n)
        _write_csv(args.vocab_dir / "gold_workbook.csv", rows, GOLD_WORKBOOK_COLUMNS)
        written.append("gold_workbook.csv")

    # Merged with any previously recorded sheet list: running one report must not
    # drop the provenance of sheets an earlier run wrote from the same corpus.
    previous = _read_corpus_meta(args.vocab_dir) or {}
    if previous.get("documents") == stats["documents"]:
        written = list(set(written) | set(previous.get("sheets") or []))
    _write_corpus_meta(args.vocab_dir, stats, written)

    return 0


if __name__ == "__main__":
    sys.exit(main())
