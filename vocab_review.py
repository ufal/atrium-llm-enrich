#!/usr/bin/env python3
"""
vocab_review.py — read-only review artifacts for issue #6's open vocabulary questions.

Five independent reports. Each is a CSV a domain reviewer reads and turns into either
a ``taxonomy_overrides.json`` entry or a ``teater_branch_map``/``heslar_map`` flip.
Nothing here writes to the vocabulary itself, and none of the five guesses a semantic
verdict — they rank and surface candidates; a human still decides.

    python3 vocab_review.py --collisions    # M8: same-label groups that might be homonyms
    python3 vocab_review.py --composites    # O1/F: "X/Y" labels vs. standalone X, Y
    python3 vocab_review.py --exclusions    # O3/O4: impact of reinstating each excluded list
    python3 vocab_review.py --subbranches   # O3/O4: the same, one level finer-grained
    python3 vocab_review.py --reinstate     # O3/O4: usable_count / collisions / token delta
    python3 vocab_review.py --all           # all five, offline from the committed flat files

Offline and pure: reads ``data_samples/vocab/*_flat.json`` + the taxonomy config, never
touches the network, never writes anywhere but the report CSVs.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import vocab_build as vb
import vocab_sources as vs
from vocab_manager import (
    DEFAULT_COMPOSITE_SEPARATORS,
    DEFAULT_EXCLUSION_STATUS,
    VocabularyManager,
    find_composite_links,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_VOCAB_DIR = REPO_ROOT / "data_samples" / "vocab"
DEFAULT_CONFIG = REPO_ROOT / "data_samples" / "taxonomy_config.json"

# The only judgement call baked into the tooling, and it is a classification of
# Display labels for the `status` each `_exclusions` entry declares (see
# vocab_manager.EXCLUSION_STATUSES for what the three mean). The bucket itself is
# config-authored, not decided here: which exclusions are still open is a reviewer's
# call, and it used to be two hard-coded sets in this file that only a developer could
# edit. This mapping only spells the token out for the CSV, and names the decision the
# reviewer is being asked about.
STATUS_LABELS = {
    "settled": "settled (M4)",
    "open_geo_ethnic": "open: geographic/ethnic (O3/O4 Q1)",
    "open_other": "open: other (O4 Q2)",
}
# A per-term override exclusion has no `_exclusions` entry to carry a status — it is
# not a list-level ruling at all. Kept distinct so it is never misread as a blessed
# M4 list.
STATUS_OVERRIDE = "settled (per-term override)"
# The one status the tool has to act on rather than just print: a `guardrail_conflict`
# column marks the rows whose reinstatement also needs the prompt's geographic wording
# relaxed, so nobody flips one half of that pair on its own.
GEO_ETHNIC_STATUS = "open_geo_ethnic"


def _status_of(notes: Dict[str, Dict[str, str]], *rules: str) -> str:
    """The declared status of the first of ``rules`` that has an `_exclusions` note.

    Callers pass most-specific-first (a TEATER sub-branch, then its root), because a
    sub-branch nobody keyed separately is covered by its root's ruling.
    """
    for rule in rules:
        note = notes.get(rule)
        if note:
            return note.get("status", DEFAULT_EXCLUSION_STATUS)
    return DEFAULT_EXCLUSION_STATUS


# ── shared loading ───────────────────────────────────────────────────────────────


def _load_filtered(vocab_dir: Path, manager: VocabularyManager) -> Dict[str, List[vs.VocabRecord]]:
    """The same B5 exclusion-before-dedup view vocab_build.py builds from, offline."""
    per_source = vb._load_flat(vocab_dir)
    return {
        name: vb._filter_excluded(records, manager) for name, (records, _meta) in per_source.items()
    }


def _label_index(
    per_source: Dict[str, Tuple[List[vs.VocabRecord], Any]],
) -> Dict[Tuple[str, str], str]:
    """``(source, id) -> cs``, built from the FULL harvests, before exclusion filtering.

    Used only to render a readable ancestor chain (``broader_labels``) for a collision
    member. Built from the unfiltered harvest deliberately: an ancestor can itself sit
    in an excluded branch (e.g. a term under 2560 has 2557 as an ancestor, and 2557 is
    excluded too), and the label is still useful context for a reviewer even though
    that ancestor is not itself offered to the model.
    """
    index: Dict[Tuple[str, str], str] = {}
    for source, (records, _meta) in per_source.items():
        for r in records:
            if r.source_id:
                index[(source, r.source_id)] = r.cs
    return index


def _render_broader(record: vs.VocabRecord, label_index: Dict[Tuple[str, str], str]) -> str:
    """The ancestor chain as labels, root-first where the source stores it that way
    (TEATER), joined ``>``. Falls back to the raw id when a label is not in the index —
    a raw id is still more useful to a reviewer than silently dropping the ancestor."""
    if not record.broader:
        return ""
    return " > ".join(label_index.get((record.source, bid), bid) for bid in record.broader)


def _aat_verdict(members: Sequence[vs.VocabRecord]) -> str:
    """Classify a collision group by whether its members' Getty AAT alignments
    (``exact_match``) agree, disagree, or say nothing.

    Measured against the shipped union vocabulary's 127 differing-gloss groups:
    ``conflicting`` 3, ``agreeing`` 17, ``one_sided`` 61, ``none`` 46. Only the first
    two are strong signal on their own (``conflicting`` is homonym evidence,
    ``agreeing`` is "bulk-confirm, same concept"); ``one_sided``/``none`` still leave
    the verdict to ``dissimilarity`` and the gloss/note columns.

    Deliberately not full pairwise agreement — matching every set against the first
    non-empty one is what the verified 3/17 split above was computed against, and
    changing the rule would silently change those numbers.
    """
    sets = [frozenset(m.exact_match) for m in members if m.exact_match]
    if not sets:
        return "none"
    if len(sets) == 1:
        return "one_sided"
    first = sets[0]
    return "agreeing" if all(s & first for s in sets) else "conflicting"


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]], columns: Sequence[str]) -> None:
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=list(columns), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(buf.getvalue(), encoding="utf-8")
    print(f"  [write] {path}  ({len(rows)} rows)")


# ── M8: collision review (Track 2) ───────────────────────────────────────────────


def gloss_similarity(a: str, b: str) -> float:
    """How similar two English glosses are: 0.0 (nothing shared) to 1.0 (identical).

    ``max()`` of a character-level ratio and a whitespace-token overlap — the
    character ratio alone catches plural/spelling variance ('amphora'/'amphorae');
    the token overlap catches a reordered or partially-shared multi-word gloss a
    pure character diff would score as very different. Validated against the real
    data: the one confirmed homonym in the current build (`zámek`, 'lock'/'châteaux')
    scores 0.17 and ranks 10th-lowest of 127 real collision groups.
    """
    a_n, b_n = vs.norm_label(a), vs.norm_label(b)
    if a_n == b_n:
        return 1.0
    # SequenceMatcher.ratio() is NOT symmetric in general (its autojunk heuristic and
    # matching-block search depend on which string is "a"): comparing in a canonical
    # (sorted) order is what makes this function's result independent of call order —
    # load-bearing here, since group_dissimilarity iterates a *set* of glosses, whose
    # order Python does not guarantee.
    lo, hi = sorted((a_n, b_n))
    char_ratio = SequenceMatcher(None, lo, hi).ratio()
    a_tok, b_tok = set(a_n.split()), set(b_n.split())
    tok_ratio = len(a_tok & b_tok) / len(a_tok | b_tok) if (a_tok or b_tok) else 0.0
    return max(char_ratio, tok_ratio)


def group_dissimilarity(members: Sequence[vs.VocabRecord]) -> float:
    """The worst (lowest) pairwise gloss similarity across every distinct EN gloss in
    the group. One member with a wildly different gloss is enough to flag the whole
    group, even if the rest agree with each other."""
    glosses = sorted({vs.norm_label(m.en) for m in members})
    if len(glosses) < 2:
        return 1.0
    best = 1.0
    for i in range(len(glosses)):
        for j in range(i + 1, len(glosses)):
            best = min(best, gloss_similarity(glosses[i], glosses[j]))
    return best


def collision_review_rows(
    records: Sequence[vs.VocabRecord],
    manager: VocabularyManager,
    label_index: Optional[Dict[Tuple[str, str], str]] = None,
) -> List[Dict[str, Any]]:
    """One row per (group, member) for every same-label group with more than one
    distinct English gloss, groups ordered ascending by dissimilarity — the
    candidates most likely to be a genuine M8 homonym first. A group whose members
    all share one gloss is plain dedup (B1/B2 already handle it — no review needed)
    and is not included.

    Each row also carries the source fields ``to_term_pairs``/``build_nested`` drop —
    ``note_cs``, ``alt_cs``, ``alt_en``, ``de``, ``uri``, ``broader_labels``,
    ``exact_match`` — none of which reached a review sheet before, even though a
    reviewer decides a homonym verdict from exactly this kind of evidence: the one
    verdict already issued (``zámek``) turns on a Czech scope note ("zařízení k
    uzavírání dveří… pomocí klíče") that sat in the committed flat file the whole
    time. ``label_index``, when given, renders ``broader`` as readable labels instead
    of raw ids (see :func:`_render_broader`); omit it and ids are shown verbatim.

    ``aat_verdict`` (see :func:`_aat_verdict`) is a group-level signal, repeated on
    every member row, same as ``dissimilarity``.
    """
    label_index = label_index or {}
    groups = vs.group_by_label(records)
    scored: List[Tuple[float, str, List[vs.VocabRecord]]] = []
    for members in groups.values():
        if len({vs.norm_label(m.en) for m in members}) < 2:
            continue
        scored.append((group_dissimilarity(members), _aat_verdict(members), members))
    scored.sort(key=lambda t: (t[0], vs.norm_label(t[2][0].cs)))

    rows: List[Dict[str, Any]] = []
    for dissimilarity, aat_verdict, members in scored:
        for m in sorted(members, key=vs.record_sort_key):
            facet, rule = manager.assign_theme(m.as_dict())
            rows.append(
                {
                    "dissimilarity": round(dissimilarity, 2),
                    "aat_verdict": aat_verdict,
                    "cs": m.cs,
                    "source": m.source,
                    "id": m.source_id,
                    "en": m.en,
                    "scheme": m.scheme or "",
                    "sub": m.sub or "",
                    "current_facet": facet,
                    "placed_by": rule,
                    "note_cs": m.note_cs or "",
                    "alt_cs": "; ".join(m.alt_cs),
                    "alt_en": "; ".join(m.alt_en),
                    "de": m.de or "",
                    "uri": m.uri or "",
                    "broader_labels": _render_broader(m, label_index),
                    "exact_match": "; ".join(m.exact_match),
                    "verdict": "",  # human fills in: "" = same concept (default),
                    # or a qualifier_cs string for a genuine homonym (M8/B3)
                }
            )
    return rows


COLLISION_COLUMNS = [
    "dissimilarity",
    "aat_verdict",
    "cs",
    "source",
    "id",
    "en",
    "scheme",
    "sub",
    "current_facet",
    "placed_by",
    "note_cs",
    "alt_cs",
    "alt_en",
    "de",
    "uri",
    "broader_labels",
    "exact_match",
    "verdict",
]


# ── O1/F: composite-label overlap (Track 3) ──────────────────────────────────────


def composite_pair_rows(
    nested: Dict[str, Dict[str, Any]], manager: Optional[VocabularyManager] = None
) -> List[Dict[str, Any]]:
    """Every offered "X/Y[/Z]" label where at least one of X, Y, Z is ALSO offered as
    its own standalone entry — both are currently selectable answers for the same
    line, and only one is "correct" under an exact-match score (O1/F).

    Detection itself lives in :func:`vocab_manager.find_composite_links`, which is
    also what :func:`vocab_manager.attach_same_as` uses to write the ``same_as``
    links into the vocabulary — so this report and the mechanism it reports on can
    never disagree about what counts as a pair. This function only decorates each
    link with the two entries' source ids for the reviewer.

    Given a ``manager``, the sheet also uses that config's composite separators and
    reports each pair's ``link_status`` — ``auto`` as detected, ``suppressed`` where
    a reviewer has already ruled the pair wrong, ``manual`` for a link declared in
    ``taxonomy_overrides.json`` that no label shape implies. A reviewer's own
    verdicts are then visible on the sheet they review, rather than only in the
    built vocabulary.
    """
    separators = manager.composite_separators() if manager else DEFAULT_COMPOSITE_SEPARATORS
    extra, suppress = manager.same_as_overrides() if manager else ([], [])
    manual = {frozenset(p) for p in extra}
    blocked = {frozenset(p) for p in suppress}

    def _status(entry_a: Dict[str, Any], entry_b: Dict[str, Any]) -> str:
        key = frozenset(
            {
                (entry_a.get("source", ""), entry_a.get("source_id", "")),
                (entry_b.get("source", ""), entry_b.get("source_id", "")),
            }
        )
        if key in blocked:
            return "suppressed"
        return "manual" if key in manual else "auto"

    rows: List[Dict[str, Any]] = []
    seen: set = set()
    for facet, cs, comp_facet, comp_cs in find_composite_links(nested, separators):
        seen.add((facet, cs, comp_facet, comp_cs))
        entry = nested[facet][cs]
        comp_entry = nested[comp_facet][comp_cs]
        rows.append(
            {
                "composite_cs": cs,
                "composite_facet": facet,
                "composite_source": entry.get("source", ""),
                "composite_id": entry.get("source_id", ""),
                "component_cs": comp_cs,
                "component_facet": comp_facet,
                "component_source": comp_entry.get("source", ""),
                "component_id": comp_entry.get("source_id", ""),
                "same_facet": facet == comp_facet,
                "link_status": _status(entry, comp_entry),
            }
        )

    # A hand-declared link the detector cannot see has no row above; add it, so the
    # sheet lists every pair the built vocabulary actually carries.
    index = {
        (e.get("source", ""), e.get("source_id", "")): (facet, cs, e)
        for facet, terms in nested.items()
        if not facet.startswith("_")
        for cs, e in terms.items()
    }
    for pair in sorted(sorted(tuple(p)) for p in extra):
        located = [index.get(tuple(side)) for side in pair]
        if len(located) != 2 or not all(located):
            continue
        (facet, cs, entry), (comp_facet, comp_cs, comp_entry) = located
        if (facet, cs, comp_facet, comp_cs) in seen or (facet, cs) == (comp_facet, comp_cs):
            continue
        rows.append(
            {
                "composite_cs": cs,
                "composite_facet": facet,
                "composite_source": entry.get("source", ""),
                "composite_id": entry.get("source_id", ""),
                "component_cs": comp_cs,
                "component_facet": comp_facet,
                "component_source": comp_entry.get("source", ""),
                "component_id": comp_entry.get("source_id", ""),
                "same_facet": facet == comp_facet,
                "link_status": _status(entry, comp_entry),
            }
        )
    rows.sort(key=lambda r: (r["composite_cs"], r["component_cs"]))
    return rows


COMPOSITE_COLUMNS = [
    "composite_cs",
    "composite_facet",
    "composite_source",
    "composite_id",
    "component_cs",
    "component_facet",
    "component_source",
    "component_id",
    "same_facet",
    "link_status",
]


# ── O3/O4: TEATER sub-branch decomposition (Track 4, finer grain) ───────────────


def _teater_children_index(
    teater_records: Sequence[vs.VocabRecord],
) -> Dict[str, List[str]]:
    """parent id -> sorted list of immediate child ids, for the whole TEATER tree.

    Built by walking every record's own ``(broader..., own id)`` chain and recording
    each consecutive edge — a sub-branch node (e.g. 2560 ``etnika``) is itself a real
    TEATER record with its own outgoing/incoming edges, reachable the same way any
    leaf term's are, so no separate branch-root traversal is needed.
    """
    children: Dict[str, set] = {}
    for r in teater_records:
        chain = list(r.broader) + ([r.source_id] if r.source_id else [])
        for parent, child in zip(chain, chain[1:], strict=False):
            children.setdefault(parent, set()).add(child)
    return {
        parent: sorted(kids, key=lambda x: (int(x) if x.isdigit() else 0, x))
        for parent, kids in children.items()
    }


def _teater_subtree(teater_records: Sequence[vs.VocabRecord], node: str) -> List[vs.VocabRecord]:
    """Every record in ``node``'s subtree, ``node``'s own record included."""
    return [r for r in teater_records if r.source_id == node or node in r.broader]


def teater_subbranch_impact_rows(
    per_source: Dict[str, Tuple[List[vs.VocabRecord], Any]], manager: VocabularyManager
) -> List[Dict[str, Any]]:
    """One row per TEATER sub-branch beneath each currently-excluded DEPTH-1 root, not
    one row per root. ``exclusion_impact.csv`` shows branch 288 as a single ~967-term
    row; it decomposes into 289 "hraniční obory" (62) and 352 "příbuzné obory" (904) —
    904 of the ~967 hang off one child, and nothing in the coarser sheet shows that.
    ``assign_theme`` already walks the ancestor chain most-specific-first at any depth
    (``vocab_manager.py``), so mapping a sub-branch individually needs no code change —
    this sheet exists to make that granularity visible to a reviewer, not to enable it.

    Only iterates the depth-1 branch roots (identified the same way
    :func:`vocab_sources.group_by_label` identifies them: ``source == "teater"`` and
    ``broader == ()``) that are currently ``__exclude__`` — never recurses a second
    level. Without that guard, decomposing an already-finest-grain sub-branch like
    2560 ``etnika`` would explode into ~390 single-leaf "sub-branch" rows, one per
    individual ethnic-group name, which is not decomposition a reviewer needs; 2557's
    own five sub-branches ARE included once, since they corroborate what
    ``exclusion_impact.csv`` already lists individually.

    ``usable_count`` is a LOCAL dedup count (``group_by_label`` over just this
    sub-branch's own records) — it does not account for a term colliding with a label
    already offered elsewhere in the vocabulary. That global check belongs to
    ``reinstatement_preview.csv``, which runs the full pipeline for the branch-map
    entries themselves; this sheet's job is decomposition, not the final go/no-go
    number.
    """
    teater_records = per_source.get("teater", ([], {}))[0]
    branch_map = manager.settings.get("teater_branch_map") or {}
    labels = manager.settings.get("teater_branch_labels") or {}
    notes = manager.exclusion_notes()
    children_index = _teater_children_index(teater_records)
    by_id = {r.source_id: r for r in teater_records if r.source_id}

    depth1_ids = {r.source_id for r in teater_records if r.source_id and not r.broader}

    rows: List[Dict[str, Any]] = []
    for root in sorted(depth1_ids, key=lambda x: int(x) if x.isdigit() else 0):
        if branch_map.get(root) != "__exclude__":
            continue
        for child in children_index.get(root, []):
            subtree = _teater_subtree(teater_records, child)
            if not subtree:
                continue
            groups = vs.group_by_label(subtree)
            own = by_id.get(child)
            if labels.get(child):
                child_label = labels[child]
            elif own and own.en:
                child_label = f"{own.cs} / {own.en}"
            else:
                child_label = own.cs if own else child
            depth = (len(own.broader) + 1) if own else None
            samples = sorted({r.cs for r in subtree})[:10]
            rows.append(
                {
                    "root": root,
                    "root_label": labels.get(root, root),
                    "subbranch": child,
                    "subbranch_label": child_label,
                    "depth": depth if depth is not None else "",
                    "term_count": len(subtree),
                    "usable_count": len(groups),
                    "sample_terms": "; ".join(samples),
                    "would_be_facet": "",  # reviewer's call — Q1/Q2 decides this, not the tool
                    "guardrail_conflict": _status_of(notes, f"teater:{child}", f"teater:{root}")
                    == GEO_ETHNIC_STATUS,
                }
            )
    rows.sort(key=lambda r: (r["root"], -r["term_count"]))
    return rows


SUBBRANCH_COLUMNS = [
    "root",
    "root_label",
    "subbranch",
    "subbranch_label",
    "depth",
    "term_count",
    "usable_count",
    "sample_terms",
    "would_be_facet",
    "guardrail_conflict",
]


# ── O3/O4: exclusion impact (Track 4) ────────────────────────────────────────────


def _excluded_buckets(
    per_source: Dict[str, Tuple[List[vs.VocabRecord], Any]], manager: VocabularyManager
) -> Dict[str, List[vs.VocabRecord]]:
    """Every record whose own placement resolves to ``__exclude__``, grouped by the
    exact rule that excluded it. Shared by :func:`exclusion_impact_rows` (which reports
    per rule) and :func:`reinstatement_preview_rows` (which measures reinstating one),
    so the two sheets can never disagree about which records a rule covers.

    Includes ``override:``-driven exclusions, not just ``heslar:``/``teater:`` ones —
    a per-term exclusion added via ``taxonomy_overrides.json`` used to vanish from
    both sheets silently; latent today (no shipped override excludes anything), but a
    real gap the moment one does.
    """
    buckets: Dict[str, List[vs.VocabRecord]] = {}
    for records, _meta in per_source.values():
        for r in records:
            if not r.cs or not r.en:
                continue
            theme, rule = manager.assign_theme(r.as_dict())
            if theme != "__exclude__" or not rule.startswith(("heslar:", "teater:", "override:")):
                continue
            buckets.setdefault(rule, []).append(r)
    return buckets


def exclusion_impact_rows(
    per_source: Dict[str, Tuple[List[vs.VocabRecord], Any]], manager: VocabularyManager
) -> List[Dict[str, Any]]:
    """Per excluded list/branch: how many terms it holds, whether it is one of the
    geographic/ethnic ones the prompt's guardrail actually conflicts with (O4's Q1),
    and a sample so a reviewer does not have to go looking. Every list here is
    already a one-line switch in ``taxonomy_config.json`` — reinstating one is
    changing its value away from ``__exclude__``, nothing more.
    """
    buckets = _excluded_buckets(per_source, manager)
    labels = manager.settings.get("teater_branch_labels") or {}
    notes = manager.exclusion_notes()  # keyed by the same rule string as `buckets`

    rows: List[Dict[str, Any]] = []
    for rule, members in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        kind, ident = rule.split(":", 1)
        note = notes.get(rule) or {}
        if kind == "override":
            # ident is "source:source_id" here (rule = "override:source:source_id");
            # the reason lives on the override entry itself, not in _exclusions.
            src, _, oid = ident.partition(":")
            label = ident
            reason = (manager.overrides.get((src, oid)) or {}).get("reason", "")
            status = STATUS_OVERRIDE
        else:
            label = labels.get(ident, "") if kind == "teater" else ident
            reason = note.get("reason", "")
            status = STATUS_LABELS.get(
                note.get("status", DEFAULT_EXCLUSION_STATUS),
                STATUS_LABELS[DEFAULT_EXCLUSION_STATUS],
            )
        samples = sorted({m.cs for m in members})[:10]
        rows.append(
            {
                "status": status,
                "rule": rule,
                "kind": kind,
                "list_or_branch": ident,
                "label": label,
                "term_count": len(members),
                "stated_reason": reason,
                "sample_terms": "; ".join(samples),
            }
        )
    rows.sort(key=lambda r: (r["status"], -r["term_count"]))
    return rows


EXCLUSION_COLUMNS = [
    "status",
    "rule",
    "kind",
    "list_or_branch",
    "label",
    "term_count",
    "stated_reason",
    "sample_terms",
]


# ── O3/O4: reinstatement preview — what would actually change ───────────────────


def _approx_prompt_chars(cs: str, en: Optional[str]) -> int:
    """The same ``"{cs} ({en})"`` + separator shape ``build_system_prompt`` renders a
    term as, for a rough token-cost estimate. Not exact (no real tokenizer is
    available offline), but consistent with the ~3.35 chars/token ratio the digest
    already used for the shipped vocabulary's own size."""
    return len(f"{cs} ({en or ''})") + 2


def reinstatement_preview_rows(
    per_source: Dict[str, Tuple[List[vs.VocabRecord], Any]], manager: VocabularyManager
) -> List[Dict[str, Any]]:
    """For every rule ``exclusion_impact_rows`` lists, measure what reinstating it
    ALONE would actually change — without writing anything to ``taxonomy_config.json``
    or mutating ``manager`` in any way. Nothing here is a full ``_nest()`` re-run: the
    two numbers a global re-nest would tell you — how many NEW enum entries appear,
    and how many of the branch's labels are already offered elsewhere — are both
    exact set operations against the real, already-built baseline vocabulary, which
    is cheaper and cannot drift from what ``_nest`` would actually do, since dedup is
    itself just a label-equality grouping (:func:`vocab_sources.group_by_label`).

    ``raw_count`` vs ``usable_count`` is the discrepancy
    ``6.O3O4.decision-package.md`` had to explain in prose (1708 raw vs 1703 usable
    for Q2) — this sheet makes it a column pair instead, per rule:

    * ``usable_count`` — distinct normalised labels in the branch that are NOT
      already offered under any facet today. This is how many NEW enum entries
      reinstating the rule would add.
    * ``collides_with_offered`` — labels that ARE already offered. These do not add
      a new entry (B1 dedup would absorb the reinstated record as a
      ``discarded_id`` on whichever term already carries that label); it is still
      useful to a reviewer as a sign the branch overlaps existing content.

    ``would_be_facet`` is intentionally left blank — the tool never picks a facet on
    a reviewer's behalf (P2); it is the one column a human fills in once Q1/Q2 name a
    verdict, exactly like ``verdict`` in ``collision_review.csv``.
    """
    filtered = {
        name: vb._filter_excluded(records, manager) for name, (records, _meta) in per_source.items()
    }
    qualifiers = manager.qualifier_overrides()
    baseline_records = filtered.get("amcr", []) + filtered.get("teater", [])
    baseline_nested, _audit, _collisions = vb._nest(
        manager, baseline_records, qualifiers=qualifiers
    )
    baseline_labels = {vs.norm_label(cs) for terms in baseline_nested.values() for cs in terms}

    buckets = _excluded_buckets(per_source, manager)
    labels = manager.settings.get("teater_branch_labels") or {}
    notes = manager.exclusion_notes()

    rows: List[Dict[str, Any]] = []
    for rule, members in buckets.items():
        eligible = [r for r in members if r.cs and r.en]
        branch_labels = {vs.norm_label(r.cs) for r in eligible}
        collides = branch_labels & baseline_labels
        net_new = branch_labels - baseline_labels

        seen: set = set()
        delta_chars = 0
        for r in eligible:
            key = vs.norm_label(r.cs)
            if key in net_new and key not in seen:
                seen.add(key)
                delta_chars += _approx_prompt_chars(r.cs, r.en)

        kind, ident = rule.split(":", 1)
        label = labels.get(ident, "") if kind == "teater" else ident
        rows.append(
            {
                "rule": rule,
                "label": label or ident,
                "raw_count": len(eligible),
                "usable_count": len(net_new),
                "would_be_facet": "",  # reviewer's call — see docstring
                "collides_with_offered": len(collides),
                "prompt_token_delta": round(delta_chars / 3.35),
                "guardrail_conflict": _status_of(notes, rule) == GEO_ETHNIC_STATUS,
            }
        )
    rows.sort(key=lambda r: -r["usable_count"])
    return rows


REINSTATEMENT_COLUMNS = [
    "rule",
    "label",
    "raw_count",
    "usable_count",
    "would_be_facet",
    "collides_with_offered",
    "prompt_token_delta",
    "guardrail_conflict",
]


# ── D1: the specificity ladder (issue #6, O2 / workstream D) ─────────────────────


def specificity_pair_rows(
    nested: Dict[str, Dict[str, Any]],
    per_source: Dict[str, Tuple[List[vs.VocabRecord], Any]],
) -> List[Dict[str, Any]]:
    """Every offered term that is ALSO offered under one of its own broader terms.

    This is the measurement behind D1's "partial credit for a correct-but-less-specific
    term" question, and it is not a corner case: TEATER curates a real ``broader``
    hierarchy and the facet rollup keeps both tiers, so a large share of the TEATER side
    of the vocabulary sits directly under an ancestor the model could equally have
    picked. Under strict exact match a model answering ``třetihory`` where the gold
    label is ``paleogén`` scores exactly what one answering ``keramika`` scores — zero —
    and the metric cannot tell a near-miss from a category error.

    ``nearest_ancestor`` is the closest offered ancestor, not the branch root: the chain
    is walked from the term outward, so the row names the smallest step the model
    actually got wrong. ``rungs_above`` counts how many of the term's ancestors are
    offered at all, which is how far a "credit any ancestor" rule would reach.

    Decides nothing — the scoring rule is @motyc's and @david-spacil's call (M10). This
    sheet only says how many answers it would change.
    """
    by_norm: Dict[str, Tuple[str, str]] = {}
    teater_ids: Dict[str, Tuple[str, str]] = {}
    for facet, terms in nested.items():
        if facet.startswith("_"):
            continue
        for cs, entry in terms.items():
            by_norm[vs.norm_label(cs)] = (facet, cs)
            if entry.get("source") == "teater" and entry.get("source_id"):
                teater_ids[str(entry["source_id"])] = (facet, cs)

    records = {r.source_id: r for r in per_source.get("teater", ([], None))[0] if r.source_id}

    rows: List[Dict[str, Any]] = []
    for source_id, (facet, cs) in teater_ids.items():
        record = records.get(source_id)
        if not record:
            continue
        # Outward from the term: broader is root-first, so reversed() is nearest-first.
        offered_ancestors = [
            a for a in reversed(record.broader) if a in teater_ids and teater_ids[a][1] != cs
        ]
        if not offered_ancestors:
            continue
        anc_facet, anc_cs = teater_ids[offered_ancestors[0]]
        root_facet, root_cs = teater_ids[offered_ancestors[-1]]
        rows.append(
            {
                "cs": cs,
                "facet": facet,
                "source_id": source_id,
                "nearest_ancestor_cs": anc_cs,
                "nearest_ancestor_id": offered_ancestors[0],
                "outermost_offered_ancestor_cs": root_cs,
                "rungs_above": len(offered_ancestors),
                "same_facet": facet == anc_facet and facet == root_facet,
                "verdict": "",  # reviewer's: full / partial / no credit
            }
        )
    rows.sort(key=lambda r: (-r["rungs_above"], r["cs"]))
    return rows


SPECIFICITY_COLUMNS = [
    "cs",
    "facet",
    "source_id",
    "nearest_ancestor_cs",
    "nearest_ancestor_id",
    "outermost_offered_ancestor_cs",
    "rungs_above",
    "same_facet",
    "verdict",
]


# ── CLI ───────────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vocab_review.py",
        description="Read-only review artifacts for issue #6's open vocabulary questions.",
    )
    p.add_argument("--collisions", action="store_true", help="M8 homonym-review sheet")
    p.add_argument("--composites", action="store_true", help="O1/F composite-label overlap")
    p.add_argument("--exclusions", action="store_true", help="O3/O4 exclusion impact")
    p.add_argument(
        "--subbranches",
        action="store_true",
        help="O3/O4 TEATER sub-branch decomposition (finer grain than --exclusions)",
    )
    p.add_argument(
        "--reinstate",
        action="store_true",
        help="O3/O4 reinstatement preview: usable_count, collisions, token delta per rule",
    )
    p.add_argument(
        "--specificity",
        action="store_true",
        help="D1 specificity ladder: terms offered alongside their own broader term",
    )
    p.add_argument("--all", action="store_true", help="all six reports")
    p.add_argument("--vocab-dir", type=Path, default=DEFAULT_VOCAB_DIR)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument("--overrides", type=Path, default=None)
    return p


def main(argv: Any = None) -> int:
    args = build_parser().parse_args(argv)
    if not (
        args.collisions
        or args.composites
        or args.exclusions
        or args.subbranches
        or args.reinstate
        or args.specificity
        or args.all
    ):
        print(
            "Nothing to do — pass --collisions, --composites, --exclusions, "
            "--subbranches, --reinstate, --specificity, or --all."
        )
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

    per_source = vb._load_flat(args.vocab_dir)
    filtered = _load_filtered(args.vocab_dir, manager)

    if args.collisions or args.all:
        records = filtered.get("amcr", []) + filtered.get("teater", [])
        label_index = _label_index(per_source)
        rows = collision_review_rows(records, manager, label_index=label_index)
        _write_csv(args.vocab_dir / "collision_review.csv", rows, COLLISION_COLUMNS)

    if args.composites or args.all:
        qualifiers = manager.qualifier_overrides()
        records = filtered.get("amcr", []) + filtered.get("teater", [])
        nested, _audit, _collisions = vb._nest(manager, records, qualifiers=qualifiers)
        rows = composite_pair_rows(nested, manager)
        _write_csv(args.vocab_dir / "composite_pairs.csv", rows, COMPOSITE_COLUMNS)

    if args.exclusions or args.all:
        rows = exclusion_impact_rows(per_source, manager)
        _write_csv(args.vocab_dir / "exclusion_impact.csv", rows, EXCLUSION_COLUMNS)

    if args.subbranches or args.all:
        rows = teater_subbranch_impact_rows(per_source, manager)
        _write_csv(args.vocab_dir / "teater_subbranch_impact.csv", rows, SUBBRANCH_COLUMNS)

    if args.reinstate or args.all:
        rows = reinstatement_preview_rows(per_source, manager)
        _write_csv(args.vocab_dir / "reinstatement_preview.csv", rows, REINSTATEMENT_COLUMNS)

    if args.specificity or args.all:
        qualifiers = manager.qualifier_overrides()
        records = filtered.get("amcr", []) + filtered.get("teater", [])
        nested, _audit, _collisions = vb._nest(manager, records, qualifiers=qualifiers)
        rows = specificity_pair_rows(nested, per_source)
        _write_csv(args.vocab_dir / "specificity_pairs.csv", rows, SPECIFICITY_COLUMNS)

    return 0


if __name__ == "__main__":
    sys.exit(main())
