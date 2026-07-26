"""
eval_metrics.py  –  Scoring primitives for the Document Understanding benchmark.

Implements the metric set decided in hub issue #22 for the head-to-head
OOTB-VLM vs. legacy ABBYY/ALTO pipeline comparison:

- **CER / WER** — character/word error rate (Levenshtein distance / reference length)
- **Normalized edit distance** — OmniDocBench-compatible (distance / max(len_ref, len_hyp))
- **Entity F1** — micro precision/recall/F1 over (type, text) entity tuples,
  aligned to the CNEC 2.0 / TEATER schema used by `atrium-nlp-enrich`
- **TEDS** — table structure similarity; optional dependency (`apted` + lxml),
  wrapped so the rest of the harness works without it

Pure-stdlib apart from the optional TEDS path; deterministic; no I/O.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

# ──────────────────────────────────────────────────────────────────────────────
# Text normalization
# ──────────────────────────────────────────────────────────────────────────────

def normalize_text(text: str, lowercase: bool = False, collapse_ws: bool = True) -> str:
    """NFC-normalize; optionally lowercase and collapse all whitespace runs to single spaces."""
    out = unicodedata.normalize("NFC", text)
    if lowercase:
        out = out.lower()
    if collapse_ws:
        out = " ".join(out.split())
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Edit distance (Levenshtein, two-row DP)
# ──────────────────────────────────────────────────────────────────────────────

def levenshtein(ref: Sequence, hyp: Sequence) -> int:
    """Levenshtein distance between two sequences (characters or word lists)."""
    if not ref:
        return len(hyp)
    if not hyp:
        return len(ref)
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, start=1):
        curr = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, start=1):
            cost = 0 if r == h else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    """Character error rate: distance / len(ref). Empty ref → 0.0 if hyp empty, else 1.0."""
    ref_n, hyp_n = normalize_text(ref), normalize_text(hyp)
    if not ref_n:
        return 0.0 if not hyp_n else 1.0
    return levenshtein(ref_n, hyp_n) / len(ref_n)


def wer(ref: str, hyp: str) -> float:
    """Word error rate: distance over whitespace tokens / word count of ref."""
    ref_w, hyp_w = normalize_text(ref).split(), normalize_text(hyp).split()
    if not ref_w:
        return 0.0 if not hyp_w else 1.0
    return levenshtein(ref_w, hyp_w) / len(ref_w)


def normalized_edit_distance(ref: str, hyp: str) -> float:
    """OmniDocBench-style NED: distance / max(len_ref, len_hyp); 0.0 for two empty strings."""
    ref_n, hyp_n = normalize_text(ref), normalize_text(hyp)
    denom = max(len(ref_n), len(hyp_n))
    if denom == 0:
        return 0.0
    return levenshtein(ref_n, hyp_n) / denom


# ──────────────────────────────────────────────────────────────────────────────
# Entity F1 (CNEC/TEATER-aligned tuples)
# ──────────────────────────────────────────────────────────────────────────────

Entity = Tuple[str, str]  # (entity_type, surface_text)


def entity_prf(ref_entities: Iterable[Entity], hyp_entities: Iterable[Entity]) -> Dict[str, float]:
    """Micro precision/recall/F1 over multisets of (type, normalized text) entity tuples."""
    ref_c = Counter((t, normalize_text(s, lowercase=True)) for t, s in ref_entities)
    hyp_c = Counter((t, normalize_text(s, lowercase=True)) for t, s in hyp_entities)
    tp = sum((ref_c & hyp_c).values())
    n_ref, n_hyp = sum(ref_c.values()), sum(hyp_c.values())
    precision = tp / n_hyp if n_hyp else 0.0
    recall = tp / n_ref if n_ref else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "n_ref": n_ref, "n_hyp": n_hyp}


# ──────────────────────────────────────────────────────────────────────────────
# TEDS (optional dependency)
# ──────────────────────────────────────────────────────────────────────────────

def teds_score(ref_html: str, hyp_html: str, structure_only: bool = False) -> float:
    """Tree-Edit-Distance-based Similarity between two HTML tables.

    Requires the optional `apted` and `lxml` packages; raises ImportError with
    installation guidance if they are absent so the core harness stays light.
    """
    try:
        from table_teds import TEDS  # local vendored scorer, added with the table track
    except ImportError as exc:  # pragma: no cover - optional path
        raise ImportError(
            "TEDS scoring needs the table track extras: pip install apted lxml "
            "and vendor table_teds.py (see hub issue #22 plan, Milestone 2)."
        ) from exc
    return TEDS(structure_only=structure_only).evaluate(hyp_html, ref_html)


# ──────────────────────────────────────────────────────────────────────────────
# Page / corpus scoring
# ──────────────────────────────────────────────────────────────────────────────

def score_page(ref_text: str, hyp_text: str) -> Dict[str, float]:
    """All transcription metrics for one page."""
    return {
        "cer": cer(ref_text, hyp_text),
        "wer": wer(ref_text, hyp_text),
        "ned": normalized_edit_distance(ref_text, hyp_text),
        "ref_chars": len(normalize_text(ref_text)),
        "hyp_chars": len(normalize_text(hyp_text)),
    }


def score_corpus(pages: List[Dict[str, float]], tiers: List[str]) -> Dict[str, Dict[str, float]]:
    """Aggregate per-page metric dicts into per-tier and overall macro averages.

    `pages` and `tiers` are parallel lists; tier keys mirror sample_stratify.py.
    """
    if len(pages) != len(tiers):
        raise ValueError("pages and tiers must be parallel lists of equal length")
    buckets: Dict[str, List[Dict[str, float]]] = {}
    for page, tier in zip(pages, tiers, strict=False):
        buckets.setdefault(tier, []).append(page)
        buckets.setdefault("overall", []).append(page)

    out: Dict[str, Dict[str, float]] = {}
    for tier, rows in buckets.items():
        agg: Dict[str, float] = {"n_pages": float(len(rows))}
        for key in ("cer", "wer", "ned"):
            agg[key] = sum(r[key] for r in rows) / len(rows)
        out[tier] = agg
    return out
