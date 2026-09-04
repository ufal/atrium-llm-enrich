#!/usr/bin/env python3
"""
vocab_build.py — build the flat and nested vocabulary artifacts.

Two stages, deliberately separable:

    stage 1  harvest   network   AMCR OAI-PMH + TEATER  ->  data_samples/vocab/*_flat.{json,csv}
    stage 2  nest      pure      flat + taxonomy_config ->  data_samples/vocab/*_nested.json

Stage 2 never touches the network, so once the flat artifacts are committed the whole
taxonomy can be re-tuned offline in under a second:

    python3 vocab_build.py --from-flat --stats          # what does the current config do?
    $EDITOR data_samples/taxonomy_config.json
    python3 vocab_build.py --from-flat --check          # did my edit move the artifact?

Exit codes: 0 ok / no drift · 1 drift, or a harvest returned nothing · 2 usage.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from configparser import ConfigParser
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import vocab_sources as vs
from vocab_manager import VocabularyManager, attach_same_as

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_VOCAB_DIR = REPO_ROOT / "data_samples" / "vocab"
DEFAULT_CONFIG = REPO_ROOT / "data_samples" / "taxonomy_config.json"
# The path every consumer already points at (llm_config.txt, service/api.py, the hub's
# e2e fixture). Kept stable so there is no flag day.
LEGACY_NESTED = REPO_ROOT / "data_samples" / "teater_nested_vocab.json"
# The module whose system prompt carries the geographic guardrail. Read, never written:
# the clause stays in Python and this is a gate, not a generator (issue #6, O4 / C1).
PROMPT_SOURCE = REPO_ROOT / "llm_run.py"

SCHEMA_VERSION = 1
CHECK_PLACEHOLDER = "<checked>"


def _tool_version() -> str:
    """Read the version from para_config.txt rather than duplicating it in code."""
    cfg = ConfigParser()
    try:
        cfg.read(REPO_ROOT / "para_config.txt", encoding="utf-8")
        return cfg.get("tool", "version", fallback="unknown")
    except Exception:
        return "unknown"


def _sha256(path: Path) -> str:
    if not path.exists():
        return "<builtin>"
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _rel(path: Path) -> str:
    return str(path.relative_to(REPO_ROOT)) if path.is_relative_to(REPO_ROOT) else str(path)


def _base_meta(config_path: Path, overrides_path: Optional[Path] = None) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generator": "vocab_build.py",
        "tool_version": _tool_version(),
        "generated_utc": _now(),
        "taxonomy_config": {"path": _rel(config_path), "sha256": _sha256(config_path)},
    }
    if overrides_path is not None:
        meta["taxonomy_overrides"] = {
            "path": _rel(overrides_path),
            "sha256": _sha256(overrides_path),
        }
    return meta


# ── writing, with --check ─────────────────────────────────────────────────────


def _normalise_for_check(text: str) -> str:
    """Blank the timestamp on both sides so --check compares content, not clock."""
    out = []
    for line in text.splitlines(keepends=True):
        if '"generated_utc"' in line:
            indent = line[: len(line) - len(line.lstrip())]
            comma = "," if line.rstrip().endswith(",") else ""
            out.append(f'{indent}"generated_utc": "{CHECK_PLACEHOLDER}"{comma}\n')
        else:
            out.append(line)
    return "".join(out)


def _emit(path: Path, text: str, check: bool) -> bool:
    """Write ``text`` to ``path``. Returns True when the content changed.

    In ``check`` mode nothing is written; the return value reports drift.
    """
    old = path.read_text(encoding="utf-8") if path.exists() else None
    changed = old is None or _normalise_for_check(old) != _normalise_for_check(text)
    if check:
        if changed:
            print(f"  [drift] {path}")
        return changed
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"  [write] {path}{'' if changed else ' (unchanged)'}")
    return changed


def _json_text(payload: Any, sort_keys: bool) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=sort_keys) + "\n"


# ── stage 1 ───────────────────────────────────────────────────────────────────


def _harvest(args: argparse.Namespace) -> Dict[str, Tuple[List[vs.VocabRecord], Dict[str, Any]]]:
    sources = []
    if args.source in ("both", "amcr") and not args.skip_amcr:
        sources.append("amcr")
    if args.source in ("both", "teater") and not args.skip_teater:
        sources.append("teater")
    if not sources:
        raise SystemExit("nothing to harvest — every source was skipped")
    return vs.harvest(sources, delay=args.delay, teater_mode=args.teater_mode)


def _load_flat(vocab_dir: Path) -> Dict[str, Tuple[List[vs.VocabRecord], Dict[str, Any]]]:
    out: Dict[str, Tuple[List[vs.VocabRecord], Dict[str, Any]]] = {}
    for name in ("amcr", "teater"):
        path = vocab_dir / f"{name}_flat.json"
        if path.exists():
            records, file_meta = vs.read_flat_json(path)
            # Unwrap to the per-source block so --from-flat and a live harvest produce
            # byte-identical downstream metadata.
            sources = file_meta.get("sources") or []
            source_meta = sources[0] if sources else {"name": name, "strategy": "flat"}
            source_meta = dict(source_meta)
            out[name] = (records, source_meta)
            print(f"[flat] {name}: {len(records)} terms from {path.name}")
    if not out:
        raise SystemExit(f"no *_flat.json artifacts found in {vocab_dir} — run a harvest first")
    return out


# ── stage 2 ───────────────────────────────────────────────────────────────────


def _rescue_map(teater: Sequence[vs.VocabRecord]) -> Dict[str, str]:
    """Normalised TEATER label -> its top-level branch id.

    Used to place AMCR terms the keyword matcher missed but which TEATER already
    classifies — the concrete mechanism behind "228 of those terms have a proper home in
    TEATER's own hierarchy".
    """
    out: Dict[str, str] = {}
    for r in teater:
        key = vs.norm_label(r.cs)
        if key and key not in out and r.scheme:
            out[key] = r.scheme
    return out


def _nest(
    manager: VocabularyManager,
    records: Sequence[vs.VocabRecord],
    rescue_branches: Optional[Dict[str, str]] = None,
    qualifiers: Optional[Dict[Tuple[str, str], str]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Tuple[str, str, str]]]:
    branch_map = manager.settings.get("teater_branch_map") or {}
    rescue = None
    if rescue_branches:
        rescue = {
            label: branch_map[branch]
            for label, branch in rescue_branches.items()
            if branch in branch_map
        }
    audit: List[Dict[str, Any]] = []
    collisions: List[Tuple[str, str, str]] = []
    pairs = vs.to_term_pairs(records, collisions=collisions, qualifiers=qualifiers)
    if collisions:
        print(f"  [warn] {len(collisions)} label collision(s) dropped, e.g. {collisions[:3]}")
    nested = manager.build_nested(pairs, audit=audit, rescue=rescue)
    return nested, audit, collisions


def _check_geo_guardrail(manager: VocabularyManager) -> None:
    """Refuse to build a vocabulary that contradicts the prompt's geographic guardrail.

    The clause ("NEVER select a country name, language name, or geographic region
    name") and the excluded geographic branches are one decision held in two files.
    Change either half alone and the scores measure the contradiction rather than the
    model — which is why the O3/O4 decision package puts the two in the same change.
    The build is where both halves are in scope, so it is where they are checked.

    A missing prompt module is not an error: this repo's vocabulary artifacts are
    copied into atrium-llm-enrich, where the prompt lives elsewhere. The vocabulary
    half is still checked there.
    """
    source = PROMPT_SOURCE.read_text(encoding="utf-8") if PROMPT_SOURCE.exists() else None
    problems = manager.geo_guardrail_problems(source)
    if problems:
        raise SystemExit(
            "[guardrail] the vocabulary and the prompt disagree:\n  - "
            + "\n  - ".join(problems)
            + "\nSee data_samples/vocab/6.O3O4.decision-package.md; flip both halves or "
            "neither."
        )


def _filter_excluded(
    records: Sequence[vs.VocabRecord], manager: VocabularyManager
) -> List[vs.VocabRecord]:
    """Drop records whose own placement rule already resolves to ``__exclude__``.

    Applied *before* label-collision dedup (issue #6, finding 3 of comment
    5395681950): without this, a term whose winning record sits in an excluded list
    is lost entirely even when a kept list also offers it under the same label — e.g.
    "papír" would vanish once ``dokument_material`` is excluded, because that record's
    ident happens to sort first, even though "papír" also lives in the kept Material
    lists.
    """
    return [r for r in records if not manager.is_excluded(r.as_dict())]


_AUDIT_COLUMNS = (
    "cs",
    "en",
    "source",
    "source_id",
    "scheme",
    "sub",
    "theme",
    "placed_by",
    "same_as_count",
)


def _audit_text(rows: Sequence[Dict[str, Any]]) -> str:
    """``sub`` (added alongside ``theme``/``placed_by`` in ``build_nested``) and
    ``same_as_count`` close the audit/nested split a reviewer used to have to bridge by
    hand: before this, the audit CSV had the placement *reason* but not the sub-header
    a term actually renders under, and the nested JSON had ``sub``/``same_as`` but not
    the reason. ``same_as_count`` defaults to 0 via ``.get`` — ``build_nested`` cannot
    know it (composite links are only resolved after nesting, by
    :func:`vocab_manager.attach_same_as`; see ``main()`` below), so callers that build
    an audit list directly from ``build_nested`` and never enrich it still get a valid
    CSV rather than a ``KeyError``.
    """
    import io

    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(_AUDIT_COLUMNS)
    for row in sorted(rows, key=lambda r: (r["source"], r["theme"], r["cs"])):
        writer.writerow(
            [row.get(c, "") if c != "same_as_count" else row.get(c, 0) for c in _AUDIT_COLUMNS]
        )
    return buf.getvalue()


def _with_same_as_counts(
    audit: Sequence[Dict[str, Any]], nested: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Join each audit row to its ``same_as`` link count in the just-nested vocabulary.

    Keyed on ``(theme, cs)`` — audit's ``theme`` is exactly which facet the term landed
    in and ``cs`` is the enum key (post-qualifier-split), so ``nested[theme][cs]`` is
    the same entry the row describes; audit and nested are 1:1 by construction (every
    surviving term gets exactly one audit row and one nested entry). Must run after
    :func:`vocab_manager.attach_same_as`, which is what populates ``same_as`` at all.
    """
    out = []
    for row in audit:
        entry = nested.get(row["theme"], {}).get(row["cs"], {})
        out.append({**row, "same_as_count": len(entry.get("same_as") or [])})
    return out


def _stats(nested: Dict[str, Any], audit: Sequence[Dict[str, Any]]) -> None:
    total = sum(len(v) for v in nested.values())
    print(f"\n  {'theme':<24} {'terms':>6}   share")
    for theme, terms in nested.items():
        share = (len(terms) / total * 100) if total else 0.0
        print(f"  {theme:<24} {len(terms):>6}   {share:5.1f}%")
    print(f"  {'TOTAL':<24} {total:>6}")
    by_rule: Dict[str, int] = {}
    for row in audit:
        by_rule[row["placed_by"].split(":")[0]] = by_rule.get(row["placed_by"].split(":")[0], 0) + 1
    print("\n  placed by rule: " + ", ".join(f"{k}={v}" for k, v in sorted(by_rule.items())))


# ── main ──────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vocab_build.py",
        description="Harvest and nest the AMCR + TEATER controlled vocabularies.",
    )
    p.add_argument("--source", choices=("both", "amcr", "teater"), default="both")
    p.add_argument("--skip-amcr", action="store_true", help="translator-compatible alias")
    p.add_argument("--skip-teater", action="store_true", help="translator-compatible alias")
    p.add_argument(
        "--from-flat",
        action="store_true",
        help="nest from committed flat artifacts instead of harvesting (offline)",
    )
    p.add_argument("--teater-mode", choices=("snapshot", "live"), default="snapshot")
    p.add_argument("--vocab-dir", type=Path, default=DEFAULT_VOCAB_DIR)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    p.add_argument(
        "--overrides",
        type=Path,
        default=None,
        help="per-term corrections (default: taxonomy_overrides.json next to --config)",
    )
    p.add_argument("--delay", type=float, default=vs.DEFAULT_DELAY)
    p.add_argument(
        "--update-legacy",
        action="store_true",
        help=f"also write the union nesting to {LEGACY_NESTED.name} (the path consumers use)",
    )
    p.add_argument("--check", action="store_true", help="report drift, write nothing")
    p.add_argument("--stats", action="store_true", help="print per-theme counts")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    vocab_dir: Path = args.vocab_dir
    changed = False

    per_source = _load_flat(vocab_dir) if args.from_flat else _harvest(args)

    for name, (records, _meta) in per_source.items():
        if not records:
            print(f"[error] {name} harvest returned no records — refusing to overwrite.")
            return 1

    # ── stage 1 artifacts ────────────────────────────────────────────────────
    if not args.from_flat:
        for name, (records, source_meta) in per_source.items():
            meta = _base_meta(args.config)
            meta["sources"] = [source_meta]
            meta["counts"] = {"total": len(records)}
            changed |= _emit(
                vocab_dir / f"{name}_flat.json", vs.flat_json_text(records, meta), args.check
            )
            changed |= _emit(vocab_dir / f"{name}_flat.csv", vs.flat_csv_text(records), args.check)

    # ── stage 2 setup ────────────────────────────────────────────────────────
    manager = VocabularyManager(
        config_path=str(args.config),
        overrides_path=str(args.overrides) if args.overrides else None,
    )
    _check_geo_guardrail(manager)

    # The build is the one place that has every harvested record, so it is the one
    # place that can tell an override whose (source, id) no longer matches anything —
    # a list renumbered upstream, or a typo'd id that has silently done nothing since
    # the day it was written.
    manager.validate_settings(
        records=[r.as_dict() for records, _meta in per_source.values() for r in records]
    )
    qualifiers = manager.qualifier_overrides()

    # B5 (issue #6, finding 3): drop excluded records *before* any dedup, so a term
    # whose winning record sits in an excluded list doesn't take a kept list's record
    # down with it. Applied once, upstream of both vocabulary.csv and the nested build.
    filtered_source: Dict[str, List[vs.VocabRecord]] = {
        name: _filter_excluded(records, manager) for name, (records, _meta) in per_source.items()
    }

    filtered_for_merge = {
        name: (records, per_source[name][1]) for name, records in filtered_source.items()
    }
    merged, _merge_collisions = vs.merge(filtered_for_merge)
    changed |= _emit(vocab_dir / "vocabulary.csv", vs.vocabulary_csv_text(merged), args.check)

    teater_records = filtered_source.get("teater", [])
    rescue_branches = _rescue_map(teater_records)

    targets: Dict[str, List[vs.VocabRecord]] = {}
    if "amcr" in filtered_source:
        targets["amcr"] = filtered_source["amcr"]
    if "teater" in filtered_source:
        targets["teater"] = teater_records
    if len(filtered_source) > 1:
        targets["union"] = filtered_source.get("amcr", []) + teater_records

    last_nested: Dict[str, Any] = {}
    for name, records in targets.items():
        nested, audit, collisions = _nest(
            manager, records, rescue_branches if name != "teater" else None, qualifiers
        )
        # Track 3 / O1 / F: link "X/Y" composites to a standalone X or Y also offered.
        # Scoring-time equivalence only (vocab_manager.attach_same_as docstring) — run
        # after nesting, since a pair can span two facets or two sources and is only
        # resolvable once dedup and placement have already happened.
        # `extra`/`suppress` are the reviewer's two corrections to what the label
        # shape alone can tell — both config, so a wrong link is a taxonomy_overrides
        # edit rather than a code change.
        same_as_extra, same_as_suppress = manager.same_as_overrides()
        same_as_links = attach_same_as(
            nested,
            separators=manager.composite_separators(),
            extra=same_as_extra,
            suppress=same_as_suppress,
        )
        audit = _with_same_as_counts(audit, nested)
        last_nested = nested
        meta = _base_meta(args.config, manager.overrides_path)
        meta["sources"] = [per_source[s][1] for s in per_source if s in ("amcr", "teater")]
        meta["counts"] = {
            "total": sum(len(v) for v in nested.values()),
            "by_theme": {k: len(v) for k, v in nested.items()},
            "collisions": len(collisions),
            "same_as_links": same_as_links,
        }
        # Theme order is priority-descending and load-bearing for prompt truncation, so
        # the nested file is NOT written with sort_keys.
        changed |= _emit(
            vocab_dir / f"{name}_nested.json", _json_text(nested, sort_keys=False), args.check
        )
        changed |= _emit(
            vocab_dir / f"{name}_nested.meta.json", _json_text(meta, sort_keys=True), args.check
        )
        changed |= _emit(vocab_dir / f"{name}_placement_audit.csv", _audit_text(audit), args.check)
        if args.stats:
            print(f"\n=== {name} ===")
            _stats(nested, audit)

    if args.update_legacy and last_nested:
        changed |= _emit(LEGACY_NESTED, _json_text(last_nested, sort_keys=False), args.check)

    if args.check:
        print("\n[check] drift detected." if changed else "\n[check] up to date.")
        return 1 if changed else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
