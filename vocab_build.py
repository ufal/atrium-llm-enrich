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
from vocab_manager import VocabularyManager

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_VOCAB_DIR = REPO_ROOT / "data_samples" / "vocab"
DEFAULT_CONFIG = REPO_ROOT / "data_samples" / "taxonomy_config.json"
# The path every consumer already points at (llm_config.txt, service/api.py, the hub's
# e2e fixture). Kept stable so there is no flag day.
LEGACY_NESTED = REPO_ROOT / "data_samples" / "teater_nested_vocab.json"

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


def _base_meta(config_path: Path) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": "vocab_build.py",
        "tool_version": _tool_version(),
        "generated_utc": _now(),
        "taxonomy_config": {
            "path": str(config_path.relative_to(REPO_ROOT))
            if config_path.is_relative_to(REPO_ROOT)
            else str(config_path),
            "sha256": _sha256(config_path),
        },
    }


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
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
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
    pairs = vs.to_term_pairs(records, collisions=collisions)
    if collisions:
        print(f"  [warn] {len(collisions)} label collision(s) dropped, e.g. {collisions[:3]}")
    nested = manager.build_nested(pairs, audit=audit, rescue=rescue)
    return nested, audit


def _audit_text(rows: Sequence[Dict[str, Any]]) -> str:
    import io

    buf = io.StringIO(newline="")
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["cs", "en", "source", "source_id", "scheme", "theme", "placed_by"])
    for row in sorted(rows, key=lambda r: (r["source"], r["theme"], r["cs"])):
        writer.writerow(
            [row[c] for c in ("cs", "en", "source", "source_id", "scheme", "theme", "placed_by")]
        )
    return buf.getvalue()


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

    merged, collisions = vs.merge(per_source)
    changed |= _emit(vocab_dir / "vocabulary.csv", vs.vocabulary_csv_text(merged), args.check)

    # ── stage 2 artifacts ────────────────────────────────────────────────────
    manager = VocabularyManager(config_path=str(args.config))
    teater_records = per_source.get("teater", ([], {}))[0]
    rescue_branches = _rescue_map(teater_records)

    targets: Dict[str, List[vs.VocabRecord]] = {}
    if "amcr" in per_source:
        targets["amcr"] = per_source["amcr"][0]
    if "teater" in per_source:
        targets["teater"] = teater_records
    if len(per_source) > 1:
        targets["union"] = merged

    last_nested: Dict[str, Any] = {}
    for name, records in targets.items():
        nested, audit = _nest(manager, records, rescue_branches if name != "teater" else None)
        last_nested = nested
        meta = _base_meta(args.config)
        meta["sources"] = [per_source[s][1] for s in per_source if s in ("amcr", "teater")]
        meta["counts"] = {
            "total": sum(len(v) for v in nested.values()),
            "by_theme": {k: len(v) for k, v in nested.items()},
            "collisions": collisions if name == "union" else 0,
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
