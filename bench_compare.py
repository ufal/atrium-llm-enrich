"""
bench_compare.py  –  Document Understanding benchmark comparison runner.

Connects the DU harness pieces decided in hub issue #22: takes the manifest
produced by `sample_stratify.py`, a gold-transcription directory, and one or
more named prediction directories (e.g. the `PAGE_TXT*` outputs of
`atrium-alto-postprocess`, or a VLM's transcriptions), scores every page with
`eval_metrics.py`, and writes per-page scores, per-tier/per-model aggregates,
and a Markdown comparison report (mirroring the model-comparison tables of
`atrium-page-classification`).

File conventions
----------------
- Predictions: ``<root>/<doc>/<doc>-<page>.txt`` (the `atrium-alto-postprocess`
  ``PAGE_TXT*`` layout); a flat ``<root>/<doc>-<page>.txt`` fallback is accepted.
- Gold transcriptions mirror the prediction layout: ``gold/<doc>/<doc>-<page>.txt``
  (UTF-8 plain text in reading order; whitespace is normalized before scoring).
- Optional entity sidecars (scored with ``--entities``):
  ``<root>/<doc>/<doc>-<page>.entities.tsv`` — one entity per line as
  ``TYPE<TAB>surface text`` (CNEC 2.0 / TEATER type codes); ``#`` comments and
  blank lines are ignored. A page joins entity aggregates only when the *gold*
  sidecar exists; a missing hypothesis sidecar counts as zero predicted entities.
- TEDS (tables) is deferred until `table_teds.py` is vendored (hub issue #22,
  Milestone 2).

Usage
-----
    python bench_compare.py --config config_docu.txt
    python bench_compare.py --manifest docu_sample_manifest.csv \\
        --gold data/gold \\
        --pred alto=../atrium-alto-postprocess/data_samples/PAGE_TXT \\
               layoutreader=../atrium-alto-postprocess/data_samples/PAGE_TXT_LR \\
        --split test --output-dir bench_results
"""

from __future__ import annotations

import argparse
import configparser
import csv
import os
import sys
from typing import Dict, Iterable, List, Optional, Tuple

from atrium_paradata import ParadataLogger
from eval_metrics import entity_prf, score_corpus, score_page
from sample_stratify import SPLIT_COL, TIER_COL, TIERS_ORDERED

# ──────────────────────────────────────────────────────────────────────────────
# Constants & defaults
# ──────────────────────────────────────────────────────────────────────────────

SPLITS = ("train", "dev", "test")
MISSING_MODES = ("skip", "error")

# score_corpus() injects this aggregate bucket — a manifest tier must never use it.
OVERALL_KEY = "overall"

TEXT_SUFFIX = ".txt"
ENTITY_SUFFIX = ".entities.tsv"

PAGE_CSV = "page_scores.csv"
AGG_CSV = "aggregate_scores.csv"
REPORT_MD = "report.md"

ENTITY_FIELDS = (
    "entity_precision",
    "entity_recall",
    "entity_f1",
    "entity_tp",
    "entity_n_ref",
    "entity_n_hyp",
)

PAGE_FIELDS = [
    "model",
    "file",
    "page_num",
    TIER_COL,
    SPLIT_COL,
    "cer",
    "wer",
    "ned",
    "ref_chars",
    "hyp_chars",
    *ENTITY_FIELDS,
]

AGG_FIELDS = [
    "model",
    TIER_COL,
    "n_pages",
    "n_missing",
    "cer",
    "wer",
    "ned",
    "entity_precision",
    "entity_recall",
    "entity_f1",
    "entity_n_ref",
    "entity_n_hyp",
    "n_entity_pages",
]

DEFAULTS: Dict[str, object] = {
    "manifest": None,
    "gold_dir": None,
    "pred_dirs": None,
    "split": None,
    "entities": False,
    "missing": "skip",
    "output_dir": "bench_results",
    "paradata_dir": "paradata",
}


# ──────────────────────────────────────────────────────────────────────────────
# Input loading
# ──────────────────────────────────────────────────────────────────────────────

def parse_pred_specs(specs: Iterable[str]) -> Dict[str, str]:
    """Parse ``NAME=PATH`` prediction specs into an order-preserving dict."""
    out: Dict[str, str] = {}
    for spec in specs:
        name, sep, path = spec.partition("=")
        name, path = name.strip(), path.strip()
        if not sep or not name or not path:
            raise ValueError(f"Prediction spec {spec!r} must look like NAME=PATH")
        if name in out:
            raise ValueError(f"Duplicate model name {name!r} in prediction specs")
        out[name] = path
    return out


def load_manifest(path: str, split: Optional[str] = None) -> List[Dict[str, object]]:
    """Load a sample_stratify.py manifest CSV, optionally filtered to one split.

    Rows are returned sorted by (file, page_num) so downstream outputs are
    deterministic regardless of manifest ordering.
    """
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        fields = [c.strip() for c in reader.fieldnames or []]
        required = ["file", "page_num", TIER_COL]
        if split is not None:
            required.append(SPLIT_COL)
        missing = [c for c in required if c not in fields]
        if missing:
            raise ValueError(f"Manifest {path} is missing required columns: {missing}")

        rows: List[Dict[str, object]] = []
        for raw in reader:
            row = {(k or "").strip(): (v or "").strip() for k, v in raw.items()}
            tier = row[TIER_COL]
            if tier == OVERALL_KEY:
                raise ValueError(
                    f"Manifest {path} uses reserved tier name {OVERALL_KEY!r}"
                )
            if split is not None and row.get(SPLIT_COL) != split:
                continue
            rows.append(
                {
                    "file": row["file"],
                    # tolerate float-formatted page numbers ("1.0") from pandas
                    "page_num": int(float(row["page_num"])),
                    TIER_COL: tier,
                    SPLIT_COL: row.get(SPLIT_COL, ""),
                }
            )
    rows.sort(key=lambda r: (r["file"], r["page_num"]))
    return rows


def page_relpath(doc: str, page_num: int, suffix: str = TEXT_SUFFIX) -> str:
    """Canonical page path relative to a gold/prediction root."""
    return os.path.join(doc, f"{doc}-{page_num}{suffix}")


def resolve_page_file(
    root: str, doc: str, page_num: int, suffix: str = TEXT_SUFFIX
) -> Optional[str]:
    """Find a page file under `root`: nested ``<doc>/<doc>-<page>`` first, then flat."""
    nested = os.path.join(root, doc, f"{doc}-{page_num}{suffix}")
    if os.path.isfile(nested):
        return nested
    flat = os.path.join(root, f"{doc}-{page_num}{suffix}")
    if os.path.isfile(flat):
        return flat
    return None


def read_page_text(path: str) -> str:
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def read_entities(path: str) -> List[Tuple[str, str]]:
    """Read a ``TYPE<TAB>surface text`` sidecar into (type, text) tuples."""
    entities: List[Tuple[str, str]] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            etype, sep, text = line.rstrip("\n").partition("\t")
            if not sep or not etype.strip() or not text.strip():
                raise ValueError(
                    f"{path}:{lineno}: expected 'TYPE<TAB>surface text', got {stripped!r}"
                )
            entities.append((etype.strip(), text.strip()))
    return entities


# ──────────────────────────────────────────────────────────────────────────────
# Scoring
# ──────────────────────────────────────────────────────────────────────────────

def score_pages(
    rows: List[Dict[str, object]],
    gold_dir: str,
    pred_dirs: Dict[str, str],
    entities: bool = False,
    missing: str = "skip",
    logger: Optional[ParadataLogger] = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    """Score every manifest page against every model's prediction directory.

    Returns long-format page rows (one per model × page found) plus counters:
    ``missing_gold``, per-model ``missing_pred`` and ``pages_scored``.
    """
    page_rows: List[Dict[str, object]] = []
    stats: Dict[str, object] = {
        "missing_gold": 0,
        "missing_pred": {model: 0 for model in pred_dirs},
        "pages_scored": 0,
    }

    for row in rows:
        doc, page_num = str(row["file"]), int(row["page_num"])
        rel = page_relpath(doc, page_num)

        gold_path = resolve_page_file(gold_dir, doc, page_num)
        if gold_path is None:
            if missing == "error":
                raise ValueError(f"Gold transcription missing: {rel} (under {gold_dir})")
            stats["missing_gold"] += 1
            if logger:
                logger.log_skip(rel, "gold_missing")
            continue
        ref_text = read_page_text(gold_path)

        ref_entities: Optional[List[Tuple[str, str]]] = None
        if entities:
            gold_ent = resolve_page_file(gold_dir, doc, page_num, suffix=ENTITY_SUFFIX)
            if gold_ent is not None:
                ref_entities = read_entities(gold_ent)

        stats["pages_scored"] += 1

        for model, pred_root in pred_dirs.items():
            pred_path = resolve_page_file(pred_root, doc, page_num)
            if pred_path is None:
                if missing == "error":
                    raise ValueError(
                        f"Prediction missing for model {model!r}: {rel} (under {pred_root})"
                    )
                stats["missing_pred"][model] += 1
                if logger:
                    logger.log_skip(f"{model}:{rel}", "prediction_missing")
                continue

            out_row: Dict[str, object] = {
                "model": model,
                "file": doc,
                "page_num": page_num,
                TIER_COL: row[TIER_COL],
                SPLIT_COL: row[SPLIT_COL],
                **score_page(ref_text, read_page_text(pred_path)),
            }
            out_row.update(dict.fromkeys(ENTITY_FIELDS))
            if ref_entities is not None:
                hyp_ent = resolve_page_file(pred_root, doc, page_num, suffix=ENTITY_SUFFIX)
                hyp_entities = read_entities(hyp_ent) if hyp_ent is not None else []
                prf = entity_prf(ref_entities, hyp_entities)
                out_row.update(
                    {
                        "entity_precision": prf["precision"],
                        "entity_recall": prf["recall"],
                        "entity_f1": prf["f1"],
                        "entity_tp": prf["tp"],
                        "entity_n_ref": prf["n_ref"],
                        "entity_n_hyp": prf["n_hyp"],
                    }
                )
            page_rows.append(out_row)
    return page_rows, stats


def _tier_order(tiers_present: Iterable[str]) -> List[str]:
    present = set(tiers_present)
    known = [t for t in TIERS_ORDERED if t in present]
    unknown = sorted(present - set(TIERS_ORDERED))
    return known + unknown


def aggregate(
    page_rows: List[Dict[str, object]],
    model_order: List[str],
    stats: Dict[str, object],
) -> List[Dict[str, object]]:
    """Per-model × per-tier aggregate rows: macro CER/WER/NED + micro entity P/R/F1."""
    agg_rows: List[Dict[str, object]] = []
    for model in model_order:
        rows = [r for r in page_rows if r["model"] == model]

        corpus: Dict[str, Dict[str, float]] = {}
        if rows:
            pages = [{k: r[k] for k in ("cer", "wer", "ned")} for r in rows]
            corpus = score_corpus(pages, [str(r[TIER_COL]) for r in rows])

        ent_pool: Dict[str, Dict[str, int]] = {}
        for r in rows:
            if r["entity_tp"] is None:
                continue
            for key in (str(r[TIER_COL]), OVERALL_KEY):
                pool = ent_pool.setdefault(
                    key, {"tp": 0, "n_ref": 0, "n_hyp": 0, "n_entity_pages": 0}
                )
                pool["tp"] += int(r["entity_tp"])
                pool["n_ref"] += int(r["entity_n_ref"])
                pool["n_hyp"] += int(r["entity_n_hyp"])
                pool["n_entity_pages"] += 1

        tiers = _tier_order(str(r[TIER_COL]) for r in rows) + [OVERALL_KEY]
        for tier in tiers:
            c = corpus.get(tier, {})
            agg_row: Dict[str, object] = {
                "model": model,
                TIER_COL: tier,
                "n_pages": int(c.get("n_pages", 0)),
                "n_missing": stats["missing_pred"][model] if tier == OVERALL_KEY else None,
                "cer": c.get("cer"),
                "wer": c.get("wer"),
                "ned": c.get("ned"),
            }
            pool = ent_pool.get(tier)
            if pool is not None:
                tp, n_ref, n_hyp = pool["tp"], pool["n_ref"], pool["n_hyp"]
                precision = tp / n_hyp if n_hyp else 0.0
                recall = tp / n_ref if n_ref else 0.0
                f1 = (
                    2 * precision * recall / (precision + recall)
                    if (precision + recall)
                    else 0.0
                )
                agg_row.update(
                    {
                        "entity_precision": precision,
                        "entity_recall": recall,
                        "entity_f1": f1,
                        "entity_n_ref": n_ref,
                        "entity_n_hyp": n_hyp,
                        "n_entity_pages": pool["n_entity_pages"],
                    }
                )
            else:
                agg_row.update(
                    dict.fromkeys(
                        (
                            "entity_precision",
                            "entity_recall",
                            "entity_f1",
                            "entity_n_ref",
                            "entity_n_hyp",
                            "n_entity_pages",
                        )
                    )
                )
            agg_rows.append(agg_row)
    return agg_rows


# ──────────────────────────────────────────────────────────────────────────────
# Reporting
# ──────────────────────────────────────────────────────────────────────────────

def _fmt_pct(value: Optional[float]) -> str:
    return f"{value * 100:.2f}" if value is not None else "–"


def _fmt_ned(value: Optional[float]) -> str:
    return f"{value:.3f}" if value is not None else "–"


def _fmt_f1(value: Optional[float]) -> str:
    return f"{value:.2f}" if value is not None else "–"


def render_markdown(
    agg_rows: List[Dict[str, object]],
    model_order: List[str],
    meta: Dict[str, object],
) -> str:
    """Deterministic Markdown report (no timestamps — paradata carries timing)."""
    by_model_tier = {(r["model"], r[TIER_COL]): r for r in agg_rows}
    overall = {m: by_model_tier.get((m, OVERALL_KEY), {}) for m in model_order}

    def best(metric: str, maximise: bool = False) -> Optional[float]:
        values = [overall[m].get(metric) for m in model_order]
        values = [v for v in values if v is not None]
        if not values:
            return None
        return max(values) if maximise else min(values)

    show_entities = bool(meta.get("entities")) and any(
        overall[m].get("entity_f1") is not None for m in model_order
    )
    best_cer, best_wer, best_ned = best("cer"), best("wer"), best("ned")
    best_f1 = best("entity_f1", maximise=True) if show_entities else None

    lines: List[str] = []
    lines.append("# Document Understanding benchmark — model comparison")
    lines.append("")
    lines.append(
        f"Manifest: `{meta['manifest']}` · Split: `{meta['split']}` · "
        f"Models: {len(model_order)} · Metrics: `eval_metrics.py` (hub issue #22)"
    )
    lines.append("")
    lines.append("## Overall")
    lines.append("")
    header = "| Model | Pages | CER (%) | WER (%) | NED |"
    rule = "|-------|-------|---------|---------|-----|"
    if show_entities:
        header += " Entity F1 |"
        rule += "-----------|"
    lines.append(header)
    lines.append(rule)
    for model in model_order:
        row = overall[model]
        name = f"**{model}**" if row.get("cer") is not None and row["cer"] == best_cer else model

        def cell(value: Optional[float], best_value: Optional[float], fmt) -> str:
            text = fmt(value)
            if value is not None and best_value is not None and value == best_value:
                return f"**{text}**"
            return text

        cells = [
            name,
            str(row.get("n_pages", 0)),
            cell(row.get("cer"), best_cer, _fmt_pct),
            cell(row.get("wer"), best_wer, _fmt_pct),
            cell(row.get("ned"), best_ned, _fmt_ned),
        ]
        if show_entities:
            cells.append(cell(row.get("entity_f1"), best_f1, _fmt_f1))
        lines.append("| " + " | ".join(cells) + " |")

    tiers = _tier_order(
        str(r[TIER_COL]) for r in agg_rows if r[TIER_COL] != OVERALL_KEY
    )
    for title, metric, fmt in (
        ("CER (%)", "cer", _fmt_pct),
        ("WER (%)", "wer", _fmt_pct),
        ("NED", "ned", _fmt_ned),
    ):
        lines.append("")
        lines.append(f"## {title} by tier")
        lines.append("")
        cols = tiers + [OVERALL_KEY]
        lines.append("| Model | " + " | ".join(cols) + " |")
        lines.append("|-------|" + "|".join("-" * (len(c) + 2) for c in cols) + "|")
        for model in model_order:
            cells = [fmt(by_model_tier.get((model, t), {}).get(metric)) for t in cols]
            lines.append(f"| {model} | " + " | ".join(cells) + " |")

    notes: List[str] = []
    missing_gold = int(meta.get("missing_gold", 0))
    if missing_gold:
        notes.append(
            f"> {missing_gold} page(s) missing gold transcription — excluded for all models."
        )
    for model in model_order:
        n = int(meta.get("missing_pred", {}).get(model, 0))
        if n:
            notes.append(f"> {model}: {n} page(s) missing prediction.")
    if notes:
        lines.append("")
        lines.extend(notes)
    lines.append("")
    return "\n".join(lines)


def write_csv(path: str, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    def prepare(value: object) -> object:
        return round(value, 6) if isinstance(value, float) else value

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: prepare(row.get(k)) for k in fieldnames})


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _load_config(path: Optional[str]) -> Dict[str, object]:
    out = dict(DEFAULTS)
    if not path:
        return out
    cfg = configparser.ConfigParser()
    cfg.read(path, encoding="utf-8")
    if cfg.has_section("BENCHMARK"):
        s = cfg["BENCHMARK"]
        out["manifest"] = s.get("MANIFEST", fallback=None)
        out["gold_dir"] = s.get("GOLD_DIR", fallback=None)
        out["pred_dirs"] = s.get("PRED_DIRS", fallback=None)
        out["split"] = s.get("SPLIT", fallback=None)
        out["entities"] = s.getboolean("ENTITIES", fallback=bool(DEFAULTS["entities"]))
        out["missing"] = s.get("MISSING", fallback=str(DEFAULTS["missing"]))
        out["output_dir"] = s.get("OUTPUT_DIR", fallback=str(DEFAULTS["output_dir"]))
        out["paradata_dir"] = s.get("PARADATA_DIR", fallback=str(DEFAULTS["paradata_dir"]))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Document Understanding benchmark comparison runner (hub issue #22)."
    )
    parser.add_argument("--config", default=None, help="INI config with a [BENCHMARK] section")
    parser.add_argument("--manifest", default=None, help="Manifest CSV from sample_stratify.py")
    parser.add_argument(
        "--gold", default=None, help="Gold transcription root (gold/<doc>/<doc>-<page>.txt)"
    )
    parser.add_argument(
        "--pred",
        nargs="+",
        default=None,
        metavar="NAME=PATH",
        help="Named prediction directories, e.g. alto=.../PAGE_TXT glm=.../PAGE_TXT_LLM",
    )
    parser.add_argument(
        "--split", default=None, choices=list(SPLITS), help="Restrict to one manifest split"
    )
    parser.add_argument(
        "--entities",
        action="store_true",
        help="Also score entity F1 from .entities.tsv sidecars",
    )
    parser.add_argument(
        "--missing",
        default=None,
        choices=list(MISSING_MODES),
        help="Missing gold/prediction files: skip (default) or error",
    )
    parser.add_argument("--output-dir", default=None, help="Directory for the result files")
    parser.add_argument("--paradata-dir", default=None, help="Directory for paradata JSON logs")
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    manifest = args.manifest or cfg.get("manifest")
    gold_dir = args.gold or cfg.get("gold_dir")
    if args.pred:
        pred_arg: Optional[List[str]] = args.pred
    elif cfg.get("pred_dirs"):
        pred_arg = [p.strip() for p in str(cfg["pred_dirs"]).split(",") if p.strip()]
    else:
        pred_arg = None
    split = args.split or cfg.get("split") or None
    entities = args.entities or bool(cfg.get("entities"))
    missing = args.missing or str(cfg.get("missing") or DEFAULTS["missing"])
    output_dir = args.output_dir or str(cfg.get("output_dir"))
    paradata_dir = args.paradata_dir or str(cfg.get("paradata_dir"))

    if not manifest or not gold_dir or not pred_arg:
        parser.error("Provide --manifest, --gold and --pred NAME=PATH (or set them in the config).")
        return 2
    if split is not None and split not in SPLITS:
        parser.error(f"SPLIT must be one of {SPLITS}, got {split!r}")
        return 2
    if missing not in MISSING_MODES:
        parser.error(f"MISSING must be one of {MISSING_MODES}, got {missing!r}")
        return 2
    try:
        pred_dirs = parse_pred_specs(pred_arg)
    except ValueError as exc:
        parser.error(str(exc))
        return 2

    rows = load_manifest(str(manifest), split=split)
    model_order = list(pred_dirs)

    logger = ParadataLogger(
        program="llm-enrich",
        config={
            "script": "bench_compare.py",
            "manifest": str(manifest),
            "gold_dir": str(gold_dir),
            "pred_dirs": dict(pred_dirs),
            "models": model_order,
            "split": split or "all",
            "entities": entities,
            "missing": missing,
            "output_dir": output_dir,
        },
        paradata_dir=paradata_dir,
        output_types=["csv", "md"],
    )

    with logger:
        page_rows, stats = score_pages(
            rows, str(gold_dir), pred_dirs, entities=entities, missing=missing, logger=logger
        )
        model_index = {m: i for i, m in enumerate(model_order)}
        page_rows.sort(key=lambda r: (model_index[str(r["model"])], r["file"], r["page_num"]))
        agg_rows = aggregate(page_rows, model_order, stats)

        os.makedirs(output_dir, exist_ok=True)
        write_csv(os.path.join(output_dir, PAGE_CSV), page_rows, PAGE_FIELDS)
        write_csv(os.path.join(output_dir, AGG_CSV), agg_rows, AGG_FIELDS)

        report = render_markdown(
            agg_rows,
            model_order,
            {
                "manifest": os.path.basename(str(manifest)),
                "split": split or "all",
                "entities": entities,
                "missing_gold": stats["missing_gold"],
                "missing_pred": stats["missing_pred"],
            },
        )
        with open(os.path.join(output_dir, REPORT_MD), "w", encoding="utf-8") as fh:
            fh.write(report)

        logger.log_success("csv", count=2)
        logger.log_success("md", count=1)
        logger.finalize(input_total=len(rows), processed_total=int(stats["pages_scored"]))

    print(
        f"Scored {stats['pages_scored']} gold page(s) × {len(model_order)} model(s) "
        f"→ {len(page_rows)} page-score rows"
    )
    for row in agg_rows:
        if row[TIER_COL] != OVERALL_KEY:
            continue
        print(
            f"  {row['model']}: CER {_fmt_pct(row['cer'])}% · WER {_fmt_pct(row['wer'])}% · "
            f"NED {_fmt_ned(row['ned'])} over {row['n_pages']} page(s), "
            f"{row['n_missing']} missing"
        )
    print(f"Wrote {PAGE_CSV}, {AGG_CSV}, {REPORT_MD} to {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
