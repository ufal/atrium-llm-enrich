"""
sample_stratify.py  –  Quality-stratified page sampling for the Document Understanding benchmark.

Consumes the per-page statistics produced by `atrium-alto-postprocess`
(`langID_aggregate_STAT.py` → `samples_page_stats.csv`, columns: file, page_num,
num_lines, Clear, Noisy, Trash, Non-text, Empty, total_word_count,
total_char_count, avg_quality_score, ...) or, alternatively, a directory of
per-document line-level category CSVs (DOC_LINE_CATEG), which it aggregates
on the fly.

Pages are bucketed into difficulty tiers and sampled deterministically into an
annotation manifest with an 80/10/10 train/dev/test split, mirroring the
`atrium-page-classification` dataset methodology (hub issue #22).

Tiers
-----
- ``clean``     : avg_quality_score >= CLEAN_MIN and valid-line ratio >= VALID_RATIO_MIN
- ``degraded``  : avg_quality_score >= DEGRADED_MIN (below ``clean``)
- ``hard``      : scored pages below DEGRADED_MIN
- ``text_poor`` : no Clear/Noisy lines at all (NaN score) — VLM-only candidates

Usage
-----
    python sample_stratify.py --config config_docu.txt
    python sample_stratify.py --page-stats samples_page_stats.csv --n 200 --output manifest.csv
    python sample_stratify.py --lines-dir data_samples/DOC_LINE_CATEG --n 40
"""

from __future__ import annotations

import argparse
import configparser
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ──────────────────────────────────────────────────────────────────────────────
# Constants & defaults
# ──────────────────────────────────────────────────────────────────────────────

TIER_COL = "tier"
SPLIT_COL = "split"

TIER_CLEAN = "clean"
TIER_DEGRADED = "degraded"
TIER_HARD = "hard"
TIER_TEXT_POOR = "text_poor"

TIERS_ORDERED = [TIER_CLEAN, TIER_DEGRADED, TIER_HARD, TIER_TEXT_POOR]

VALID_CATEGS = ["Clear", "Noisy"]
ALL_CATEGS = ["Clear", "Noisy", "Trash", "Non-text", "Empty"]

DEFAULTS: Dict[str, object] = {
    "clean_min": 0.85,
    "degraded_min": 0.60,
    "valid_ratio_min": 0.50,
    "n_pages": 200,
    "seed": 42,
    "split_ratios": (0.8, 0.1, 0.1),
    "output": "docu_sample_manifest.csv",
}


# ──────────────────────────────────────────────────────────────────────────────
# Input loading
# ──────────────────────────────────────────────────────────────────────────────

def load_page_stats(path: str) -> pd.DataFrame:
    """Load an aggregated per-page stats CSV (output of langID_aggregate_STAT.py)."""
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    required = ["file", "page_num", "avg_quality_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Page-stats CSV {path} is missing required columns: {missing}")
    for col in ALL_CATEGS:
        if col not in df.columns:
            df[col] = 0
    return df


def aggregate_line_csvs(lines_dir: str) -> pd.DataFrame:
    """Aggregate per-document line-level category CSVs (DOC_LINE_CATEG) into page stats."""
    frames: List[pd.DataFrame] = []
    for name in sorted(os.listdir(lines_dir)):
        if not name.endswith(".csv"):
            continue
        df = pd.read_csv(os.path.join(lines_dir, name))
        df.columns = df.columns.str.strip()
        if not {"file", "page_num", "categ", "quality_score"}.issubset(df.columns):
            continue
        frames.append(df[["file", "page_num", "categ", "quality_score"]])
    if not frames:
        raise ValueError(f"No usable line-level CSVs found in {lines_dir}")
    lines = pd.concat(frames, ignore_index=True)

    counts = lines.groupby(["file", "page_num", "categ"]).size().unstack(fill_value=0).reset_index()
    for col in ALL_CATEGS:
        if col not in counts.columns:
            counts[col] = 0

    valid = lines[lines["categ"].isin(VALID_CATEGS)]
    scores = (
        valid.groupby(["file", "page_num"])
        .agg(avg_quality_score=("quality_score", "mean"))
        .reset_index()
    )
    page_df = pd.merge(counts, scores, on=["file", "page_num"], how="left")
    page_df["num_lines"] = page_df[ALL_CATEGS].sum(axis=1)
    return page_df


# ──────────────────────────────────────────────────────────────────────────────
# Tiering & sampling
# ──────────────────────────────────────────────────────────────────────────────

def assign_tiers(
    page_df: pd.DataFrame,
    clean_min: float = float(DEFAULTS["clean_min"]),
    degraded_min: float = float(DEFAULTS["degraded_min"]),
    valid_ratio_min: float = float(DEFAULTS["valid_ratio_min"]),
) -> pd.DataFrame:
    """Assign a difficulty tier to every page based on its aggregated quality signals."""
    df = page_df.copy()
    num_lines = df[ALL_CATEGS].sum(axis=1).replace(0, np.nan)
    valid_ratio = (df["Clear"] + df["Noisy"]) / num_lines

    score = df["avg_quality_score"]
    tier = pd.Series(TIER_HARD, index=df.index, dtype=object)
    tier[score >= degraded_min] = TIER_DEGRADED
    tier[(score >= clean_min) & (valid_ratio >= valid_ratio_min)] = TIER_CLEAN
    tier[score.isna()] = TIER_TEXT_POOR

    df[TIER_COL] = tier
    df["valid_line_ratio"] = valid_ratio.fillna(0.0).round(4)
    return df


def stratified_sample(
    tiered_df: pd.DataFrame,
    n_pages: int = int(DEFAULTS["n_pages"]),
    seed: int = int(DEFAULTS["seed"]),
) -> pd.DataFrame:
    """Sample ~n_pages pages with equal per-tier allocation (falling back to tier size)."""
    rng = np.random.default_rng(seed)
    present = [t for t in TIERS_ORDERED if (tiered_df[TIER_COL] == t).any()]
    per_tier = max(1, n_pages // max(1, len(present)))

    picked: List[pd.DataFrame] = []
    for t in present:
        pool = tiered_df[tiered_df[TIER_COL] == t]
        k = min(per_tier, len(pool))
        idx = rng.choice(pool.index.to_numpy(), size=k, replace=False)
        picked.append(pool.loc[idx])
    return pd.concat(picked, ignore_index=True)


def assign_splits(
    sampled_df: pd.DataFrame,
    ratios=DEFAULTS["split_ratios"],
    seed: int = int(DEFAULTS["seed"]),
) -> pd.DataFrame:
    """Assign train/dev/test splits per tier so every tier is represented in every split."""
    train_r, dev_r, _ = ratios
    rng = np.random.default_rng(seed + 1)
    df = sampled_df.copy()
    df[SPLIT_COL] = "train"
    for t in df[TIER_COL].unique():
        idx = df.index[df[TIER_COL] == t].to_numpy().copy()
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(round(n * train_r))
        n_dev = int(round(n * dev_r))
        df.loc[idx[n_train:n_train + n_dev], SPLIT_COL] = "dev"
        df.loc[idx[n_train + n_dev:], SPLIT_COL] = "test"
    return df


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def _load_config(path: Optional[str]) -> Dict[str, object]:
    out = dict(DEFAULTS)
    if not path:
        return out
    cfg = configparser.ConfigParser()
    cfg.read(path, encoding="utf-8")
    if cfg.has_section("STRATIFY"):
        s = cfg["STRATIFY"]
        out["page_stats"] = s.get("PAGE_STATS_CSV", fallback=None)
        out["lines_dir"] = s.get("LINES_DIR", fallback=None)
        out["clean_min"] = s.getfloat("CLEAN_MIN", fallback=float(DEFAULTS["clean_min"]))
        out["degraded_min"] = s.getfloat("DEGRADED_MIN", fallback=float(DEFAULTS["degraded_min"]))
        out["valid_ratio_min"] = s.getfloat("VALID_RATIO_MIN", fallback=float(DEFAULTS["valid_ratio_min"]))
        out["n_pages"] = s.getint("N_PAGES", fallback=int(DEFAULTS["n_pages"]))
        out["seed"] = s.getint("SEED", fallback=int(DEFAULTS["seed"]))
        out["output"] = s.get("OUTPUT_MANIFEST", fallback=str(DEFAULTS["output"]))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Quality-stratified page sampler (hub issue #22).")
    parser.add_argument("--config", default=None, help="INI config with a [STRATIFY] section")
    parser.add_argument("--page-stats", default=None, help="Aggregated per-page stats CSV")
    parser.add_argument("--lines-dir", default=None, help="Directory of DOC_LINE_CATEG CSVs")
    parser.add_argument("--n", type=int, default=None, help="Target number of sampled pages")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--output", default=None, help="Output manifest CSV path")
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    page_stats = args.page_stats or cfg.get("page_stats")
    lines_dir = args.lines_dir or cfg.get("lines_dir")
    n_pages = args.n if args.n is not None else int(cfg["n_pages"])
    seed = args.seed if args.seed is not None else int(cfg["seed"])
    output = args.output or str(cfg["output"])

    if page_stats:
        page_df = load_page_stats(page_stats)
    elif lines_dir:
        page_df = aggregate_line_csvs(lines_dir)
    else:
        parser.error("Provide --page-stats or --lines-dir (or set them in the config).")
        return 2

    tiered = assign_tiers(
        page_df,
        clean_min=float(cfg["clean_min"]),
        degraded_min=float(cfg["degraded_min"]),
        valid_ratio_min=float(cfg["valid_ratio_min"]),
    )
    sampled = stratified_sample(tiered, n_pages=n_pages, seed=seed)
    manifest = assign_splits(sampled, seed=seed)

    cols = ["file", "page_num", TIER_COL, SPLIT_COL, "avg_quality_score", "valid_line_ratio"] + ALL_CATEGS
    manifest = manifest[[c for c in cols if c in manifest.columns]]
    manifest.to_csv(output, index=False, encoding="utf-8")

    summary = manifest.groupby([TIER_COL, SPLIT_COL]).size().unstack(fill_value=0)
    print(f"Wrote {len(manifest)} pages to {output}")
    print(summary.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
