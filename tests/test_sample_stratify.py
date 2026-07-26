"""
tests/test_sample_stratify.py – Unit tests for sample_stratify.py, the
quality-stratified page sampler (hub issue #22): input loading, tier assignment,
deterministic sampling, split assignment, config parsing, and the in-process
main() CLI.
"""

import pandas as pd
import pytest

from sample_stratify import (
    TIER_CLEAN,
    TIER_DEGRADED,
    TIER_HARD,
    TIER_TEXT_POOR,
    _load_config,
    aggregate_line_csvs,
    assign_splits,
    assign_tiers,
    load_page_stats,
    main,
    stratified_sample,
)


def _page_stats_df():
    return pd.DataFrame(
        {
            "file": ["a", "a", "b", "c"],
            "page_num": [1, 2, 1, 1],
            "avg_quality_score": [0.9, 0.7, 0.4, float("nan")],
            "Clear": [8, 5, 1, 0],
            "Noisy": [1, 2, 1, 0],
            "Trash": [1, 1, 3, 0],
            "Non-text": [0, 0, 0, 0],
            "Empty": [0, 0, 0, 1],
        }
    )


# ── load_page_stats ─────────────────────────────────────────────────────────
def test_load_page_stats_fills_missing_categ_columns(tmp_path):
    path = tmp_path / "s.csv"
    pd.DataFrame({"file": ["a"], "page_num": [1], "avg_quality_score": [0.9]}).to_csv(path, index=False)
    df = load_page_stats(str(path))
    for col in ["Clear", "Noisy", "Trash", "Non-text", "Empty"]:
        assert col in df.columns


def test_load_page_stats_missing_required_raises(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"file": ["a"]}).to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_page_stats(str(path))


# ── aggregate_line_csvs ─────────────────────────────────────────────────────
def test_aggregate_line_csvs_scores_valid_lines_only(tmp_path):
    d = tmp_path / "lines"
    d.mkdir()
    pd.DataFrame(
        {
            "file": ["a", "a", "a"],
            "page_num": [1, 1, 1],
            "categ": ["Clear", "Noisy", "Trash"],
            "quality_score": [0.9, 0.6, 0.1],
        }
    ).to_csv(d / "a.csv", index=False)

    page_df = aggregate_line_csvs(str(d))
    row = page_df.iloc[0]
    assert row["num_lines"] == 3
    # average over VALID categories (Clear, Noisy) only → (0.9 + 0.6) / 2
    assert row["avg_quality_score"] == pytest.approx(0.75)


def test_aggregate_line_csvs_empty_dir_raises(tmp_path):
    d = tmp_path / "empty"
    d.mkdir()
    with pytest.raises(ValueError):
        aggregate_line_csvs(str(d))


# ── assign_tiers ────────────────────────────────────────────────────────────
def test_assign_tiers_classifies_each_tier():
    tiered = assign_tiers(_page_stats_df(), clean_min=0.85, degraded_min=0.60, valid_ratio_min=0.50)
    assert tiered.iloc[0]["tier"] == TIER_CLEAN  # 0.9, valid_ratio 0.9
    assert tiered.iloc[1]["tier"] == TIER_DEGRADED  # 0.7
    assert tiered.iloc[2]["tier"] == TIER_HARD  # 0.4
    assert tiered.iloc[3]["tier"] == TIER_TEXT_POOR  # NaN score


def test_assign_tiers_clean_requires_valid_ratio():
    df = pd.DataFrame(
        {
            "file": ["x"],
            "page_num": [1],
            "avg_quality_score": [0.95],
            "Clear": [1],
            "Noisy": [0],
            "Trash": [9],
            "Non-text": [0],
            "Empty": [0],
        }
    )
    tiered = assign_tiers(df, clean_min=0.85, degraded_min=0.60, valid_ratio_min=0.50)
    # high score but valid_ratio 0.1 < 0.5 → demoted from clean to degraded
    assert tiered.iloc[0]["tier"] == TIER_DEGRADED


# ── stratified_sample ───────────────────────────────────────────────────────
def test_stratified_sample_is_deterministic():
    tiered = assign_tiers(_page_stats_df())
    a = stratified_sample(tiered, n_pages=4, seed=42).reset_index(drop=True)
    b = stratified_sample(tiered, n_pages=4, seed=42).reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def test_stratified_sample_capped_by_pool_size():
    tiered = assign_tiers(_page_stats_df())
    out = stratified_sample(tiered, n_pages=100, seed=1)
    assert len(out) == 4  # one page per tier available


# ── assign_splits ───────────────────────────────────────────────────────────
def test_assign_splits_labels_all_rows():
    tiered = assign_tiers(_page_stats_df())
    sampled = stratified_sample(tiered, n_pages=4, seed=42)
    out = assign_splits(sampled, seed=42)
    assert set(out["split"].unique()) <= {"train", "dev", "test"}
    assert len(out) == len(sampled)


# ── _load_config ────────────────────────────────────────────────────────────
def test_load_config_none_returns_defaults():
    cfg = _load_config(None)
    assert cfg["n_pages"] == 200 and cfg["seed"] == 42


def test_load_config_reads_stratify_section(tmp_path):
    path = tmp_path / "c.txt"
    path.write_text("[STRATIFY]\nN_PAGES = 50\nCLEAN_MIN = 0.9\nSEED = 7\n", encoding="utf-8")
    cfg = _load_config(str(path))
    assert cfg["n_pages"] == 50
    assert cfg["clean_min"] == pytest.approx(0.9)
    assert cfg["seed"] == 7


# ── main (in-process CLI) ───────────────────────────────────────────────────
def test_main_end_to_end_writes_manifest(tmp_path):
    stats = tmp_path / "stats.csv"
    _page_stats_df().to_csv(stats, index=False)
    out = tmp_path / "manifest.csv"

    rc = main(["--page-stats", str(stats), "--n", "4", "--seed", "42", "--output", str(out)])

    assert rc == 0
    assert out.exists()
    manifest = pd.read_csv(out)
    assert "tier" in manifest.columns and "split" in manifest.columns
    assert len(manifest) >= 1
