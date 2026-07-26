"""
tests/test_bench_compare.py – Unit tests for bench_compare.py, the Document
Understanding benchmark comparison runner (hub issue #22): prediction-spec
parsing, manifest loading, page-file resolution, per-page scoring against
gold, missing-file handling, entity sidecar scoring, per-tier aggregation,
Markdown report rendering, and the in-process main() CLI.
"""

import csv
import glob
import json

import pytest

from bench_compare import (
    AGG_CSV,
    PAGE_CSV,
    REPORT_MD,
    _load_config,
    aggregate,
    load_manifest,
    main,
    parse_pred_specs,
    render_markdown,
    resolve_page_file,
    score_pages,
)

# gold docA-1 "abcd efgh" (9 chars, 2 words) vs noisy "abxd efgh": 1 char edit
NOISY_CER = 1 / 9
NOISY_WER = 1 / 2


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_bench(tmp_path, entities=False):
    """Manifest + gold + two models (perfect / noisy) with known-by-hand scores.

    - docC-1 is in the manifest but has no gold file (missing-gold case).
    - noisy/ lacks docB-2.txt (missing-prediction case).
    - entity sidecars (when requested): gold docA-1 = {PER Jan, LOC Praha};
      noisy hyp = {PER Jan, LOC Brno} → P = R = F1 = 0.5.
    """
    gold = tmp_path / "gold"
    perfect = tmp_path / "perfect"
    noisy = tmp_path / "noisy"

    texts = {
        ("docA", 1): "abcd efgh",
        ("docA", 2): "ijkl mnop",
        ("docB", 1): "qrst uvwx",
        ("docB", 2): "yz12 3456",
    }
    for (doc, page), text in texts.items():
        _write(gold / doc / f"{doc}-{page}.txt", text)
        _write(perfect / doc / f"{doc}-{page}.txt", text)
        if (doc, page) != ("docB", 2):
            _write(noisy / doc / f"{doc}-{page}.txt", text)
    _write(noisy / "docA" / "docA-1.txt", "abxd efgh")

    if entities:
        _write(gold / "docA" / "docA-1.entities.tsv", "PER\tJan\nLOC\tPraha\n")
        _write(perfect / "docA" / "docA-1.entities.tsv", "# comment\nPER\tJan\nLOC\tPraha\n")
        _write(noisy / "docA" / "docA-1.entities.tsv", "PER\tJan\nLOC\tBrno\n")

    manifest = tmp_path / "manifest.csv"
    _write(
        manifest,
        "file,page_num,tier,split\n"
        "docA,1,clean,test\n"
        "docA,2,degraded,test\n"
        "docB,1,clean,dev\n"
        "docB,2,hard,test\n"
        "docC,1,hard,test\n",
    )
    pred_dirs = {"perfect": str(perfect), "noisy": str(noisy)}
    return manifest, gold, pred_dirs


# ── parse_pred_specs ────────────────────────────────────────────────────────
def test_parse_pred_specs_preserves_order():
    out = parse_pred_specs(["b=/tmp/b", "a=/tmp/a"])
    assert list(out) == ["b", "a"]
    assert out["a"] == "/tmp/a"


def test_parse_pred_specs_rejects_bad_and_duplicate():
    with pytest.raises(ValueError):
        parse_pred_specs(["noequals"])
    with pytest.raises(ValueError):
        parse_pred_specs(["a=/x", "a=/y"])


# ── load_manifest ───────────────────────────────────────────────────────────
def test_load_manifest_split_filter(tmp_path):
    manifest, _, _ = _make_bench(tmp_path)
    all_rows = load_manifest(str(manifest))
    test_rows = load_manifest(str(manifest), split="test")
    assert len(all_rows) == 5
    assert len(test_rows) == 4
    assert all(r["split"] == "test" for r in test_rows)


def test_load_manifest_missing_column_raises(tmp_path):
    path = tmp_path / "bad.csv"
    _write(path, "file,page_num\ndocA,1\n")
    with pytest.raises(ValueError):
        load_manifest(str(path))


def test_load_manifest_parses_float_page_numbers(tmp_path):
    path = tmp_path / "m.csv"
    _write(path, "file,page_num,tier\ndocA,1.0,clean\ndocA,2,clean\n")
    rows = load_manifest(str(path))
    assert [r["page_num"] for r in rows] == [1, 2]


def test_load_manifest_rejects_reserved_tier(tmp_path):
    path = tmp_path / "m.csv"
    _write(path, "file,page_num,tier\ndocA,1,overall\n")
    with pytest.raises(ValueError):
        load_manifest(str(path))


# ── resolve_page_file ───────────────────────────────────────────────────────
def test_resolve_page_file_nested_flat_none(tmp_path):
    _write(tmp_path / "docA" / "docA-1.txt", "x")
    _write(tmp_path / "docB-1.txt", "y")
    assert resolve_page_file(str(tmp_path), "docA", 1).endswith("docA-1.txt")
    assert resolve_page_file(str(tmp_path), "docB", 1).endswith("docB-1.txt")
    assert resolve_page_file(str(tmp_path), "docC", 1) is None


# ── score_pages ─────────────────────────────────────────────────────────────
def test_score_pages_perfect_model_is_zero(tmp_path):
    manifest, gold, pred_dirs = _make_bench(tmp_path)
    rows = load_manifest(str(manifest))
    page_rows, _ = score_pages(rows, str(gold), {"perfect": pred_dirs["perfect"]})
    assert len(page_rows) == 4
    assert all(r["cer"] == r["wer"] == r["ned"] == 0.0 for r in page_rows)


def test_score_pages_noisy_model_expected_scores(tmp_path):
    manifest, gold, pred_dirs = _make_bench(tmp_path)
    rows = load_manifest(str(manifest))
    page_rows, _ = score_pages(rows, str(gold), {"noisy": pred_dirs["noisy"]})
    edited = next(r for r in page_rows if r["file"] == "docA" and r["page_num"] == 1)
    assert edited["cer"] == pytest.approx(NOISY_CER)
    assert edited["wer"] == pytest.approx(NOISY_WER)
    assert edited["ned"] == pytest.approx(NOISY_CER)


def test_score_pages_missing_prediction_skip_and_error(tmp_path):
    manifest, gold, pred_dirs = _make_bench(tmp_path)
    rows = load_manifest(str(manifest))

    page_rows, stats = score_pages(rows, str(gold), pred_dirs, missing="skip")
    noisy_pages = [(r["file"], r["page_num"]) for r in page_rows if r["model"] == "noisy"]
    assert ("docB", 2) not in noisy_pages
    assert stats["missing_pred"] == {"perfect": 0, "noisy": 1}

    with pytest.raises(ValueError):
        score_pages(rows, str(gold), pred_dirs, missing="error")


def test_score_pages_missing_gold_excluded_for_all_models(tmp_path):
    manifest, gold, pred_dirs = _make_bench(tmp_path)
    rows = load_manifest(str(manifest))
    page_rows, stats = score_pages(rows, str(gold), pred_dirs)
    assert stats["missing_gold"] == 1
    assert stats["pages_scored"] == 4
    assert all(r["file"] != "docC" for r in page_rows)


def test_score_pages_entity_f1(tmp_path):
    manifest, gold, pred_dirs = _make_bench(tmp_path, entities=True)
    rows = load_manifest(str(manifest))
    page_rows, _ = score_pages(rows, str(gold), pred_dirs, entities=True)

    noisy_a1 = next(
        r for r in page_rows if r["model"] == "noisy" and (r["file"], r["page_num"]) == ("docA", 1)
    )
    assert noisy_a1["entity_f1"] == pytest.approx(0.5)
    assert noisy_a1["entity_tp"] == 1

    # page without a gold sidecar carries no entity fields
    noisy_a2 = next(
        r for r in page_rows if r["model"] == "noisy" and (r["file"], r["page_num"]) == ("docA", 2)
    )
    assert noisy_a2["entity_f1"] is None


# ── aggregate ───────────────────────────────────────────────────────────────
def test_aggregate_macro_average_and_row_order(tmp_path):
    manifest, gold, pred_dirs = _make_bench(tmp_path)
    rows = load_manifest(str(manifest))
    page_rows, stats = score_pages(rows, str(gold), pred_dirs)
    agg_rows = aggregate(page_rows, ["perfect", "noisy"], stats)

    noisy_overall = next(
        r for r in agg_rows if r["model"] == "noisy" and r["tier"] == "overall"
    )
    noisy_scores = [r["cer"] for r in page_rows if r["model"] == "noisy"]
    assert noisy_overall["cer"] == pytest.approx(sum(noisy_scores) / len(noisy_scores))
    assert noisy_overall["n_pages"] == 3
    assert noisy_overall["n_missing"] == 1

    # model order preserved; tiers ordered clean → degraded → hard → overall
    assert [r["model"] for r in agg_rows[:4]] == ["perfect"] * 4
    assert [r["tier"] for r in agg_rows[:4]] == ["clean", "degraded", "hard", "overall"]


def test_aggregate_pools_entities_micro(tmp_path):
    manifest, gold, pred_dirs = _make_bench(tmp_path, entities=True)
    rows = load_manifest(str(manifest))
    page_rows, stats = score_pages(rows, str(gold), pred_dirs, entities=True)
    agg_rows = aggregate(page_rows, ["perfect", "noisy"], stats)

    noisy_overall = next(
        r for r in agg_rows if r["model"] == "noisy" and r["tier"] == "overall"
    )
    # single entity page: tp=1, n_ref=2, n_hyp=2 → micro P = R = F1 = 0.5
    assert noisy_overall["entity_f1"] == pytest.approx(0.5)
    assert noisy_overall["n_entity_pages"] == 1


# ── render_markdown ─────────────────────────────────────────────────────────
def test_render_markdown_tables_and_bold(tmp_path):
    manifest, gold, pred_dirs = _make_bench(tmp_path)
    rows = load_manifest(str(manifest))
    page_rows, stats = score_pages(rows, str(gold), pred_dirs)
    agg_rows = aggregate(page_rows, ["perfect", "noisy"], stats)
    meta = {
        "manifest": "manifest.csv",
        "split": "all",
        "entities": False,
        "missing_gold": stats["missing_gold"],
        "missing_pred": stats["missing_pred"],
    }

    report = render_markdown(agg_rows, ["perfect", "noisy"], meta)
    assert "| Model | Pages | CER (%) | WER (%) | NED |" in report
    assert "| **perfect** | 4 | **0.00** | **0.00** | **0.000** |" in report
    assert "## CER (%) by tier" in report
    assert "> noisy: 1 page(s) missing prediction." in report
    assert "> 1 page(s) missing gold transcription — excluded for all models." in report
    # deterministic: identical output on a second call
    assert report == render_markdown(agg_rows, ["perfect", "noisy"], meta)


# ── _load_config ────────────────────────────────────────────────────────────
def test_load_config_reads_benchmark_section(tmp_path):
    path = tmp_path / "c.txt"
    _write(
        path,
        "[BENCHMARK]\nMANIFEST = m.csv\nGOLD_DIR = gold\n"
        "PRED_DIRS = a=/x, b=/y\nSPLIT = test\nENTITIES = true\n",
    )
    cfg = _load_config(str(path))
    assert cfg["manifest"] == "m.csv"
    assert cfg["gold_dir"] == "gold"
    assert cfg["pred_dirs"] == "a=/x, b=/y"
    assert cfg["split"] == "test"
    assert cfg["entities"] is True
    assert cfg["missing"] == "skip"


# ── main (in-process CLI) ───────────────────────────────────────────────────
def _run_main(tmp_path, out_name, para_name, entities=False):
    manifest, gold, pred_dirs = _make_bench(tmp_path, entities=entities)
    argv = [
        "--manifest", str(manifest),
        "--gold", str(gold),
        "--pred", f"perfect={pred_dirs['perfect']}", f"noisy={pred_dirs['noisy']}",
        "--output-dir", str(tmp_path / out_name),
        "--paradata-dir", str(tmp_path / para_name),
    ]
    if entities:
        argv.append("--entities")
    return main(argv), tmp_path / out_name, tmp_path / para_name


def test_main_end_to_end(tmp_path):
    rc, out_dir, para_dir = _run_main(tmp_path, "out", "para", entities=True)
    assert rc == 0
    for name in (PAGE_CSV, AGG_CSV, REPORT_MD):
        assert (out_dir / name).exists()

    with open(out_dir / PAGE_CSV, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    # 4 gold pages × 2 models − 1 missing noisy prediction
    assert len(rows) == 7
    assert rows[0]["model"] == "perfect"
    assert {"cer", "wer", "ned", "tier", "split", "entity_f1"} <= set(rows[0])

    para_files = glob.glob(str(para_dir / "*_llm-enrich.json"))
    assert len(para_files) == 1
    with open(para_files[0], encoding="utf-8") as fh:
        para = json.load(fh)
    assert para["statistics"]["skipped_files"] >= 1


def test_main_is_deterministic(tmp_path):
    rc1, out1, _ = _run_main(tmp_path, "out1", "para1")
    rc2, out2, _ = _run_main(tmp_path, "out2", "para2")
    assert rc1 == rc2 == 0
    for name in (PAGE_CSV, REPORT_MD):
        assert (out1 / name).read_bytes() == (out2 / name).read_bytes()
