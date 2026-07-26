"""
tests/test_bbox_scale.py – Unit tests for api_util/bbox_scale.py, the pure
hOCR/TEITOK coordinate-rescaling helpers (unit conversion, bbox scaling, surface
extent rewriting, source-size detection).

This module is byte-identical to the copy in atrium-nlp-enrich; here it is
covered directly (nlp reaches it via its service/rescale layer).
"""

from api_util.bbox_scale import (
    detect_source_size,
    dpi_scale,
    fix_name_close_tags,
    rewrite_bboxes,
    scale_bbox_coords,
    set_surface_extent,
    unit_per_inch,
)


# ── unit_per_inch / dpi_scale ───────────────────────────────────────────────
def test_unit_per_inch_known_units():
    assert unit_per_inch("inch1200") == 1200.0
    assert unit_per_inch("mm10") == 254.0


def test_unit_per_inch_unknown_returns_none():
    assert unit_per_inch("pixel") is None


def test_dpi_scale_no_dpi_is_identity():
    assert dpi_scale("inch1200", None) == (1.0, 1.0)


def test_dpi_scale_known_unit():
    assert dpi_scale("inch1200", 600) == (0.5, 0.5)


def test_dpi_scale_pixel_uses_alto_dpi():
    assert dpi_scale("pixel", 300, alto_dpi=600) == (0.5, 0.5)


def test_dpi_scale_unknown_unit_is_identity():
    assert dpi_scale("weird", 300) == (1.0, 1.0)


# ── scale_bbox_coords ───────────────────────────────────────────────────────
def test_scale_bbox_coords_scales_and_rounds():
    assert scale_bbox_coords("10 20 30 40", 2.0, 2.0) == "20 40 60 80"


def test_scale_bbox_coords_applies_offset():
    assert scale_bbox_coords("10 10 20 20", 1.0, 1.0, dx=5.0, dy=5.0) == "5 5 15 15"


def test_scale_bbox_coords_non_four_parts_unchanged():
    assert scale_bbox_coords("10 20 30", 2.0, 2.0) == "10 20 30"


def test_scale_bbox_coords_non_numeric_unchanged():
    assert scale_bbox_coords("a b c d", 2.0, 2.0) == "a b c d"


# ── tag repair / surface / bbox rewrite ─────────────────────────────────────
def test_fix_name_close_tags_repairs_and_counts():
    fixed, n = fix_name_close_tags("<name>X</n> and <name>Y</n>")
    assert fixed == "<name>X</name> and <name>Y</name>"
    assert n == 2


def test_fix_name_close_tags_noop_when_well_formed():
    fixed, n = fix_name_close_tags("<name>X</name>")
    assert n == 0


def test_set_surface_extent_rewrites_lrx_lry():
    out = set_surface_extent('<surface lrx="100" lry="200">', 999, 888)
    assert 'lrx="999"' in out and 'lry="888"' in out


def test_rewrite_bboxes_applies_scale_fn():
    out = rewrite_bboxes('x <span bbox="1 2 3 4"> y', lambda v: "9 9 9 9")
    assert 'bbox="9 9 9 9"' in out


# ── detect_source_size ──────────────────────────────────────────────────────
def test_detect_source_size_from_surface():
    assert detect_source_size('<surface lrx="1000" lry="2000"></surface>') == (1000, 2000, "surface")


def test_detect_source_size_from_bbox_extent():
    xml = '<a bbox="0 0 100 50"/><b bbox="0 0 300 200"/>'
    assert detect_source_size(xml) == (300, 200, "bbox-extent")


def test_detect_source_size_none_when_empty():
    assert detect_source_size("<doc/>") == (None, None, None)
