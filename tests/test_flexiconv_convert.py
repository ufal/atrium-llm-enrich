import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_api_util_path = str(Path(__file__).parent.parent / "api_util")
if _api_util_path not in sys.path:
    sys.path.insert(0, _api_util_path)

from api_util import flexiconv_convert  # noqa: E402
from api_util.flexiconv_convert import (  # noqa: E402
    FlexiconvConversionError,
    FlexiconvNotInstalled,
    convert_to_teitok,
    flexiconv_available,
    is_flexiconv_format,
    normalize_ext_list,
    output_path_for,
)
from api_util.teitok_read import read_teitok_rows  # noqa: E402


def test_ext_normalization():
    assert normalize_ext_list("pdf docx, ODT") == frozenset({"pdf", "docx", "odt"})
    assert normalize_ext_list(".md") == frozenset({"md"})
    assert normalize_ext_list("") == frozenset()


def test_is_flexiconv_format():
    allowed = frozenset({"pdf", "docx", "txt"})
    assert is_flexiconv_format("doc.pdf", allowed) is True
    assert is_flexiconv_format("path/to/file.TXT", allowed) is True
    assert is_flexiconv_format("data.csv", allowed) is False
    assert is_flexiconv_format("data.xlsx", allowed) is False


def test_layout_formats_are_routed_by_default():
    """PAGE XML / ALTO / TEI arrive as .xml (flexiconv sniffs the content), hOCR as .hocr."""
    assert is_flexiconv_format("scan.page.xml") is True
    assert is_flexiconv_format("page_1.hocr") is True


def test_output_name_is_stem_when_unique(tmp_path):
    (tmp_path / "report.txt").write_text("x")
    assert output_path_for(tmp_path / "report.txt", tmp_path / "out").name == "report.teitok.xml"


def test_same_stem_inputs_get_distinct_outputs(tmp_path):
    """report.txt and report.md used to both map to report.teitok.xml; the second one then
    failed on flexiconv's refusal to overwrite."""
    for name in ("report.txt", "report.md"):
        (tmp_path / name).write_text("x")
    out = tmp_path / "out"
    assert output_path_for(tmp_path / "report.txt", out).name == "report.txt.teitok.xml"
    assert output_path_for(tmp_path / "report.md", out).name == "report.md.teitok.xml"


def _hide_library(monkeypatch):
    monkeypatch.setattr(flexiconv_convert, "_library_run_convert", lambda: None)


def test_cli_fallback_passes_no_auto_install(monkeypatch, tmp_path):
    """Without the library the adapter shells out -- and must stop the CLI from
    pip-installing format extras in the middle of a run."""
    in_file = tmp_path / "test.docx"
    in_file.write_text("dummy")
    out_dir = tmp_path / "out"
    _hide_library(monkeypatch)
    monkeypatch.setattr(shutil, "which", lambda x: "/usr/bin/flexiconv")

    called = []

    def fake_run(args, **kwargs):
        called.append(args)
        Path(args[-1]).write_text("<TEI/>", encoding="utf-8")

    monkeypatch.setattr(subprocess, "run", fake_run)

    out_path = convert_to_teitok(in_file, out_dir)
    assert Path(out_path).name == "test.teitok.xml"
    assert called == [
        ["/usr/bin/flexiconv", "--no-auto-install", "-t", "teitok", str(in_file), str(out_path)]
    ]


def test_cli_fallback_force_flag(monkeypatch, tmp_path):
    in_file = tmp_path / "test.docx"
    in_file.write_text("dummy")
    _hide_library(monkeypatch)
    monkeypatch.setattr(shutil, "which", lambda x: "/usr/bin/flexiconv")
    called = []

    def fake_run(args, **kwargs):
        called.append(args)
        Path(args[-1]).write_text("<TEI/>", encoding="utf-8")

    monkeypatch.setattr(subprocess, "run", fake_run)
    convert_to_teitok(in_file, tmp_path / "out", force=True)
    assert "--force" in called[0]


def test_cli_failure_is_a_typed_error_not_a_traceback(monkeypatch, tmp_path):
    in_file = tmp_path / "test.docx"
    in_file.write_text("dummy")
    _hide_library(monkeypatch)
    monkeypatch.setattr(shutil, "which", lambda x: "/usr/bin/flexiconv")

    def fake_run(args, **kwargs):
        raise subprocess.CalledProcessError(1, args, stderr="[flexiconv] DOCX support requires x")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(FlexiconvConversionError, match="DOCX support requires"):
        convert_to_teitok(in_file, tmp_path / "out")


def test_library_path_uses_run_convert_and_checks_success(monkeypatch, tmp_path):
    in_file = tmp_path / "doc.txt"
    in_file.write_text("dummy")
    calls = []

    class Result:
        def __init__(self, success, error_message=None):
            self.success = success
            self.error_message = error_message

    def fake_run_convert(inp, out, to_format=None, options=None):
        calls.append((inp, out, to_format, options))
        return Result(False, "Unknown input format: txt")

    monkeypatch.setattr(flexiconv_convert, "_library_run_convert", lambda: fake_run_convert)
    with pytest.raises(FlexiconvConversionError, match="Unknown input format"):
        convert_to_teitok(in_file, tmp_path / "out")
    assert calls[0][2] == "teitok" and calls[0][3] == {"force": False}


def test_existing_output_is_kept_unless_forced(monkeypatch, tmp_path):
    in_file = tmp_path / "doc.txt"
    in_file.write_text("dummy")
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    existing = out_dir / "doc.teitok.xml"
    existing.write_text("<TEI>old</TEI>", encoding="utf-8")

    def boom(*a, **k):
        raise AssertionError("must not convert again")

    monkeypatch.setattr(flexiconv_convert, "_library_run_convert", lambda: boom)
    assert convert_to_teitok(in_file, out_dir) == str(existing)
    assert existing.read_text(encoding="utf-8") == "<TEI>old</TEI>"


def test_convert_raises_when_missing(monkeypatch, tmp_path):
    """Missing both Python library and CLI cleanly raises FlexiconvNotInstalled."""
    in_file = tmp_path / "test.docx"
    _hide_library(monkeypatch)
    monkeypatch.setattr(shutil, "which", lambda x: None)

    with pytest.raises(FlexiconvNotInstalled, match="Flexiconv is not installed"):
        convert_to_teitok(in_file, tmp_path / "out")


def test_main_exits_3_when_flexiconv_is_missing(monkeypatch, tmp_path, capsys):
    """Not installed is not a per-file failure: api_flexiconv.sh stops on exit 3."""
    in_file = tmp_path / "test.docx"
    in_file.write_text("x")
    _hide_library(monkeypatch)
    monkeypatch.setattr(shutil, "which", lambda x: None)
    assert flexiconv_convert.main([str(in_file), "--out-dir", str(tmp_path / "out")]) == 3
    err = capsys.readouterr().err
    assert err.startswith("[FAIL] test.docx:") and "Traceback" not in err


def test_main_prints_one_line_and_exits_1_when_a_conversion_fails(monkeypatch, tmp_path, capsys):
    in_file = tmp_path / "test.docx"
    in_file.write_text("x")

    class _Result:
        success = False
        error_message = "Package not found"

    monkeypatch.setattr(
        flexiconv_convert, "_library_run_convert", lambda: lambda *a, **k: _Result()
    )
    assert flexiconv_convert.main([str(in_file), "--out-dir", str(tmp_path / "out")]) == 1
    err = capsys.readouterr().err
    assert err.strip() == "[FAIL] test.docx: Package not found"


def test_check_reports_availability_without_converting(monkeypatch, capsys):
    _hide_library(monkeypatch)
    monkeypatch.setattr(shutil, "which", lambda x: None)
    assert flexiconv_convert.main(["--check"]) == 3
    assert "not installed" in capsys.readouterr().err
    monkeypatch.setattr(shutil, "which", lambda x: "/usr/bin/flexiconv")
    assert flexiconv_convert.main(["--check"]) == 0


def test_main_requires_an_input_without_check():
    with pytest.raises(SystemExit):
        flexiconv_convert.main([])


@pytest.mark.skipif(not flexiconv_available(), reason="flexiconv library or CLI is not installed")
def test_live_conversion_yields_readable_rows(tmp_path):
    """Live: a real conversion must produce rows the readers can use -- the file merely
    existing is what the old test checked, and it hid that every converted document read
    as zero rows."""
    in_file = tmp_path / "sample.txt"
    in_file.write_text("Hello world. This is a real test.\n\nSecond paragraph.\n", encoding="utf-8")
    out_path = convert_to_teitok(in_file, tmp_path / "out")
    assert Path(out_path).name == "sample.teitok.xml"
    rows = read_teitok_rows(out_path)
    assert [r["text"] for r in rows] == ["Hello world. This is a real test.", "Second paragraph."]


@pytest.mark.skipif(
    not shutil.which("flexiconv"), reason="flexiconv CLI not installed in current environment"
)
def test_live_cli_fallback(monkeypatch, tmp_path):
    """Live: the CLI path works against a real flexiconv installation."""
    _hide_library(monkeypatch)
    in_file = tmp_path / "sample.txt"
    in_file.write_text("Fallback test.", encoding="utf-8")
    out_path = convert_to_teitok(in_file, tmp_path / "out")
    assert Path(out_path).name == "sample.teitok.xml"
    assert read_teitok_rows(out_path)
