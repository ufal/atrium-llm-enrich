"""flexiconv_convert.py -- convert one non-tabular document to TEITOK XML with flexiconv.

Canonical copy: atrium-nlp-enrich ``api_util/flexiconv_convert.py`` (atrium-llm-enrich
vendors it verbatim; its ``tests/test_vendored_teitok_parity.py`` pins the hash).

The only module that touches flexiconv (optional dependency, lazily imported). Called per
file by atrium-nlp-enrich's ``api_flexiconv.sh``, or standalone:
``python3 api_util/flexiconv_convert.py IN --out-dir DIR [--force]``. Readers
(``teitok_read.py``) never import it.

* Library path: ``flexiconv.api.run_convert()`` -- the package's programmatic entry point.
  It reports failure through ``ConvertResult.success``/``error_message``, not by raising.
  (There is no top-level ``flexiconv.convert()``; calling it used to fail with
  ``AttributeError`` and silently fall through to the CLI on every file.)
* CLI fallback: ``flexiconv --no-auto-install -t teitok IN OUT``. ``--no-auto-install``
  stops the CLI from ``pip install``-ing format extras mid-run; the extras are pinned in
  ``requirements_flexiconv.txt`` instead.
* Output name: ``<stem>.teitok.xml``. When the same input directory holds several
  convertible files with one stem (``report.txt`` + ``report.md``), each one is written as
  ``<stem>.<ext>.teitok.xml`` so no conversion overwrites another. The rule depends only on
  the directory listing, so it is the same in every run.
* An existing output is kept (resume) unless ``force`` is set.
* Exit codes of the command line: 0 converted (or kept), 1 flexiconv could not convert this
  file, 3 flexiconv is not installed (neither library nor CLI). ``--check`` exits 0 or 3
  without converting anything, so a batch can stop before its first file instead of
  failing on every one of them.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

INSTALL_HINT = "Flexiconv is not installed. Please run: pip install -r requirements_flexiconv.txt"
FLEXICONV_EXTENSIONS = frozenset(
    {
        "pdf",
        "docx",
        "odt",
        "rtf",
        "html",
        "htm",
        "md",
        "txt",
        "epub",
        "tei",
        "conllu",
        "vert",
        "folia",
        "srt",
        # Layout formats. flexiconv sniffs .xml content itself (PAGE XML, ALTO, TEI, TEITOK).
        "xml",
        "hocr",
    }
)
_TABULAR = frozenset({"csv", "xlsx"})
CLI_TIMEOUT_S = 300
EXIT_CONVERSION_FAILED = 1
EXIT_NOT_INSTALLED = 3


class FlexiconvNotInstalled(RuntimeError):
    pass


class FlexiconvConversionError(RuntimeError):
    """flexiconv is installed but could not convert this file."""


def normalize_ext_list(s: str) -> frozenset:
    if not s:
        return frozenset()
    return frozenset(
        ext.strip().lower().lstrip(".") for ext in s.replace(",", " ").split() if ext.strip()
    )


def is_flexiconv_format(path: str | Path, allowed: frozenset = None) -> bool:
    if allowed is None:
        allowed = FLEXICONV_EXTENSIONS
    ext = Path(path).suffix.lower().lstrip(".")
    return ext in allowed and ext not in _TABULAR


def _library_run_convert():
    """Return ``flexiconv.api.run_convert`` or None if the library is not importable."""
    try:
        from flexiconv.api import run_convert
    except ImportError:
        return None
    return run_convert


def flexiconv_available() -> bool:
    """Checks if flexiconv is available via Python lib or CLI without raising."""
    return _library_run_convert() is not None or shutil.which("flexiconv") is not None


def output_path_for(in_path: str | Path, out_dir: str | Path, allowed: frozenset = None) -> Path:
    """Where the TEITOK for ``in_path`` goes (see the module docstring for the naming rule)."""
    in_path = Path(in_path)
    out_dir = Path(out_dir)
    stem = in_path.stem
    siblings = []
    if in_path.parent.is_dir():
        siblings = [
            p
            for p in in_path.parent.iterdir()
            if p.is_file() and p != in_path and p.stem == stem and is_flexiconv_format(p, allowed)
        ]
    if siblings:
        ext = in_path.suffix.lower().lstrip(".")
        return out_dir / f"{stem}.{ext}.teitok.xml"
    return out_dir / f"{stem}.teitok.xml"


def _run_cli(cli_path: str, in_path: Path, out_path: Path, force: bool) -> None:
    cmd = [cli_path, "--no-auto-install"]
    if force:
        cmd.append("--force")
    cmd += ["-t", "teitok", str(in_path), str(out_path)]
    try:
        subprocess.run(cmd, check=True, timeout=CLI_TIMEOUT_S, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip().splitlines()
        raise FlexiconvConversionError(detail[-1] if detail else f"exit {exc.returncode}") from None
    except subprocess.TimeoutExpired:
        raise FlexiconvConversionError(f"timed out after {CLI_TIMEOUT_S}s") from None


def convert_to_teitok(
    in_path: str | Path,
    out_dir: str | Path,
    force: bool = False,
    allowed: frozenset = None,
) -> str:
    """Converts ``in_path`` to TEITOK XML in ``out_dir`` and returns the output path.

    Raises ``FlexiconvNotInstalled`` when neither the library nor the CLI is present and
    ``FlexiconvConversionError`` when flexiconv cannot convert the file.
    """
    if not flexiconv_available():
        raise FlexiconvNotInstalled(INSTALL_HINT) from None

    in_path = Path(in_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_path_for(in_path, out_dir, allowed)

    if out_path.exists() and not force:
        return str(out_path)

    run_convert = _library_run_convert()
    if run_convert is not None:
        result = run_convert(
            str(in_path), str(out_path), to_format="teitok", options={"force": force}
        )
        if not getattr(result, "success", False):
            message = getattr(result, "error_message", None) or "flexiconv reported failure"
            raise FlexiconvConversionError(message)
    else:
        cli_path = shutil.which("flexiconv")
        if not cli_path:
            raise FlexiconvNotInstalled(INSTALL_HINT) from None
        _run_cli(cli_path, in_path, out_path, force)

    if not out_path.exists():
        raise FlexiconvConversionError(f"flexiconv reported success but wrote no {out_path.name}")
    return str(out_path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Convert one document to TEITOK XML (flexiconv).")
    parser.add_argument("input_file", type=Path, nargs="?")
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Only report whether flexiconv is installed (exit 0) or not (exit 3).",
    )
    parser.add_argument(
        "--force", action="store_true", help="Overwrite an existing <stem>.teitok.xml."
    )
    parser.add_argument(
        "--formats",
        default="",
        help="Allowed extensions (space/comma separated) used for the same-stem naming rule.",
    )
    args = parser.parse_args(argv)
    if args.check:
        if flexiconv_available():
            print("[OK] flexiconv is available")
            return 0
        print(f"[FAIL] {INSTALL_HINT}", file=sys.stderr)
        return EXIT_NOT_INSTALLED
    if args.input_file is None or args.out_dir is None:
        parser.error("input_file and --out-dir are required unless --check is given")
    allowed = normalize_ext_list(args.formats) or None

    existed = output_path_for(args.input_file, args.out_dir, allowed).exists()
    try:
        out = convert_to_teitok(args.input_file, args.out_dir, force=args.force, allowed=allowed)
    except FlexiconvNotInstalled as exc:
        print(f"[FAIL] {args.input_file.name}: {exc}", file=sys.stderr)
        return EXIT_NOT_INSTALLED
    except FlexiconvConversionError as exc:
        print(f"[FAIL] {args.input_file.name}: {exc}", file=sys.stderr)
        return EXIT_CONVERSION_FAILED
    status = "SKIP" if existed and not args.force else "OK"
    print(f"[{status}] {args.input_file.name} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
