"""
tests/conftest.py
=================
Shared pytest fixtures and sys.path wiring for atrium-nlp-enrich unit tests.

sys.path is patched here (once, at collection time) so that every test module
can import from both the repo root (``keywords.py``, ``atrium_paradata.py``)
and the ``api_util/`` subdirectory (``call_udpipe``, ``call_nametag``,
``summarize_nt_udp``).
"""

import sys
from pathlib import Path

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: marks tests as slow integration smoke tests")


# ── path wiring ───────────────────────────────────────────────────────────────
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "api_util"))

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ── CoNLL-U fixtures (shared with atrium-nlp-enrich; teitok_alto is identical) ──


@pytest.fixture
def sample_conllu(tmp_path):
    """Three-sentence CoNLL-U file written to a temp path."""
    content = (FIXTURES_DIR / "sample.conllu").read_text(encoding="utf-8")
    dest = tmp_path / "sample.conllu"
    dest.write_text(content, encoding="utf-8")
    return str(dest)


@pytest.fixture
def two_page_conllu(tmp_path):
    """CoNLL-U whose sent_id counter resets to 1 mid-file (two-page document)."""
    content = (FIXTURES_DIR / "two_page.conllu").read_text(encoding="utf-8")
    dest = tmp_path / "two_page.conllu"
    dest.write_text(content, encoding="utf-8")
    return str(dest)


@pytest.fixture
def page_break_conllu(tmp_path):
    """Merged CoNLL-U using ``# page_break = true`` comments instead of sent_id resets."""
    content = (FIXTURES_DIR / "page_break.conllu").read_text(encoding="utf-8")
    dest = tmp_path / "page_break.conllu"
    dest.write_text(content, encoding="utf-8")
    return str(dest)

@pytest.fixture
def empty_conllu(tmp_path):
    """CoNLL-U file with only a comment header — no token lines."""
    dest = tmp_path / "empty.conllu"
    dest.write_text("# newdoc\n", encoding="utf-8")
    return str(dest)

