import xml.etree.ElementTree as ET
from pathlib import Path

from api_util.bbox_scale import fix_name_close_tags
from atrium_document import canonical_doc_id


def doc_id_from_path(path: str | Path) -> str:
    """Strips a known pipeline suffix (.teitok.xml, .udpipe.conllu, .conllu, …) to
    produce a clean document ID.

    Delegates to ``atrium_document.canonical_doc_id()`` (atrium-project#10, D3). This
    function is the "bespoke TEITOK/CoNLL-U stripper" the hub changelog names as one of the
    four derivations ``canonical_doc_id()`` was written to retire, and it was wrong in a way
    no caller could see: it sliced ``.teitok.xml`` and ``.conllu`` off by LITERAL LENGTH, so
    ``X.udpipe.conllu`` came back as ``X.udpipe`` while every other tool in the pipeline
    resolves that same file to ``X`` — ``KNOWN_PIPELINE_SUFFIXES`` lists ``.udpipe.conllu``
    ahead of ``.conllu`` precisely so the longer suffix matches first. Latent only because
    this repo's input filters never feed it a ``.conllu`` today; the function is public and
    documented for it.

    The name and signature stay: ``llm_run.py``, ``llm_client_shared.py``, ``xml_to_md.py``,
    ``docx_to_md.py`` and ``pdf_to_md.py`` all call it, so delegating here fixes every
    caller at one change point instead of eleven.
    """
    return canonical_doc_id(path)


def parse_teitok(path: str | Path) -> ET.Element:
    """Reads a TEITOK/TEI XML file and returns its root Element.

    Repairs the known ``<name>...</n>`` mis-close quirk (issue #13 §D.2; see
    ``bbox_scale.fix_name_close_tags``) before parsing, so a document that's
    well-formed except for that one documented quirk doesn't hard-crash
    ``ET.parse`` with an unhandled ``ParseError``. Centralised here so every
    TEITOK reader (this module's own functions, plus ``xml_to_md.py``'s
    layout reader) applies the same repair instead of each parsing the raw
    file directly and re-discovering the same crash independently.
    """
    text = Path(path).read_text(encoding="utf-8")
    fixed_text, _ = fix_name_close_tags(text)
    return ET.fromstring(fixed_text)


def read_teitok_rows(path: str | Path) -> list[dict]:
    """
    Parses TEITOK XML.
    Returns: list of dicts [{"page_num": int, "line_num": int, "text": str}]
    """
    root = parse_teitok(path)
    rows = []

    page_num = 1
    line_num = 1

    for elem in root.iter():
        tag = elem.tag.split("}")[-1]  # Namespace agnostic

        if tag == "pb":
            # `n` is usually a plain page count, but archival front matter /
            # appendices legitimately use roman numerals or other non-numeric
            # labels (e.g. n="I", n="priloha1") — fall back to a simple
            # increment rather than crashing on int(), mirroring how the
            # ALTO reader already treats an unparseable PHYSICAL_IMG_NR.
            try:
                page_num = int(elem.get("n", page_num + 1))
            except (TypeError, ValueError):
                page_num += 1
        elif tag == "lb":
            line_num += 1
        elif tag == "s":
            text = elem.get("text")

            # If @text is missing, fallback to joining <tok> elements
            if not text:
                toks = []
                for tok in elem.iter():
                    tok_tag = tok.tag.split("}")[-1]
                    if tok_tag == "tok":
                        toks.append(tok.text or "")
                        if tok.get("join") != "right" and tok.get("spaceAfter") != "No":
                            toks.append(" ")
                text = "".join(toks).strip()

            if text:
                rows.append({"page_num": page_num, "line_num": line_num, "text": text})

    return rows


def read_teitok_text(path: str | Path) -> str:
    """Returns the surface text as a single string."""
    rows = read_teitok_rows(path)
    return "\n".join(r["text"] for r in rows)


def read_teitok_tokens(path: str | Path) -> list[dict]:
    """
    Returns token-level annotations.
    Returns: list of dicts [{"form", "lemma", "upos", "space_after"}]
    """
    root = parse_teitok(path)
    tokens = []

    for tok in root.iter():
        tag = tok.tag.split("}")[-1]
        if tag == "tok":
            tokens.append(
                {
                    "form": tok.text or "",
                    "lemma": tok.get("lemma", ""),
                    "upos": tok.get("pos", tok.get("type", "")),
                    "space_after": tok.get("join") != "right" and tok.get("spaceAfter") != "No",
                }
            )

    return tokens
