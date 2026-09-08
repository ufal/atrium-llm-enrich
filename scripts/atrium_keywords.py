#!/usr/bin/env python3
"""Zero-dependency client for the ATRIUM LLM Enrichment (keyword) API.

Uploads document text (TXT/MD document-level, or CSV/TEITOK line-level file, or plain
text on stdin) to a running instance of the FastAPI service in `service/api.py` and
returns vocabulary-guided archaeological keywords: Czech + English terms, a TEATER
category, and a confidence score per line or per located passage (local server by
default, remote via --base-url or the ATRIUM_LE_URL env variable).

NOTE on backend/model selection: unlike earlier versions of this client, the LLM
backend (`openrouter` or `ollama`) and model are chosen server-side, once, at server
startup (`LLM_BACKEND` / `OPENROUTER_MODEL` / `OLLAMA_MODEL` env vars) — there is no
per-request override. Use `--info` to see which backend/model the running server has
warmed up.

Only the Python 3 standard library is used - no pip installs required.

Usage:
    python3 scripts/atrium_keywords.py lines.csv               # line-level (CSV/TEITOK)
    python3 scripts/atrium_keywords.py page.teitok.xml --format json
    python3 scripts/atrium_keywords.py notes.md                # document-level (MD/TXT)
    python3 scripts/atrium_keywords.py - < notes.txt            # document-level, stdin
    python3 scripts/atrium_keywords.py --info

    # ATRIUM Document JSON accretion (docs/document_schema.md, issue #13): accrete this
    # tool's `enrichment` block onto an existing baseline record
    python3 scripts/atrium_keywords.py page.teitok.xml --document-json in.document.json \
        --document-json-out-file out.document.json

Exit codes:
    0 - success
    1 - client-side error (bad arguments, unreadable file)
    2 - server unreachable (connection refused / timeout)
    3 - server-side error (HTTP 4xx/5xx)
"""

import argparse
import csv
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

DEFAULT_BASE_URL = os.environ.get("ATRIUM_LE_URL", "http://localhost:8000")
LINE_SUFFIXES = (".csv", ".teitok.xml")
DOC_SUFFIXES = (".md", ".txt")
MAX_UPLOAD_MB = 10  # mirrors the server's MAX_UPLOAD_MB default
RETRY_STATUS = {502, 503, 504}
RETRY_ATTEMPTS = 3
RETRY_WAIT_S = 10


def build_multipart(files: dict) -> tuple[bytes, str]:
    """Encode one or more files as multipart/form-data using only the stdlib.

    `files` maps the multipart field name to a `Path` (e.g. `{"file": page.teitok.xml,
    "document_json": baseline.json}` for the accretion contract).
    """
    boundary = uuid.uuid4().hex
    lines = []
    for field_name, file_path in files.items():
        mime = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        lines.append(f"--{boundary}".encode())
        lines.append(
            f'Content-Disposition: form-data; name="{field_name}"; filename="{file_path.name}"'.encode()
        )
        lines.append(f"Content-Type: {mime}".encode())
        lines.append(b"")
        lines.append(file_path.read_bytes())
    lines.append(f"--{boundary}--".encode())
    lines.append(b"")

    body = b"\r\n".join(lines)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def http_json(url: str, data: bytes = None, content_type: str = None, timeout: int = 1800) -> dict:
    """POST (or GET when data is None) and decode a JSON response, with retry on 502/503/504.

    The long default timeout is deliberate: LLM extraction is the slowest call
    in the ATRIUM family (minutes for a full document).
    """
    last_error = None
    for attempt in range(1, RETRY_ATTEMPTS + 1):
        request = urllib.request.Request(url, data=data, method="POST" if data else "GET")
        if content_type:
            request.add_header("Content-Type", content_type)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", errors="replace")
            if e.code in RETRY_STATUS and attempt < RETRY_ATTEMPTS:
                print(
                    f"[retry {attempt}/{RETRY_ATTEMPTS}] HTTP {e.code}, waiting {RETRY_WAIT_S}s...",
                    file=sys.stderr,
                )
                time.sleep(RETRY_WAIT_S)
                last_error = f"HTTP {e.code}: {detail}"
                continue
            if e.code == 503:
                print(
                    "Server busy or backend not ready (HTTP 503): check --info for `ready` "
                    "and the server logs for the configured LLM_BACKEND.",
                    file=sys.stderr,
                )
            else:
                print(f"Server error - HTTP {e.code}: {detail}", file=sys.stderr)
            sys.exit(3)
        except (urllib.error.URLError, TimeoutError) as e:
            print(
                f"Cannot reach the API at {url} ({e}).\nIs the server running? Start it with: bash scripts/server.sh",
                file=sys.stderr,
            )
            sys.exit(2)
    print(f"Server error after {RETRY_ATTEMPTS} attempts - {last_error}", file=sys.stderr)
    sys.exit(3)


def extract_file(base_url: str, path: Path, document_json: Optional[Path] = None) -> dict:
    """Upload one file to POST /extract_keywords (line-level or document-level by suffix)."""
    if not path.name.lower().endswith(LINE_SUFFIXES + DOC_SUFFIXES):
        print(
            f"Skipping {path}: unsupported type. Allowed: {', '.join(LINE_SUFFIXES + DOC_SUFFIXES)}",
            file=sys.stderr,
        )
        return {}
    size = path.stat().st_size
    if size > MAX_UPLOAD_MB * 1024 * 1024:
        print(
            f"Skipping {path}: {size} bytes exceeds the {MAX_UPLOAD_MB} MB server upload limit - "
            "split the document first",
            file=sys.stderr,
        )
        return {}
    files = {"file": path}
    if document_json is not None:
        files["document_json"] = document_json
    body, content_type = build_multipart(files)
    return http_json(f"{base_url}/extract_keywords", data=body, content_type=content_type)


def extract_stdin(base_url: str, document_json: Optional[Path] = None) -> dict:
    """Read plain document text from stdin and send it to POST /extract_keywords_text.

    Document-level only (mirrors uploading a .md/.txt file): the server has no
    per-line JSON entry point any more, only `{"text": ..., "document_json": ...}`.
    """
    text = sys.stdin.read()
    if not text.strip():
        print("No text on stdin.", file=sys.stderr)
        sys.exit(1)
    payload = {"text": text}
    if document_json is not None:
        payload["document_json"] = json.loads(document_json.read_text(encoding="utf-8"))
    return http_json(
        f"{base_url}/extract_keywords_text",
        data=json.dumps(payload).encode("utf-8"),
        content_type="application/json",
    )


def result_rows(name: str, envelope: dict) -> list[tuple]:
    """Flatten an envelope's `results` into (doc, locator, category, confidence, kw_cs, kw_en) rows.

    Line-level records carry `page`/`line`; document-level records carry `page`/`locator`
    instead (no fixed line grid) - both are rendered into one LOCATOR column.
    """
    rows = []
    doc_id = envelope.get("doc_id", name)
    for item in envelope.get("results", []):
        enrichment = item.get("enrichment", {})
        if "line" in item:
            locator = f"p{item.get('page')}/l{item.get('line')}"
        else:
            locator = f"p{item.get('page')}:{item.get('locator', '')}"
        rows.append(
            (
                doc_id,
                locator,
                enrichment.get("teater_category", ""),
                enrichment.get("confidence_score"),
                "; ".join(enrichment.get("extracted_keywords_cs") or []),
                "; ".join(enrichment.get("extracted_keywords_en") or []),
            )
        )
    return rows


def print_table(rows: list[tuple], as_csv: bool) -> None:
    header = ("DOC", "LOCATOR", "CATEGORY", "CONF", "KEYWORDS_CS", "KEYWORDS_EN")
    if as_csv:
        writer = csv.writer(sys.stdout)
        writer.writerow(header)
        for row in rows:
            conf = "" if row[3] is None else f"{row[3]:.2f}"
            writer.writerow([row[0], row[1], row[2], conf, row[4], row[5]])
    else:
        print(f"{header[0]:<20} {header[1]:<10} {header[2]:<24} {header[3]:>5} {header[4]}")
        for row in rows:
            conf = "    -" if row[3] is None else f"{row[3]:>5.2f}"
            keywords = row[4] if len(row[4]) <= 45 else row[4][:42] + "..."
            print(f"{row[0]:<20} {row[1]:<10} {row[2]:<24} {conf} {keywords}")


def summarize(envelope: dict) -> None:
    stats = envelope.get("stats") or {}
    print(
        f"# doc_id={envelope.get('doc_id')} mode={envelope.get('mode')} "
        f"backend={envelope.get('backend')} model={envelope.get('model')} "
        f"processed={stats.get('processed')} filtered={stats.get('skipped_filter')} "
        f"errors={stats.get('skipped_error')}",
        file=sys.stderr,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("files", nargs="*", help="CSV/TEITOK (line-level) or MD/TXT (document-level) file(s), or '-' for stdin text")
    parser.add_argument(
        "--base-url", default=DEFAULT_BASE_URL, help=f"API base URL (default: {DEFAULT_BASE_URL}, env: ATRIUM_LE_URL)"
    )
    parser.add_argument(
        "--format", choices=["table", "csv", "json"], default="table", help="output format (default: table)"
    )
    parser.add_argument("--info", action="store_true", help="print service capabilities, backend/model, and limits, then exit")
    parser.add_argument(
        "--document-json",
        metavar="PATH",
        help="baseline ATRIUM Document JSON to accrete this tool's `enrichment` block onto "
        "(docs/document_schema.md); requires exactly one input file",
    )
    parser.add_argument(
        "--document-json-out-file",
        metavar="PATH",
        help="save the returned document_json record to PATH (default: only embedded in --format json output)",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")

    if args.info:
        print(json.dumps(http_json(f"{base_url}/info", timeout=60), indent=2))
        return

    if not args.files:
        parser.error("no input files given (or use --info)")

    document_json_path = None
    if args.document_json:
        if len(args.files) != 1:
            parser.error("--document-json accretes onto a single document; pass exactly one input file")
        document_json_path = Path(args.document_json)
        if not document_json_path.is_file():
            print(f"--document-json file not found: {document_json_path}", file=sys.stderr)
            sys.exit(1)
    if args.document_json_out_file and document_json_path is None:
        parser.error("--document-json-out-file requires --document-json")

    envelopes = {}
    rows = []
    document_record = None
    for name in args.files:
        if name == "-":
            envelope = extract_stdin(base_url, document_json_path)
        else:
            path = Path(name)
            if not path.is_file():
                print(f"File not found: {path}", file=sys.stderr)
                sys.exit(1)
            envelope = extract_file(base_url, path, document_json_path)
        if not envelope:
            continue
        envelopes[name] = envelope
        summarize(envelope)
        rows.extend(result_rows(name, envelope))
        if envelope.get("document_json") is not None:
            document_record = envelope["document_json"]
        if envelope.get("document_json_schema_error"):
            print(
                f"Warning: uploaded document_json baseline did not validate: "
                f"{envelope['document_json_schema_error']} (record still returned, per rule 6)",
                file=sys.stderr,
            )

    if not envelopes:
        print("No results produced.", file=sys.stderr)
        sys.exit(1)

    if args.document_json_out_file and document_record is not None:
        Path(args.document_json_out_file).write_text(json.dumps(document_record, indent=2), encoding="utf-8")
        print(f"Document JSON record written to {args.document_json_out_file}", file=sys.stderr)

    if args.format == "json":
        print(json.dumps(envelopes if len(envelopes) > 1 else next(iter(envelopes.values())), indent=2, ensure_ascii=False))
    else:
        if not rows:
            print("No passages located. Use --format json for the full envelope.", file=sys.stderr)
            return
        print_table(rows, as_csv=(args.format == "csv"))


if __name__ == "__main__":
    main()
