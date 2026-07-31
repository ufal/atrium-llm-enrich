"""
openrouter_client.py — Remote LLM-as-a-service backend (BACKEND=openrouter).

Reuses the same quality filter, context-window builder, archaeological
schema, and validate_llm_output() as the transformers/vLLM engine (via
llm_client_shared.py — see that module's docstring for why this is a
duplicate front-end rather than an import of llm_utils.py/llm_run.py).

Two input modes, dispatched by file extension:
  * .csv / *.teitok.xml — line-level enrichment, one OpenRouter call per
    qualifying line (matches the transformers/vLLM backends' contract).
  * .md / .txt           — whole-document enrichment, one OpenRouter call
    per document (see llm_client_shared.run_document_level()); typically fed
    from api_util/xml_to_md.py output. Optionally sent as a file attachment
    via --attach-as-file (see _build_attachment_content — best-effort,
    provider/model-dependent, per #24's "explore file-attachment options").

Data-sovereignty: --provider-data-collection deny restricts routing to
OpenRouter providers that don't retain prompts/completions. This does not by
itself resolve model licensing — see the TODO in para_config.txt.

Env:
  OPENROUTER_API_KEY — required unless --api-key is passed.
"""

import argparse
import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import requests
from tqdm import tqdm

from atrium_paradata import ParadataLogger
from llm_client_shared import (
    DOC_CONVERT_EXTENSIONS,
    build_document_schema,
    build_document_system_prompt,
    build_schema,
    build_system_prompt,
    load_config,
    prepare_document_input,
    run_document_level,
    run_line_level,
    write_document_record,
)
from vocab_manager import VocabularyManager

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Mirrors llm_utils.MAX_NEW_TOKENS / CONTEXT_RESERVED — kept as separate
# constants here rather than imported, see llm_client_shared.py docstring.
MAX_NEW_TOKENS = 2048
CONTEXT_RESERVED = MAX_NEW_TOKENS + 512
_DOC_INPUT_EXTENSIONS = {".md", ".txt"}


def _build_headers(
    api_key: str, site_url: Optional[str], app_name: Optional[str]
) -> Dict[str, str]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if site_url:
        headers["HTTP-Referer"] = site_url
    if app_name:
        headers["X-Title"] = app_name
    return headers


def _build_provider_block(
    data_collection: Optional[str], only: Optional[str], order: Optional[str]
) -> Optional[Dict[str, Any]]:
    """OpenRouter 'provider' routing preferences. data_collection='deny'
    restricts routing to providers that don't retain prompts/completions —
    the data-sovereignty option requested in #24."""
    provider: Dict[str, Any] = {}
    if data_collection:
        provider["data_collection"] = data_collection
    if only:
        provider["only"] = [p.strip() for p in only.split(",") if p.strip()]
    if order:
        provider["order"] = [p.strip() for p in order.split(",") if p.strip()]
    return provider or None


def _build_attachment_content(doc_text: str, filename: str, as_file: bool) -> Any:
    """
    Best-effort file-attachment path (per #24: "explore file-attachment
    options"). as_file=True sends the document as an OpenRouter file content
    part (base64 data URL); support for this varies by model/provider, and
    falls back silently to plain inlined text on providers that ignore it.
    as_file=False (default) inlines the markdown directly as message text,
    which works uniformly across every OpenRouter model.
    """
    if not as_file:
        return f"DOCUMENT:\n{doc_text}"

    b64 = base64.b64encode(doc_text.encode("utf-8")).decode("ascii")
    return [
        {"type": "text", "text": "DOCUMENT (attached below):"},
        {
            "type": "file",
            "file": {
                "filename": filename,
                "file_data": f"data:text/markdown;base64,{b64}",
            },
        },
    ]


def make_chat_fn(
    session: requests.Session,
    headers: Dict[str, str],
    model: str,
    schema: Optional[dict],
    max_retries: int,
    timeout: int,
    provider_block: Optional[Dict[str, Any]],
):
    """Returns a llm_client_shared.ChatFn bound to this OpenRouter model/session."""

    def chat_fn(messages: List[Dict[str, str]]) -> str:
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": MAX_NEW_TOKENS,
            "response_format": (
                {"type": "json_schema", "json_schema": {"name": "enrichment", "schema": schema}}
                if schema is not None
                else {"type": "json_object"}
            ),
        }
        if provider_block:
            body["provider"] = provider_block

        last_exc: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = session.post(OPENROUTER_API_URL, headers=headers, json=body, timeout=timeout)
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {resp.status_code}: {resp.text[:200]}")
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except (requests.RequestException, KeyError, IndexError, ValueError) as exc:
                last_exc = exc
                if attempt < max_retries:
                    time.sleep(min(2**attempt, 30))
        raise RuntimeError(f"OpenRouter request failed after {max_retries} attempts: {last_exc}")

    return chat_fn


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="llm_config.txt", help="Shared config file.")
    parser.add_argument(
        "--input", type=Path, default=None, help="File or directory (overrides INPUT_DIR)."
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Overrides OUTPUT_DIR.")
    parser.add_argument(
        "--model", default=None, help="OpenRouter model slug, e.g. 'openai/gpt-4o-mini'."
    )
    parser.add_argument("--api-key", default=None, help="Overrides OPENROUTER_API_KEY.")
    parser.add_argument(
        "--site-url", default=None, help="Sent as HTTP-Referer (OpenRouter app attribution)."
    )
    parser.add_argument(
        "--app-name", default=None, help="Sent as X-Title (OpenRouter app attribution)."
    )
    parser.add_argument("--provider-data-collection", choices=["allow", "deny"], default=None)
    parser.add_argument("--provider-only", default=None, help="Comma-separated provider allowlist.")
    parser.add_argument(
        "--provider-order", default=None, help="Comma-separated provider preference order."
    )
    parser.add_argument(
        "--structured-outputs",
        action="store_true",
        help="Send response_format=json_schema instead of json_object.",
    )
    parser.add_argument(
        "--attach-as-file",
        action="store_true",
        help="Send .md/.txt input as a file content part (best-effort).",
    )
    parser.add_argument(
        "--ocr", action="store_true", help="OCR text-less pages when auto-converting .pdf inputs."
    )
    parser.add_argument(
        "--document-json-dir",
        type=Path,
        default=None,
        help=(
            "Enable the paired per-document record (atrium_document.py): read "
            "<doc_id>.document.json from this dir as the baseline and write it back with "
            "llm-enrich's enrichment block updated. Other tools' blocks pass through."
        ),
    )
    parser.add_argument(
        "--context-window",
        type=int,
        default=128_000,
        help="Model context window, for vocab-truncation budget.",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=120)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    config = load_config(args.config)

    api_key = (
        args.api_key or os.environ.get("OPENROUTER_API_KEY") or config.get("OPENROUTER_API_KEY")
    )
    if not api_key:
        print(
            "[ERROR] No OpenRouter API key: pass --api-key or set OPENROUTER_API_KEY.",
            file=sys.stderr,
        )
        sys.exit(1)

    model = args.model or config.get("OPENROUTER_MODEL")
    if not model:
        print(
            "[ERROR] No model: pass --model or set OPENROUTER_MODEL in llm_config.txt.",
            file=sys.stderr,
        )
        sys.exit(1)

    input_path = args.input or Path(config.get("INPUT_DIR", "data_samples/DOC_LINE_LANG_CLASS"))
    vocab_path = config.get("VOCAB_PATH", "data_samples/teater_nested_vocab.json")
    paradata_dir = config.get("PARADATA_DIR", "paradata")

    model_suffix = model.replace("/", "_").replace(".", "").replace(":", "_")
    output_base = Path(config.get("OUTPUT_DIR", "data_samples/KW_PER_DOC_LLM"))
    output_dir = args.output_dir or (output_base.parent / f"{output_base.name}_{model_suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    include_non_text = config.get("INCLUDE_NON_TEXT", "true").lower() == "true"
    min_char_count = int(config.get("MIN_CHAR_COUNT", "3"))
    min_char_non_text = int(config.get("MIN_CHAR_NON_TEXT", "8"))
    min_alpha_ratio_non_text = float(config.get("MIN_ALPHA_RATIO_NON_TEXT", "0.40"))

    provider_block = _build_provider_block(
        args.provider_data_collection, args.provider_only, args.provider_order
    )

    print(
        f"\n=== LLM Semantic Enrichment Pipeline (BACKEND=openrouter) ===\n    model:   {model}\n    output:  {output_dir}\n"
    )
    if provider_block:
        print(f"  Provider routing: {provider_block}")

    logger = ParadataLogger(
        program="llm-enrich",
        config={
            **config,
            "backend": "openrouter",
            "model": model,
            "output_dir_resolved": str(output_dir),
        },
        paradata_dir=paradata_dir,
        output_types=["json"],
    )

    with logger:
        vocab_mgr = VocabularyManager(vocab_path=vocab_path)
        vocab_data = vocab_mgr.load()

        max_input_tokens = args.context_window - CONTEXT_RESERVED
        line_prompt, line_terms = build_system_prompt(vocab_data, max_tokens=max_input_tokens)
        doc_prompt, doc_terms = build_document_system_prompt(
            vocab_data, max_tokens=max_input_tokens
        )
        LineModel = build_schema(line_terms)
        DocModel = build_document_schema(doc_terms)

        headers = _build_headers(api_key, args.site_url, args.app_name)
        session = requests.Session()

        line_schema = LineModel.model_json_schema() if args.structured_outputs else None
        doc_schema = DocModel.model_json_schema() if args.structured_outputs else None
        line_chat_fn = make_chat_fn(
            session, headers, model, line_schema, args.max_retries, args.timeout, provider_block
        )
        doc_chat_fn = make_chat_fn(
            session, headers, model, doc_schema, args.max_retries, args.timeout, provider_block
        )

        def _make_doc_builder(filename: str) -> Callable[[str], Any]:
            """Per-document user-content builder for run_document_level(); routes
            the .md/.txt body through _build_attachment_content so --attach-as-file
            actually takes effect (inlined text when the flag is off)."""
            return lambda doc_text: _build_attachment_content(
                doc_text, filename, args.attach_as_file
            )

        if input_path.is_file():
            input_files = [input_path]
        else:
            input_files = sorted(
                p
                for p in input_path.iterdir()
                if p.suffix.lower() in _DOC_INPUT_EXTENSIONS
                or p.suffix.lower() in DOC_CONVERT_EXTENSIONS
                or p.suffix.lower() == ".csv"
                or p.name.lower().endswith(".teitok.xml")
            )

        total_processed = total_errors = total_aborted = 0
        for f in tqdm(input_files, desc="Documents", unit="doc", dynamic_ncols=True):
            doc_id = f.stem
            out_file = output_dir / f"{doc_id}_enriched.json"
            if out_file.exists():
                logger.log_skip(f.name, "already_exists")
                continue

            # .pdf/.docx → visually-rich .md (document-level) before dispatch.
            source_file = f
            try:
                f = prepare_document_input(f, ocr=args.ocr)
            except Exception as exc:
                tqdm.write(f"  [skip] {f.name}: conversion failed ({exc})")
                logger.log_skip(f.name, f"conversion_failed: {exc}")
                continue

            try:
                is_document_level = f.suffix.lower() in _DOC_INPUT_EXTENSIONS
                if is_document_level:
                    results, stats = run_document_level(
                        f,
                        doc_chat_fn,
                        doc_prompt,
                        DocModel,
                        user_content_builder=_make_doc_builder(f.name),
                    )
                else:
                    results, stats = run_line_level(
                        f,
                        line_chat_fn,
                        line_prompt,
                        LineModel,
                        include_non_text=include_non_text,
                        min_char_count=min_char_count,
                        min_char_non_text=min_char_non_text,
                        min_alpha_ratio_non_text=min_alpha_ratio_non_text,
                    )

                total_processed += stats["processed"]
                total_errors += stats["skipped_error"]
                total_aborted += int(bool(stats.get("aborted")))

                if results:
                    with open(out_file, "w", encoding="utf-8") as out_f:
                        json.dump(results, out_f, indent=4, ensure_ascii=False)
                    tqdm.write(f"  -> {len(results)} records -> {out_file.name}")
                    logger.log_success("json", count=1)
                    logger.log_document_success()

                    if args.document_json_dir is not None:
                        write_document_record(
                            doc_id,
                            results,
                            args.document_json_dir,
                            run_id=logger.run_id,
                            paradata_ref=os.path.join(
                                logger.paradata_dir, f"{logger.run_id}_{logger.program}.json"
                            ),
                            enriched_path=out_file,
                            # Only a real conversion leaves a regenerable derivation.
                            markdown_from=source_file if f != source_file else None,
                            used_markdown_input=is_document_level,
                            license_detail=logger.get_license_block(),
                        )
                else:
                    logger.log_skip(f.name, "No records produced.")

            except Exception as exc:
                tqdm.write(f"  Critical error on {f.name}: {exc}")
                logger.log_skip(f.name, str(exc))

        print(
            f"\n=== Run complete ===\n"
            f"    records enriched:  {total_processed}\n"
            f"    inference errors:  {total_errors}\n"
            f"    aborted documents: {total_aborted}\n"
            f"    files processed:   {len(input_files)}\n"
        )
        logger.finalize(input_total=len(input_files))


if __name__ == "__main__":
    main()
